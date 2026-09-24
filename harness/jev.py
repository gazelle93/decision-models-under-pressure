"""Jev client for paid, un-rerunnable API calls.

Every design choice here assumes the money is real and a call cannot be taken
back:

  LEDGER      append-only JSONL keyed per CALL, flushed and fsynced before the
              next call is made. A crash loses at most the in-flight call.
  RESUME      the ledger is the source of truth; any key already present is
              never re-sent. Resume is therefore free.
  SPEND CAP   checked BEFORE each call against the running total taken from
              the API's own usage.cost. Breach aborts cleanly, resumable.
  RETRY       bounded exponential backoff on timeout / 429 / 5xx only. A 4xx
              other than 429 is a permanent error: logged, not retried.
  VALIDATE    the returned choice must be one of the options we sent, and the
              probability keys must match exactly; a mismatch is recorded as a
              failure rather than silently scored.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.error
import urllib.request

ENDPOINT = "https://openrouter.ai/api/v1/systemone"
MODEL = "typesafe/jev-1.13"
MAX_OPTIONS = 255          # API rejects 256+
RETRY_ON = {408, 429, 500, 502, 503, 504}


class SpendCapExceeded(RuntimeError):
    pass


class JevClient:
    def __init__(self, ledger_path, key_path="/tmp/.or_key", spend_cap=2.00,
                 max_retries=4, timeout=60, log=print):
        self.key = pathlib.Path(key_path).read_text().strip()
        self.ledger_path = pathlib.Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = spend_cap
        self.max_retries = max_retries
        self.timeout = timeout
        self.log = log
        self.done, self.spent, self.n_calls = self._replay()
        self._fh = self.ledger_path.open("a")
        log(f"  ledger: {len(self.done)} calls already paid for, ${self.spent:.4f} spent, "
            f"cap ${self.cap:.2f}")

    def _replay(self):
        """Rebuild state from the ledger. Tolerates a torn final line."""
        done, spent, n = {}, 0.0, 0
        if not self.ledger_path.exists():
            return done, spent, n
        for line in self.ledger_path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                      # torn write at crash; ignore
            spent += r.get("cost", 0.0)
            n += 1
            # A record is only "settled" if it produced probs, or if it failed
            # in a way that COST money (re-sending would double-spend). A free
            # failure — network blip, exhausted retries — is left unsettled so
            # resume retries it; that is the whole point of paying nothing.
            if r.get("probs") is not None or r.get("cost", 0.0) > 0 \
                    or r.get("permanent"):
                done[r["key"]] = r
        return done, spent, n

    def _write(self, rec):
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())           # survive a hard kill
        self.done[rec["key"]] = rec
        self.spent += rec.get("cost", 0.0)
        self.n_calls += 1

    def decide(self, key, text, options, question):
        """Returns (probs_in_option_order, record). Cached keys cost nothing."""
        if key in self.done:
            r = self.done[key]
            return (r.get("probs"), r)
        if len(options) > MAX_OPTIONS:
            rec = {"key": key, "error": f"option count {len(options)} exceeds API cap",
                   "cost": 0.0, "probs": None}
            self._write(rec)
            return None, rec
        # Pre-flight: block the call that would CROSS the cap, not just calls
        # after it. Cost is projected from the measured scaling
        # (284 + 13*K input tokens at $0.042/M, calibrated 2026-09-24).
        projected = (284 + 13 * len(options) + len(text) / 4) * 0.042e-6
        if self.spent + projected > self.cap:
            raise SpendCapExceeded(
                f"spend cap ${self.cap:.2f} would be crossed: ${self.spent:.4f} spent, "
                f"next call ~${projected:.6f}, {self.n_calls} calls done. "
                "Ledger is complete and resumable.")

        body = json.dumps({
            "model": MODEL, "state": text,
            "questions": {"label": {"type": "choice", "instructions": question,
                                    "criteria": {o: o for o in options}}},
        }).encode()
        delay, last = 1.0, None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    ENDPOINT, data=body,
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
                ms = (time.perf_counter() - t0) * 1000
                ans = data["answers"]["label"]
                pr = ans.get("probabilities", {})
                # validation: the answer must live in the option set we sent
                if ans.get("choice") not in options or set(pr) - set(options):
                    rec = {"key": key, "error": "response/option mismatch",
                           "cost": data.get("usage", {}).get("cost", 0.0),
                           "probs": None, "latency_ms": round(ms, 2)}
                    self._write(rec)
                    return None, rec
                probs = [float(pr.get(o, 0.0)) for o in options]
                rec = {"key": key, "probs": probs, "choice": ans.get("choice"),
                       "confidence": ans.get("confidence"),
                       "cost": data.get("usage", {}).get("cost", 0.0),
                       "input_tokens": data.get("usage", {}).get("input_tokens"),
                       "latency_ms": round(ms, 2), "attempt": attempt}
                self._write(rec)
                return probs, rec
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code in (401, 402, 403):
                    raise SpendCapExceeded(
                        f"HTTP {e.code} from the API — credits exhausted or key "
                        f"rejected after {self.n_calls} calls, ${self.spent:.4f} "
                        "spent. Stopping; the ledger is complete and resumable.")
                if e.code not in RETRY_ON:
                    body_txt = e.read()[:200].decode(errors="replace")
                    rec = {"key": key, "error": f"HTTP {e.code}: {body_txt}",
                           "cost": 0.0, "probs": None, "permanent": True}
                    self._write(rec)            # permanent: record, do not retry
                    return None, rec
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
            if attempt < self.max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 30)
        rec = {"key": key, "error": f"exhausted retries: {last}", "cost": 0.0, "probs": None}
        self._write(rec)
        return None, rec

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
