"""Run the Jev arm against the frozen dataset. Paid, un-rerunnable, resumable.

Usage:
  .venv/bin/python -m harness.run.jev --rq rq3 [--cap 2.00] [--workers 5] [--limit N]
  .venv/bin/python -m harness.run.jev --rq rq2 --det --cap 0.60   # determinism control
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from ..dataset import load_rq
from ..models.jev import JevClient, SpendCapExceeded

OUT = pathlib.Path("results/jev")
QUESTION = "Which label applies here?"
N_ORDERS = 5
ORDER_KS = [16, 64]
DET_KS = [64]        # determinism control runs only where the headline lives
MAX_K = 255          # API cap


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def options_for(item, tier, K, seed=101, perm=None):
    d = item["distractors"][tier]
    if len(d) < K - 1:
        return None
    opts = [item["gold"]] + d[: K - 1]
    tag = f"order|{item['uid']}|{tier}|{K}|{perm}" if perm is not None \
        else f"{seed}|{item['uid']}|{tier}|{K}"
    random.Random(tag).shuffle(opts)
    return opts


def plan(items, spec, rq, det=False):
    """Every call this run needs, as (key, item, tier, K, perm).

    det=True emits ONLY the determinism control: same item, same option set,
    same ORDER (permutation 0, so it sits inside the distribution the RQ2
    headline was measured over), five separate calls. Whatever flips there is
    the API disagreeing with itself. It is a noise floor, not a term to
    subtract — noise and order can flip the same item."""
    jobs = []
    for it in items:
        for tier in spec["tiers"]:
            if det:
                for K in DET_KS:
                    if options_for(it, tier, K) is None:
                        continue
                    for s in range(N_ORDERS):
                        jobs.append((f"{rq}|det|{it['uid']}|{tier}|{K}|{s}",
                                     it, tier, K, 0))
                continue
            for K in spec["k_grid"]:
                if K > MAX_K:
                    continue
                if options_for(it, tier, K) is None:
                    continue
                jobs.append((f"{rq}|acc|{it['uid']}|{tier}|{K}", it, tier, K, None))
            if rq == "rq2":
                for K in ORDER_KS:
                    if options_for(it, tier, K) is None:
                        continue
                    for s in range(N_ORDERS):
                        jobs.append((f"{rq}|ord|{it['uid']}|{tier}|{K}|{s}", it, tier, K, s))
    return jobs


def main():
    a = sys.argv
    rq = a[a.index("--rq") + 1] if "--rq" in a else "rq3"
    cap = float(a[a.index("--cap") + 1]) if "--cap" in a else 2.00
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 5
    limit = int(a[a.index("--limit") + 1]) if "--limit" in a else None

    items, spec = load_rq(rq)
    if limit:
        by = defaultdict(list)
        for it in items:
            by[it["domain"]].append(it)
        items = [x for v in by.values() for x in v[:limit]]

    OUT.mkdir(parents=True, exist_ok=True)
    client = JevClient(OUT / f"{rq}_ledger.jsonl", spend_cap=cap, log=log)
    jobs = plan(items, spec, rq, det="--det" in a)
    todo = [j for j in jobs if j[0] not in client.done]
    est = sum((284 + 13 * j[3] + len(j[1]["text"]) / 4) * 0.042e-6 for j in todo)
    log(f"{rq}: {len(items)} items -> {len(jobs)} calls, {len(todo)} unpaid "
        f"(est ${est:.3f}, cap ${cap:.2f}, {workers} workers)")
    if not todo:
        log("nothing to do")
        return

    t0, done_n, lock_fail = time.time(), 0, []

    def work(job):
        key, it, tier, K, perm = job
        opts = options_for(it, tier, K, perm=perm)
        probs, rec = client.decide(key, it["text"], opts, QUESTION)
        return key, it, tier, K, perm, opts, probs, rec

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for key, it, tier, K, perm, opts, probs, rec in ex.map(work, todo):
                done_n += 1
                if probs is None:
                    lock_fail.append((key, rec.get("error", "?")[:60]))
                if done_n % 500 == 0 or done_n == len(todo):
                    el = time.time() - t0
                    rate = done_n / el
                    log(f"  {done_n}/{len(todo)} calls  ${client.spent:.4f}  "
                        f"{rate:.1f}/s  eta {(len(todo)-done_n)/rate/60:.1f}m  "
                        f"fails {len(lock_fail)}")
    except SpendCapExceeded as e:
        log(f"STOPPED: {e}")
    finally:
        client.close()

    log(f"spent ${client.spent:.4f} over {client.n_calls} ledger records; "
        f"{len(lock_fail)} failed calls")
    for k, e in lock_fail[:5]:
        log(f"  fail {k}: {e}")


if __name__ == "__main__":
    main()
