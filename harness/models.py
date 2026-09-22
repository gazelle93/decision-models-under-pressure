"""Model adapters.

- ZeroShotNLI: per-label entailment cross-encoders (pass-per-label cost shape).
  decide() = honest bs=1 latency; decide_many() = batched accuracy sweeps
  (latency reported as batched-avg and flagged).
- GLiClassZS: single-pass label-token architecture (true per-example latency).
- LayaChoice: Laya's choice primitive via the pip package (true per-example
  latency). Full distribution read from the answer's probability field.
"""
from __future__ import annotations

import time


class ZeroShotNLI:
    def __init__(self, model_id, device=None):
        from transformers import pipeline

        self.model_id = model_id
        if device is None:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = pipeline("zero-shot-classification", model=model_id, device=device)
        self.revision = getattr(self.pipe.model.config, "_commit_hash", None) or "unpinned"
        self.latency_mode = "bs1"

    def decide(self, text, options, hypothesis_template, question=None):
        t0 = time.perf_counter()
        out = self.pipe(text, candidate_labels=options,
                        hypothesis_template=hypothesis_template, multi_label=False)
        latency_ms = (time.perf_counter() - t0) * 1000
        score_by_label = dict(zip(out["labels"], out["scores"]))
        return [score_by_label[o] for o in options], latency_ms

    def decide_many(self, texts, options, hypothesis_template, question=None, batch_size=16):
        t0 = time.perf_counter()
        outs = self.pipe(list(texts), candidate_labels=options,
                         hypothesis_template=hypothesis_template,
                         multi_label=False, batch_size=batch_size)
        avg_ms = (time.perf_counter() - t0) * 1000 / len(texts)
        if isinstance(outs, dict):
            outs = [outs]
        res = []
        for out in outs:
            sb = dict(zip(out["labels"], out["scores"]))
            res.append([sb[o] for o in options])
        return res, avg_ms


class GLiClassZS:
    def __init__(self, model_id="knowledgator/gliclass-large-v3.0", device=None):
        import torch
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer

        self.model_id = model_id
        model = GLiClassModel.from_pretrained(model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            self.pipe = ZeroShotClassificationPipeline(
                model, tok, classification_type="multi-label", device=device)
        except Exception:
            self.pipe = ZeroShotClassificationPipeline(
                model, tok, classification_type="multi-label", device="cpu")
        self.revision = getattr(model.config, "_commit_hash", None) or "unpinned"
        self.latency_mode = "bs1"

    def decide(self, text, options, hypothesis_template=None, question=None):
        """Post-review fix: the single-label pipeline returns only the argmax,
        which reached the harness as a one-hot and invalidated every
        calibration number (review #1). The multi-label path sigmoids the SAME
        per-label logits and returns all of them at threshold 0, so inverting
        the sigmoid and softmaxing reconstructs the exact single-label
        distribution from raw scores."""
        import math
        t0 = time.perf_counter()
        res = self.pipe(text, list(options), threshold=0.0)[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        sb = {r["label"]: float(r["score"]) for r in res}
        eps = 1e-9
        logits = []
        for o in options:
            s = min(max(sb.get(o, eps), eps), 1 - eps)
            logits.append(math.log(s / (1 - s)))
        mx = max(logits)
        e = [math.exp(z - mx) for z in logits]
        tot = sum(e)
        return [v / tot for v in e], latency_ms


class LayaChoice:
    def __init__(self, model_id="convaiinnovations/laya"):
        import laya

        self.model_id = model_id
        self.agent = laya.load(model_id)
        self.default_max_len = self.agent.cfg.get("max_len", 512)
        self.default_head_max_len = self.agent.cfg.get("head_max_len", 192)
        self.revision = "unpinned"
        self.latency_mode = "bs1"

    def set_budgets(self, max_len=None, head_max_len=None):
        """Raise context budgets for many-option questions, per the Laya repo's
        own Banking77 guidance. Decision quality at extended lengths is
        unvalidated by the vendor; runs using this are flagged in notes."""
        try:
            if max_len is not None:
                self.agent.cfg["max_len"] = max_len
            if head_max_len is not None:
                self.agent.cfg["head_max_len"] = head_max_len
            return True
        except Exception:
            return False

    def decide(self, text, options, hypothesis_template=None, question=None):
        """Post-review fix: agent.predict() rounds probabilities to 4 decimals
        (laya/agent.py), which manufactured sparse support and NLL-floor
        artifacts (review #1). This replicates system_one's forward using
        laya's own building blocks and returns the unrounded shipped-
        temperature distribution. Gold always reaches the model: markers are
        built for every option or build_sequence raises."""
        import numpy as np
        import torch
        from laya.common import QTYPES, build_sequence, collate_items, render_options
        try:
            from laya.common import temp_bucket
        except ImportError:
            from laya.agent import temp_bucket

        agent = self.agent
        t0 = time.perf_counter()
        qi = agent._to_internal({"type": "choice",
                                 "instructions": question or "Which option best describes this text?",
                                 "criteria": {o: o for o in options}})
        max_len = agent.cfg.get("max_len", 512)
        hml = agent.cfg.get("head_max_len", 192)
        seq, markers = build_sequence(agent.tok, {"text": text}, qi, max_len, hml)
        if len(markers) != len(render_options(qi)):
            raise ValueError(f"options exceed head_max_len={hml}")
        items = [{"ids": seq, "markers": markers, "qtype": QTYPES["choice"]}]
        b = collate_items([items], agent.tok.pad_token_id)
        with torch.no_grad():
            logits, act = agent.model(
                b["input_ids"].to(agent.device), b["attention_mask"].to(agent.device),
                b["marker_pos"].to(agent.device), b["marker_mask"].to(agent.device),
                b["qtype"].to(agent.device))
        latency_ms = (time.perf_counter() - t0) * 1000
        z_raw = logits.float().cpu().numpy()[0, :len(options)]
        t_scale = agent.temperature_by_options.get(
            temp_bucket(QTYPES["choice"], len(options)), agent.temperature[QTYPES["choice"]])
        z = z_raw / t_scale
        e = np.exp(z - z.max())
        p = e / e.sum()
        self.last_raw_logits = [round(float(v), 6) for v in z_raw]
        return [float(v) for v in p], latency_ms


class EmbeddingSim:
    """A2: bi-encoder label-similarity zero-shot. Options verbalized with the
    dataset's hypothesis template, embedded, cosine-scored against the text,
    softmax at a fixed scale. Uncalibrated by construction; the temperature
    stage refits it like everything else."""

    def __init__(self, model_id="BAAI/bge-large-en-v1.5", device=None, scale=20.0):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.scale = scale
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.m = SentenceTransformer(model_id, device=device)
        self.revision = "unpinned"
        self.latency_mode = "bs1"
        self._opt_cache = {}

    def _softmax(self, sims):
        import math
        z = [s * self.scale for s in sims]
        mx = max(z)
        e = [math.exp(v - mx) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def _opts(self, options, template):
        key = (tuple(options), template)
        if key not in self._opt_cache:
            texts = [template.format(o) for o in options]
            self._opt_cache[key] = self.m.encode(texts, normalize_embeddings=True)
        return self._opt_cache[key]

    def decide(self, text, options, hypothesis_template, question=None):
        import time as _t
        t0 = _t.perf_counter()
        opt = self._opts(options, hypothesis_template)
        v = self.m.encode([text], normalize_embeddings=True)[0]
        probs = self._softmax(list(opt @ v))
        return probs, (_t.perf_counter() - t0) * 1000

    def decide_many(self, texts, options, hypothesis_template, question=None, batch_size=64):
        import time as _t
        t0 = _t.perf_counter()
        opt = self._opts(options, hypothesis_template)
        vs = self.m.encode(list(texts), normalize_embeddings=True, batch_size=batch_size)
        res = [self._softmax(list(opt @ v)) for v in vs]
        return res, (_t.perf_counter() - t0) * 1000 / len(texts)
