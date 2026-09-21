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
                model, tok, classification_type="single-label", device=device)
        except Exception:
            self.pipe = ZeroShotClassificationPipeline(
                model, tok, classification_type="single-label", device="cpu")
        self.revision = getattr(model.config, "_commit_hash", None) or "unpinned"
        self.latency_mode = "bs1"

    def decide(self, text, options, hypothesis_template=None, question=None):
        t0 = time.perf_counter()
        res = self.pipe(text, list(options), threshold=0.0)[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        sb = {r["label"]: float(r["score"]) for r in res}
        probs = [sb.get(o, 0.0) for o in options]
        s = sum(probs) or 1.0
        return [p / s for p in probs], latency_ms


class LayaChoice:
    def __init__(self, model_id="convaiinnovations/laya"):
        import laya

        self.model_id = model_id
        self.agent = laya.load(model_id)
        self.revision = "unpinned"
        self.latency_mode = "bs1"

    def set_head_budget(self, n_tokens):
        try:
            self.agent.cfg["head_max_len"] = n_tokens
            return True
        except Exception:
            return False

    def decide(self, text, options, hypothesis_template=None, question=None):
        t0 = time.perf_counter()
        q = {"label": {"type": "choice",
                       "instructions": question or "Which option best describes this text?",
                       "criteria": {o: o for o in options}}}
        res = self.agent.predict({"text": text}, q)
        latency_ms = (time.perf_counter() - t0) * 1000
        ans = res["answers"]["label"]
        dist = None
        for key in ("probabilities", "distribution", "probs"):
            if isinstance(ans.get(key), dict):
                dist = ans[key]
                break
        if dist is not None:
            probs = [float(dist.get(o, 0.0)) for o in options]
        else:  # degenerate fallback so a schema surprise is visible, not fatal
            probs = [1.0 if o == ans.get("choice") else 0.0 for o in options]
        s = sum(probs) or 1.0
        return [p / s for p in probs], latency_ms
