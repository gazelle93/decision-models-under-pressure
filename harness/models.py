"""Model adapters. v0: zero-shot NLI cross-encoders via the HF pipeline
(pass-per-label cost shape). GLiClass, Laya, and SetFit adapters are the
next baselines per the brief's phased build.
"""
from __future__ import annotations

import time


class ZeroShotNLI:
    """Per-label entailment, softmax over the request's options."""

    def __init__(self, model_id, device=None):
        from transformers import pipeline

        self.model_id = model_id
        if device is None:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = pipeline("zero-shot-classification", model=model_id, device=device)
        self.revision = getattr(self.pipe.model.config, "_commit_hash", None) or "unpinned"

    def decide(self, text, options, hypothesis_template):
        t0 = time.perf_counter()
        out = self.pipe(text, candidate_labels=options,
                        hypothesis_template=hypothesis_template, multi_label=False)
        latency_ms = (time.perf_counter() - t0) * 1000
        score_by_label = dict(zip(out["labels"], out["scores"]))
        probs = [score_by_label[o] for o in options]
        return probs, latency_ms
