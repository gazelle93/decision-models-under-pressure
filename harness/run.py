"""Run one model over one registry dataset; write per-decision JSONL + metrics JSON.

Usage:
  .venv/bin/python -m harness.run --model MoritzLaurer/deberta-v3-base-zeroshot-v2.0 \
      --dataset enron_spam --limit 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

from . import metrics
from .models import ZeroShotNLI
from .registry import REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    spec = REGISTRY[args.dataset]
    if spec.status != "ready":
        raise SystemExit(f"{spec.key} is not wired yet (status={spec.status})")

    examples = spec.loader(spec, args.limit)
    model = ZeroShotNLI(args.model, device=args.device)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = f"{spec.key}__{args.model.split('/')[-1]}__{stamp}"
    jsonl_path = outdir / f"{run_id}.jsonl"

    records = []
    with jsonl_path.open("w") as f:
        for i, ex in enumerate(examples):
            probs, latency_ms = model.decide(ex.text, ex.options, spec.hypothesis_template)
            rec = {
                "i": i,
                "text_sha1": hashlib.sha1(ex.text.encode()).hexdigest()[:12],
                "options": ex.options,
                "probs": [round(p, 6) for p in probs],
                "pred": int(max(range(len(probs)), key=probs.__getitem__)),
                "gold": ex.gold,
                "latency_ms": round(latency_ms, 2),
            }
            records.append(rec)
            f.write(json.dumps(rec) + "\n")

    summary = metrics.summarize(records)
    summary.update({
        "run_id": run_id, "dataset": spec.key, "hf_id": spec.hf_id, "split": spec.split,
        "model": args.model, "model_revision": model.revision,
        "hypothesis_template": spec.hypothesis_template, "limit": args.limit, "notes": spec.notes,
    })
    (outdir / f"{run_id}.metrics.json").write_text(json.dumps(summary, indent=2))
    keys = ["n", "accuracy", "macro_f1", "brier", "nll", "ece_15bin",
            "latency_ms_p50", "latency_ms_p95"]
    print(json.dumps({k: summary[k] for k in keys}, indent=2))
    print("wrote", jsonl_path)


if __name__ == "__main__":
    main()
