"""Wave-1 baseline sweep (brief Section 9, phased build step 1).

Fixed seeded subsets per dataset; accuracy runs are batched for NLI models
(latency flagged batched-avg) with a separate bs=1 latency probe on the
2/20/77-label trio; GLiClass and Laya run per-example, so their latencies
are honest bs=1 numbers. Every (model, dataset) pair is independent: one
failure logs and the sweep continues.

Usage: .venv/bin/python -m harness.sweep
"""
from __future__ import annotations

import gc
import json
import pathlib
import time
import traceback

from . import metrics
from .registry import REGISTRY

SEED = 42
CAPS = {"enron_spam": 500, "daily_dialog_emotion": 500, "twitter_fin_topic": 500,
        "sst5": 500, "clinc150_oos": 300, "banking77": 300}
PROBE_SETS = ["enron_spam", "twitter_fin_topic", "banking77"]  # 2 / 20 / 77 labels
PROBE_N = 25
BATCH = 16

OUT = pathlib.Path("results")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_nli(model_id):
    from .models import ZeroShotNLI
    return ZeroShotNLI(model_id)


MODELS = {
    "deberta-v3-base-zeroshot-v2.0": lambda: make_nli("MoritzLaurer/deberta-v3-base-zeroshot-v2.0"),
    "gliclass-large-v3.0": lambda: __import__("harness.models", fromlist=["GLiClassZS"]).GLiClassZS("knowledgator/gliclass-large-v3.0"),
    "laya": lambda: __import__("harness.models", fromlist=["LayaChoice"]).LayaChoice("convaiinnovations/laya"),
    "deberta-v3-large-zeroshot-v2.0": lambda: make_nli("MoritzLaurer/deberta-v3-large-zeroshot-v2.0"),
    "bart-large-mnli": lambda: make_nli("facebook/bart-large-mnli"),
    "bge-large-en-v1.5": lambda: __import__("harness.models", fromlist=["EmbeddingSim"]).EmbeddingSim(),
    "gte-large": lambda: __import__("harness.models", fromlist=["EmbeddingSim"]).EmbeddingSim("thenlper/gte-large"),
}


def run_pair(adapter, model_name, spec, examples):
    texts = [e.text for e in examples]
    options = examples[0].options
    if hasattr(adapter, "decide_many"):
        probs_list, avg_ms = adapter.decide_many(
            texts, options, spec.hypothesis_template, question=spec.question, batch_size=BATCH)
        lats = [avg_ms] * len(texts)
        lat_mode = "batched-avg"
    else:
        if hasattr(adapter, "set_budgets") and len(options) > 20:
            adapter.set_budgets(max_len=2048, head_max_len=1024)
        probs_list, lats = [], []
        for e in examples:
            p, ms = adapter.decide(e.text, e.options, spec.hypothesis_template, question=spec.question)
            probs_list.append(p)
            lats.append(ms)
        lat_mode = getattr(adapter, "latency_mode", "bs1")
        if hasattr(adapter, "set_budgets") and len(options) > 20:
            adapter.set_budgets(max_len=adapter.default_max_len,
                                head_max_len=adapter.default_head_max_len)
            lat_mode = "bs1-extended-ctx"

    records = [{"i": i, "probs": [round(x, 6) for x in p], "gold": ex.gold,
                "latency_ms": round(l, 2)}
               for i, (p, ex, l) in enumerate(zip(probs_list, examples, lats))]
    summary = metrics.summarize(records)
    summary.update({
        "dataset": spec.key, "hf_id": spec.hf_id, "split": spec.split,
        "model": model_name, "model_revision": getattr(adapter, "revision", "unpinned"),
        "seed": SEED, "latency_mode": lat_mode, "notes": spec.notes,
    })
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{spec.key}__{model_name}__{stamp}"
    with (OUT / f"{run_id}.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (OUT / f"{run_id}.metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def latency_probe(adapter, model_name):
    """Honest bs=1 latency for pass-per-label models on the 2/20/77-label trio."""
    probe = {}
    for key in PROBE_SETS:
        spec = REGISTRY[key]
        examples = spec.loader(spec, PROBE_N, SEED)
        lats = []
        for e in examples:
            _, ms = adapter.decide(e.text, e.options, spec.hypothesis_template, question=spec.question)
            lats.append(ms)
        lats.sort()
        probe[key] = {"n_labels": len(examples[0].options), "n": len(lats),
                      "p50_ms": round(lats[len(lats) // 2], 1), "max_ms": round(lats[-1], 1)}
        log(f"  probe {model_name}/{key}: p50 {probe[key]['p50_ms']}ms @ {probe[key]['n_labels']} labels")
    (OUT / f"latencyprobe__{model_name}.json").write_text(json.dumps(probe, indent=2))


def have_result(key, model_name):
    return any(OUT.glob(f"{key}__{model_name}__*.metrics.json"))


def main():
    import sys
    resume = "--resume" in sys.argv
    only = None
    if "--models" in sys.argv:
        only = set(sys.argv[sys.argv.index("--models") + 1].split(","))
    OUT.mkdir(exist_ok=True)
    all_summaries = []
    for model_name, factory in MODELS.items():
        if only and model_name not in only:
            continue
        log(f"=== loading {model_name}")
        try:
            adapter = factory()
        except Exception:
            log(f"LOAD FAILED {model_name}\n{traceback.format_exc()}")
            all_summaries.append({"model": model_name, "error": "load_failed"})
            continue
        for key, cap in CAPS.items():
            spec = REGISTRY[key]
            if resume and have_result(key, model_name):
                log(f"skip {model_name}/{key} (resume)")
                continue
            try:
                t0 = time.time()
                examples = spec.loader(spec, cap, SEED)
                s = run_pair(adapter, model_name, spec, examples)
                all_summaries.append(s)
                log(f"{model_name} / {key}: acc {s['accuracy']} f1 {s['macro_f1']} "
                    f"ece {s['ece_15bin']} ({int(time.time() - t0)}s)")
            except Exception:
                log(f"FAILED {model_name}/{key}\n{traceback.format_exc()}")
                all_summaries.append({"model": model_name, "dataset": key, "error": "run_failed"})
        if hasattr(adapter, "decide_many") and not (resume and (OUT / f"latencyprobe__{model_name}.json").exists()):
            try:
                latency_probe(adapter, model_name)
            except Exception:
                log(f"probe failed for {model_name}\n{traceback.format_exc()}")
        del adapter
        gc.collect()
        try:
            import torch
            torch.mps.empty_cache()
        except Exception:
            pass

    # rebuild the summary from every metrics file (keep latest per pair)
    latest = {}
    for f in sorted(OUT.glob("*.metrics.json")):
        s = json.loads(f.read_text())
        latest[(s["dataset"], s["model"])] = s
    all_summaries = list(latest.values())
    (OUT / "wave1_summary.json").write_text(json.dumps(all_summaries, indent=2))
    log("SWEEP DONE")
    cols = ["model", "dataset", "n", "accuracy", "macro_f1", "brier", "nll", "ece_15bin",
            "latency_ms_p50", "latency_mode"]
    for s in all_summaries:
        print(" | ".join(str(s.get(c, "-")) for c in cols))


if __name__ == "__main__":
    main()
