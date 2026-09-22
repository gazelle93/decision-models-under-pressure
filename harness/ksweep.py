"""K-sweep pilot (brief Section 8.2): nested distractors over a real label
universe, CLINC gold items, K in {2..256}. n=50 items -> a pilot for curve
shape, not headline numbers (flagged). K=512/1024 wait on a larger universe.

Usage: .venv/bin/python -m harness.ksweep
"""
from __future__ import annotations

import json
import pathlib
import random
import time
import traceback

from .registry import _clean

KS = [2, 4, 8, 16, 32, 64, 128, 256]
N_ITEMS = 50
FLIP_K = 16
FLIP_SHUFFLES = 5
OUT = pathlib.Path("results")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_universe():
    from datasets import load_dataset

    opts = set()
    def add(names): opts.update(_clean(n) for n in names)

    clinc = load_dataset("clinc/clinc_oos", "plus", split="test")
    clinc_names = [_clean(n) for n in clinc.features["intent"].names if n != "oos"]
    add(clinc_names)
    b77 = load_dataset("mteb/banking77", split="test")
    add({r["label_text"] for r in b77})
    try:
        mas = load_dataset("AmazonScience/massive", "en-US", split="test")
        add(mas.features["intent"].names)
    except Exception:
        log("massive skipped:\n" + traceback.format_exc(limit=1))
    try:
        ge = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
        add(ge.features["labels"].feature.names)
    except Exception:
        log("go_emotions skipped")
    try:
        db = load_dataset("fancyzhx/dbpedia_14", split="test")
        add(db.features["label"].names)
    except Exception:
        log("dbpedia skipped")
    from .registry import TWITTER_FIN_TOPICS
    add(TWITTER_FIN_TOPICS)
    return sorted(opts), clinc, clinc_names


def load_universe_v1():
    """Frozen Phase-A universe: canonical options + conflict matrix.
    Returns (options, conflicts_by_option, alias_of) or None if not frozen yet."""
    import json
    f = pathlib.Path("results/universe_v1.json")
    if not f.exists():
        return None
    v1 = json.loads(f.read_text())
    conf = {}
    for a, b in v1["conflicts"]:
        conf.setdefault(a, set()).add(b)
        conf.setdefault(b, set()).add(a)
    alias_of = {al: canon for canon, als in v1["merges"].items() for al in als}
    return sorted(v1["options"]), conf, alias_of


def build_items(clinc, clinc_names, universe, conflicts=None, alias_of=None, text_excl_m=10):
    rows = [r for r in clinc if clinc.features["intent"].names[r["intent"]] != "oos"]
    rng = random.Random(7)
    picked = rng.sample(rows, N_ITEMS)

    text_top = None
    if conflicts is not None and text_excl_m:
        # Phase-B per-item exclusion: drop each text's top-M nearest options
        # (filter model = mpnet, NOT in the eval roster).
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        fm = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                 device="mps" if torch.backends.mps.is_available() else "cpu")
        opt_emb = np.asarray(fm.encode(universe, normalize_embeddings=True, batch_size=64))
        txt_emb = np.asarray(fm.encode([r["text"] for r in picked],
                                       normalize_embeddings=True, batch_size=64))
        sims = txt_emb @ opt_emb.T
        text_top = [
            {universe[j] for j in row.argsort()[-text_excl_m:]} for row in sims
        ]

    items = []
    for i, r in enumerate(picked):
        gold = _clean(clinc.features["intent"].names[r["intent"]])
        if alias_of:
            gold = alias_of.get(gold, gold)
        excluded = {gold}
        if conflicts is not None:
            excluded |= conflicts.get(gold, set())
        if text_top is not None:
            excluded |= (text_top[i] - {gold})
        pool = [o for o in universe if o not in excluded]
        distractors = random.Random(1000 + i).sample(pool, max(KS) - 1)  # nested prefix
        items.append({"text": r["text"], "gold": gold, "distractors": distractors,
                      "n_excluded": len(excluded) - 1})
    return items


def options_for(item, K, order_seed=None):
    """Default order seed is a stable CRC of the gold string, NOT Python hash()
    (which is salted per process and broke run-to-run reproducibility in the
    2026-09-22 pilot; orders were consistent within that run only)."""
    import zlib
    opts = [item["gold"]] + item["distractors"][: K - 1]
    seed = order_seed if order_seed is not None else zlib.crc32(item["gold"].encode()) % 10**6 + K
    random.Random(seed).shuffle(opts)
    return opts


QUESTION = "What is the intent or category of this text?"
TEMPLATE = "The intent or category of this text is {}."


def run_model(name, adapter, items):
    import math
    per_k = {}
    for K in KS:
        recs, fails = [], 0
        t_lat = []
        for i, item in enumerate(items):
            opts = options_for(item, K)
            gold_idx = opts.index(item["gold"])
            try:
                probs, ms = adapter.decide(item["text"], opts, TEMPLATE, question=QUESTION)
            except Exception:
                fails += 1
                continue
            top5 = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)[:5]
            recs.append({
                "acc": 1.0 if max(range(len(probs)), key=probs.__getitem__) == gold_idx else 0.0,
                "top5": 1.0 if gold_idx in top5 else 0.0,
                "nll": -math.log(max(probs[gold_idx], 1e-12)),
            })
            t_lat.append(ms)
        if fails == len(items):
            per_k[K] = {"infeasible": True}
            log(f"{name} K={K}: INFEASIBLE for all items")
            continue
        n = len(recs)
        t_lat.sort()
        per_k[K] = {
            "n": n, "fails": fails,
            "acc": round(sum(r["acc"] for r in recs) / n, 3),
            "top5": round(sum(r["top5"] for r in recs) / n, 3),
            "nll": round(sum(r["nll"] for r in recs) / n, 3),
            "p50_ms": round(t_lat[n // 2], 1),
        }
        log(f"{name} K={K}: acc {per_k[K]['acc']} top5 {per_k[K]['top5']} "
            f"nll {per_k[K]['nll']} p50 {per_k[K]['p50_ms']}ms fails {fails}")
    # permutation flip test at FLIP_K
    flips, total = 0, 0
    for i, item in enumerate(items):
        preds = []
        for s in range(FLIP_SHUFFLES):
            opts = options_for(item, FLIP_K, order_seed=5000 + i * 10 + s)
            try:
                probs, _ = adapter.decide(item["text"], opts, TEMPLATE, question=QUESTION)
            except Exception:
                continue
            preds.append(opts[max(range(len(probs)), key=probs.__getitem__)])
        if len(preds) >= 2:
            total += 1
            if len(set(preds)) > 1:
                flips += 1
    per_k["flip_rate_K16"] = round(flips / total, 3) if total else None
    log(f"{name} flip-rate@K16 over {FLIP_SHUFFLES} orders: {per_k['flip_rate_K16']}")
    return per_k


def main():
    import sys
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    OUT.mkdir(exist_ok=True)
    v1 = load_universe_v1()
    if v1:
        universe, conflicts, alias_of = v1
        from datasets import load_dataset
        clinc = load_dataset("clinc/clinc_oos", "plus", split="test")
        clinc_names = [_clean(n) for n in clinc.features["intent"].names if n != "oos"]
        log(f"universe_v1 (frozen): {len(universe)} canonical options, "
            f"{sum(len(v) for v in conflicts.values()) // 2} conflict pairs")
        items = build_items(clinc, clinc_names, universe, conflicts, alias_of)
        excl = [it["n_excluded"] for it in items]
        log(f"per-item exclusions (conflicts + text top-10): "
            f"min {min(excl)} / median {sorted(excl)[len(excl)//2]} / max {max(excl)}")
    else:
        universe, clinc, clinc_names = build_universe()
        log(f"universe size: {len(universe)} unique option strings (DRAFT, no conflict matrix)")
        items = build_items(clinc, clinc_names, universe)

    from .models import EmbeddingSim, GLiClassZS, LayaChoice, ZeroShotNLI
    results = {"universe_size": len(universe), "n_items": N_ITEMS, "Ks": KS,
               "note": "pilot n=50; curve shape only, not headline numbers", "models": {}}

    factories = {
        "bge-large-en-v1.5": lambda: EmbeddingSim(),
        "laya": lambda: (lambda a: (a.set_budgets(2048, 1024), a)[1])(LayaChoice()),
        "gliclass-large-v3.0": lambda: GLiClassZS(),
        "deberta-v3-base-zeroshot-v2.0": lambda: ZeroShotNLI("MoritzLaurer/deberta-v3-base-zeroshot-v2.0"),
    }
    for name, factory in factories.items():
        log(f"=== {name}")
        try:
            adapter = factory()
        except Exception:
            log(f"LOAD FAILED {name}\n{traceback.format_exc(limit=3)}")
            results["models"][name] = {"error": "load_failed"}
            continue
        try:
            results["models"][name] = run_model(name, adapter, items)
        except Exception:
            log(f"RUN FAILED {name}\n{traceback.format_exc(limit=3)}")
            results["models"][name] = {"error": "run_failed"}
        del adapter
        (OUT / f"ksweep_pilot{suffix}.json").write_text(json.dumps(results, indent=2))
    log("KSWEEP DONE")


if __name__ == "__main__":
    main()
