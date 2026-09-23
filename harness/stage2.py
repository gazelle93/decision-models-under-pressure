"""Stage-2 confirmatory runs per PREREGISTRATION.md (approved 2026-09-22).

Ladder: run with --n 200 first, then --n 500. Resume-safe: finished
(model, domain, tier) cells are skipped. Deviations: DEVIATIONS.md.

Usage: .venv/bin/python -m harness.stage2 --n 200
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import time
import traceback
import zlib

from .registry import TWITTER_FIN_TOPICS
from .universe import norm as _clean  # universe's normalizer: golds must match option keys exactly

SEED_ITEMS = 101
KS = [2, 4, 8, 16, 32, 64, 128, 256]
ORDER_KS = [16, 64]
N_ORDERS = 5
TIERS = ["far", "near", "mixed"]
LARGE_SUBSAMPLE = 150  # deberta-large prefix at K in {128, 256} (prereg)
OUT = pathlib.Path("results/stage2")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_universe():
    v2 = json.loads(pathlib.Path("results/universe_v2.json").read_text())
    options = sorted(v2["options"])
    sources = {k: set(v) for k, v in v2["options"].items()}
    conflicts = {}
    for a, b in v2["conflicts"]:
        conflicts.setdefault(a, set()).add(b)
        conflicts.setdefault(b, set()).add(a)
    alias_of = {al: c for c, als in v2["merges"].items() for al in als}
    return options, sources, conflicts, alias_of


def pilot_clinc_texts():
    """Reproduce the pilot's 50 CLINC picks (seed 7) to exclude them."""
    from datasets import load_dataset
    clinc = load_dataset("clinc/clinc_oos", "plus", split="test")
    rows = [r for r in clinc if clinc.features["intent"].names[r["intent"]] != "oos"]
    return {r["text"] for r in random.Random(7).sample(rows, 50)}, clinc


def build_domain_items(n, alias_of):
    from datasets import load_dataset

    domains = {}
    pilot, clinc = pilot_clinc_texts()
    names = clinc.features["intent"].names
    rows = [r for r in clinc if names[r["intent"]] != "oos" and r["text"] not in pilot]
    picked = random.Random(SEED_ITEMS).sample(rows, n)
    domains["clinc"] = [{"text": r["text"],
                         "gold": alias_of.get(_clean(names[r["intent"]]), _clean(names[r["intent"]])),
                         "source": "clinc150"} for r in picked]

    ge = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    ge_names = ge.features["labels"].feature.names
    rows = [r for r in ge if len(r["labels"]) == 1]
    picked = random.Random(SEED_ITEMS).sample(rows, n)
    domains["goemotions"] = [{"text": r["text"],
                              "gold": alias_of.get(_clean(ge_names[r["labels"][0]]), _clean(ge_names[r["labels"][0]])),
                              "source": "go_emotions"} for r in picked]

    ft = load_dataset("zeroshot/twitter-financial-news-topic", split="validation")
    picked = random.Random(SEED_ITEMS).sample(list(ft), n)
    domains["fintopic"] = [{"text": r["text"],
                            "gold": alias_of.get(_clean(TWITTER_FIN_TOPICS[r["label"]]), _clean(TWITTER_FIN_TOPICS[r["label"]])),
                            "source": "fin_topics"} for r in picked]
    return domains


def text_exclusions(all_items, universe):
    """Union of top-10 text-nearest options from BOTH filter models (prereg)."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    excl = [set() for _ in all_items]
    for fm_id in ("sentence-transformers/all-mpnet-base-v2", "intfloat/e5-large-v2"):
        fm = SentenceTransformer(fm_id, device=device)
        po = "passage: " if "e5" in fm_id else ""
        pt = "query: " if "e5" in fm_id else ""
        opt = np.asarray(fm.encode([po + o for o in universe], normalize_embeddings=True, batch_size=64))
        txt = np.asarray(fm.encode([pt + it["text"] for it in all_items], normalize_embeddings=True, batch_size=64))
        sims = txt @ opt.T
        for i, row in enumerate(sims):
            excl[i] |= {universe[j] for j in row.argsort()[-10:]}
        del fm
    return excl


def build_pools(domains, universe, sources, conflicts):
    """Per item, per tier: seeded nested distractor list (prefix grows with K)."""
    order = []
    for dname, items in domains.items():
        for it in items:
            order.append((dname, it))
    excl = text_exclusions([it for _, it in order], universe)
    built = {d: [] for d in domains}
    idx = 0
    for dname, it in order:
        gold = it["gold"]
        banned = {gold} | conflicts.get(gold, set()) | (excl[idx] - {gold})
        src = it["source"]
        pools = {
            "far": [o for o in universe if src not in sources.get(o, set()) and o not in banned],
            "near": [o for o in universe if src in sources.get(o, set()) and o not in banned],
            "mixed": [o for o in universe if o not in banned],
        }
        entry = {"text": it["text"], "gold": gold, "source": src, "tiers": {}}
        for ti, tier in enumerate(TIERS):
            pool = pools[tier]
            kmax = min(max(KS) - 1, len(pool))
            rng = random.Random(SEED_ITEMS * 7919 + idx * 10 + ti)
            entry["tiers"][tier] = rng.sample(pool, kmax)
        built[dname].append(entry)
        idx += 1
    return built


def opts_for(entry, tier, K, order_seed=None):
    d = entry["tiers"][tier]
    if len(d) < K - 1:
        return None
    opts = [entry["gold"]] + d[: K - 1]
    seed = order_seed if order_seed is not None else zlib.crc32((entry["gold"] + tier).encode()) % 10**6 + K
    random.Random(seed).shuffle(opts)
    return opts


# Prompt wording must not share content words with any candidate option:
# v1 used "...category of this text?" while CLINC ships an intent literally
# named "text" (send a text message). Laya was pulled to it on 94% of its
# near-tier CLINC errors at K=64; GLiClass and bge were unaffected. See
# DEVIATIONS.md D3. assert_no_collision() below enforces this going forward.
QUESTION = "Which label applies here?"
TEMPLATE = "This example is labeled {}."

PROMPT_STOP = {"which", "label", "applies", "here", "this", "is", "the", "a", "an",
               "of", "to", "in", "for", "and", "or", "example", "labeled"}


def assert_no_collision(universe):
    """Fail loudly if any option string equals a content word of the prompt."""
    words = set()
    for s in (QUESTION, TEMPLATE.replace("{}", " ")):
        words |= {w.strip(".,?!").lower() for w in s.split()}
    content = words - PROMPT_STOP
    hits = sorted(set(universe) & content)
    if hits:
        raise SystemExit(f"PROMPT-OPTION COLLISION: options {hits} appear in the prompt. "
                         "Reword QUESTION/TEMPLATE before running.")
    # also flag options that are substrings of the prompt as a warning
    soft = sorted(o for o in universe if len(o) > 3 and o in (QUESTION + " " + TEMPLATE).lower())
    if soft:
        log(f"  WARNING soft prompt-option overlap: {soft}")


def run_cell(adapter, model_name, dname, tier, items, f):
    stats = {}
    for K in KS:
        if model_name == "deberta-v3-large-zeroshot-v2.0" and K >= 128:
            use = items[:LARGE_SUBSAMPLE]
        else:
            use = items
        accs, top5s, nlls, fails, infeas = [], [], [], 0, 0
        for i, entry in enumerate(use):
            opts = opts_for(entry, tier, K)
            if opts is None:
                infeas += 1
                continue
            gi = opts.index(entry["gold"])
            try:
                probs, ms = adapter.decide(entry["text"], opts, TEMPLATE, question=QUESTION)
            except Exception:
                fails += 1
                continue
            pred = max(range(len(probs)), key=probs.__getitem__)
            top5 = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)[:5]
            accs.append(1.0 if pred == gi else 0.0)
            top5s.append(1.0 if gi in top5 else 0.0)
            nlls.append(-math.log(max(probs[gi], 1e-12)))
            f.write(json.dumps({"phase": "acc", "domain": dname, "tier": tier, "K": K, "i": i,
                                "gold_idx": gi, "probs": [round(p, 6) for p in probs],
                                "support": sum(1 for p in probs if p > 0),
                                "latency_ms": round(ms, 2)}) + "\n")
        n = len(accs)
        stats[K] = ({"n": n, "infeasible": infeas, "fails": fails,
                     "acc": round(sum(accs) / n, 4), "top5": round(sum(top5s) / n, 4),
                     "nll": round(sum(nlls) / n, 4)} if n else {"n": 0, "infeasible": infeas})
        log(f"  {model_name} {dname}/{tier} K={K}: " +
            (f"acc {stats[K]['acc']} (n={n}, infeas {infeas})" if n else f"ALL INFEASIBLE ({infeas})"))
    # order runs (near+far only)
    if tier in ("far", "near"):
        for K in ORDER_KS:
            flips, total = 0, 0
            for i, entry in enumerate(items):
                preds = []
                for s in range(N_ORDERS):
                    opts = opts_for(entry, tier, K, order_seed=90000 + i * 10 + s)
                    if opts is None:
                        break
                    try:
                        probs, _ = adapter.decide(entry["text"], opts, TEMPLATE, question=QUESTION)
                    except Exception:
                        continue
                    pred_lab = opts[max(range(len(probs)), key=probs.__getitem__)]
                    preds.append(pred_lab)
                    f.write(json.dumps({"phase": "order", "domain": dname, "tier": tier, "K": K,
                                        "i": i, "perm": s, "pred": pred_lab,
                                        "correct": pred_lab == entry["gold"]}) + "\n")
                if len(preds) >= 2:
                    total += 1
                    flips += 1 if len(set(preds)) > 1 else 0
            stats[f"flip_K{K}"] = {"n": total, "flip_rate": round(flips / total, 4) if total else None}
            log(f"  {model_name} {dname}/{tier} flip@K{K}: {stats[f'flip_K{K}']['flip_rate']} (n={total})")
    return stats


def main():
    import sys
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 200
    only = set(sys.argv[sys.argv.index("--models") + 1].split(",")) if "--models" in sys.argv else None
    OUT.mkdir(parents=True, exist_ok=True)

    universe, sources, conflicts, alias_of = load_universe()
    log(f"universe_v2: {len(universe)} options")
    assert_no_collision(universe)
    log("  prompt-option collision check: passed")
    domains = build_domain_items(n, alias_of)
    # drop items whose gold fell out of the universe (shouldn't happen; log if it does)
    for d in domains:
        bad = [it for it in domains[d] if it["gold"] not in sources]
        if bad:
            log(f"  WARNING {d}: {len(bad)} items with gold not in universe, dropped")
            domains[d] = [it for it in domains[d] if it["gold"] in sources][:n]
    log("building pools (dual-filter exclusions)...")
    built = build_pools(domains, universe, sources, conflicts)
    for d in built:
        near_sizes = sorted(len(e["tiers"]["near"]) for e in built[d])
        med = near_sizes[len(near_sizes) // 2] if near_sizes else "n/a"
        log(f"  {d}: n={len(built[d])}, near-pool median {med}")

    from .models import EmbeddingSim, GLiClassZS, LayaChoice, ZeroShotNLI
    factories = {
        "laya": lambda: (lambda a: (a.set_budgets(2048, 1024), a)[1])(LayaChoice()),
        "bge-large-en-v1.5": lambda: EmbeddingSim(),
        "gte-large": lambda: EmbeddingSim("thenlper/gte-large"),
        "gliclass-large-v3.0": lambda: GLiClassZS(),
        "deberta-v3-base-zeroshot-v2.0": lambda: ZeroShotNLI("MoritzLaurer/deberta-v3-base-zeroshot-v2.0"),
        "deberta-v3-large-zeroshot-v2.0": lambda: ZeroShotNLI("MoritzLaurer/deberta-v3-large-zeroshot-v2.0"),
    }
    summary_path = OUT / f"stage2_summary_n{n}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for mname, factory in factories.items():
        if only and mname not in only:
            continue
        cells = [(d, tier) for d in built for tier in TIERS
                 if f"{mname}|{d}|{tier}" not in summary]
        if not cells:
            log(f"=== {mname}: all cells done (resume)")
            continue
        log(f"=== loading {mname} ({len(cells)} cells)")
        try:
            adapter = factory()
        except Exception:
            log(f"LOAD FAILED {mname}\n{traceback.format_exc(limit=2)}")
            continue
        for d, tier in cells:
            fpath = OUT / f"items__{mname}__{d}__{tier}__n{n}.jsonl"
            try:
                with fpath.open("w") as f:
                    stats = run_cell(adapter, mname, d, tier, built[d], f)
                summary[f"{mname}|{d}|{tier}"] = stats
                summary_path.write_text(json.dumps(summary, indent=1))
            except Exception:
                log(f"CELL FAILED {mname}/{d}/{tier}\n{traceback.format_exc(limit=2)}")
        del adapter
        try:
            import torch
            torch.mps.empty_cache()
        except Exception:
            pass
    log("STAGE2 DONE")


if __name__ == "__main__":
    main()
