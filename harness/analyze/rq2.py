"""Pre-registered analysis for RQ2 / contrast C1 (order sensitivity)."""
from __future__ import annotations

import json
import pathlib
import random
from collections import defaultdict

FAM = {"deberta-v3-base-zeroshot-v2.0": "A1", "deberta-v3-large-zeroshot-v2.0": "A1",
       "bge-large-en-v1.5": "A2", "gte-large": "A2",
       "laya": "A3", "gliclass-large-v3.0": "A3"}
B = 2000


def load():
    """(model, K) -> list of per-item flip indicators; plus per-domain detail."""
    flips = defaultdict(list)
    bydom = defaultdict(list)
    fails = defaultdict(int)
    for f in pathlib.Path("results/v3").glob("rq2__*.jsonl"):
        _, model, dom, tier = f.stem.split("__")
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("phase") != "order":
                continue
            fails[model] += r.get("n_failed_perms", 0)
            preds = r.get("preds", [])
            if len(preds) != 5:          # incomplete sets excluded (H4)
                continue
            v = 1.0 if len(set(preds)) > 1 else 0.0
            flips[(model, r["K"])].append(v)
            bydom[(model, r["K"], dom, tier)].append(v)
    return flips, bydom, fails


def ci(vals, seed=5):
    rng = random.Random(seed)
    n = len(vals)
    bs = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(vals) / n, bs[int(.025 * B)], bs[int(.975 * B)]


def main():
    flips, bydom, fails = load()
    models = sorted({k[0] for k in flips}, key=lambda m: (FAM[m], m))
    print("=" * 78)
    print("RQ2 / C1 — does option ORDER change the answer?")
    print("5 permutations per item, identical option set; flip = argmax changed")
    print("=" * 78)
    print(f"\n{'model':32s} {'fam':4s} {'K=16 flip [95% CI]':>26s} {'K=64 flip [95% CI]':>26s}")
    verdict = {}
    for m in models:
        cells = []
        for K in (16, 64):
            v = flips.get((m, K), [])
            if not v:
                cells.append("          n/a          "); continue
            mu, lo, hi = ci(v)
            verdict.setdefault(m, {})[K] = (mu, lo, hi, len(v))
            cells.append(f"{mu:.4f} [{lo:.4f},{hi:.4f}] n={len(v)}")
        print(f"{m:32s} {FAM[m]:4s} " + " ".join(f"{c:>26s}" for c in cells))

    print("\n" + "-" * 78)
    print("HARNESS CHECK — A1 and A2 score each option independently, so order")
    print("cannot exist for them. Any non-zero rate is a tie broken by position.")
    print("-" * 78)
    for m in models:
        if FAM[m] in ("A1", "A2"):
            r = [f"K{K}={verdict[m][K][0]:.4f}" for K in (16, 64) if K in verdict.get(m, {})]
            ok = all(verdict[m][K][0] < 0.02 for K in verdict.get(m, {}))
            print(f"  {m:32s} {'  '.join(r)}   {'OK' if ok else 'INVESTIGATE'}")

    print("\n" + "-" * 78)
    print("C1 — 'residual order tax under 5%' requires the 95% CI UPPER bound < 0.05")
    print("-" * 78)
    for m in models:
        if FAM[m] != "A3":
            continue
        for K in (16, 64):
            if K not in verdict.get(m, {}):
                continue
            mu, lo, hi, n = verdict[m][K]
            print(f"  {m:24s} K={K:>2}: flip {mu:.4f}, CI upper {hi:.4f} -> "
                  f"{'CLAIM HOLDS (<5%)' if hi < 0.05 else 'CLAIM FAILS (>=5%)'}")

    print("\n" + "-" * 78)
    print("A3 detail by domain and tier (where do the flips concentrate?)")
    print("-" * 78)
    doms = sorted({k[2] for k in bydom})
    for m in [x for x in models if FAM[x] == "A3"]:
        print(f"  {m}")
        for K in (16, 64):
            row = []
            for d in doms:
                for tier in ("near", "far"):
                    v = bydom.get((m, K, d, tier), [])
                    if v:
                        row.append(f"{d[:4]}/{tier[0]} {sum(v)/len(v):.3f}")
            print(f"    K={K:>2}: " + "  ".join(row))
    tot = sum(fails.values())
    print(f"\nfailed permutations across all models: {tot} "
          f"({'none — no downward bias' if tot == 0 else 'EXCLUDED from rates'})")


if __name__ == "__main__":
    main()
