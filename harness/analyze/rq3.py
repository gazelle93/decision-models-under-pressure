"""Pre-registered analysis for RQ3 / contrast C2.

Order matters and is fixed by PLAN.md: within-family variance is
reported BEFORE any between-family claim, because a family of two that
disagrees is not a family.
"""
from __future__ import annotations

import json
import pathlib
import random
from collections import defaultdict

FAM = {"deberta-v3-base-zeroshot-v2.0": "A1", "deberta-v3-large-zeroshot-v2.0": "A1",
       "bge-large-en-v1.5": "A2", "gte-large": "A2",
       "laya": "A3", "gliclass-large-v3.0": "A3"}
KS = [16, 32, 64]
B = 2000


def load():
    """per (model, domain, tier, K): uid -> correct?"""
    out = defaultdict(dict)
    for f in pathlib.Path("results/v3").glob("rq3__*.jsonl"):
        _, model, dom, tier = f.stem.split("__")
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("phase") != "acc" or "probs" not in r:
                continue
            out[(model, dom, r["tier"], r["K"])][r["uid"]] = \
                float(r["pred_idx"] == r["gold_idx"])
    return out


def delta(cells, model, K, doms=("clinc", "mtop")):
    """paired far-near per item, pooled over domains"""
    pairs = []
    for d in doms:
        far = cells.get((model, d, "far", K), {})
        near = cells.get((model, d, "near", K), {})
        for uid in set(far) & set(near):
            pairs.append(far[uid] - near[uid])
    return pairs


def boot(vals, rng):
    n = len(vals)
    return sum(vals[rng.randrange(n)] for _ in range(n)) / n


def ci(vals, seed=7):
    rng = random.Random(seed)
    bs = sorted(boot(vals, rng) for _ in range(B))
    return sum(vals) / len(vals), bs[int(.025 * B)], bs[int(.975 * B)]


def main():
    cells = load()
    models = sorted({k[0] for k in cells})
    print("=" * 74)
    print("RQ3 / C2 — does option-conditioning help against near distractors?")
    print("Δ = acc(far) − acc(near), paired per item, pooled clinc+mtop, n=400")
    print("=" * 74)

    print(f"\n{'model':32s} {'fam':4s} " + " ".join(f"{'K='+str(k):>18s}" for k in KS))
    per_model = {}
    for m in sorted(models, key=lambda x: (FAM[x], x)):
        row, cells_ok = [], True
        for K in KS:
            v = delta(cells, m, K)
            if not v:
                row.append("     n/a     "); cells_ok = False; continue
            mu, lo, hi = ci(v)
            per_model.setdefault(m, {})[K] = (mu, lo, hi, v)
            row.append(f"{mu:+.3f} [{lo:+.2f},{hi:+.2f}]")
        print(f"{m:32s} {FAM[m]:4s} " + " ".join(f"{r:>18s}" for r in row))

    print("\n" + "-" * 74)
    print("STEP 1 (pre-registered gate): within-family spread vs between-family spread")
    print("-" * 74)
    applicable = True
    for K in KS:
        fam_vals = defaultdict(list)
        for m, d in per_model.items():
            if K in d:
                fam_vals[FAM[m]].append(d[K][0])
        within = max(max(v) - min(v) for v in fam_vals.values() if len(v) > 1)
        means = {f: sum(v) / len(v) for f, v in fam_vals.items()}
        between = max(means.values()) - min(means.values())
        ok = within < between
        applicable &= ok
        print(f"  K={K:>2}: within-family max spread {within:.3f} | between-family spread "
              f"{between:.3f} | {'families cohere' if ok else 'WITHIN > BETWEEN'}")
        print(f"        family means: " + "  ".join(f"{f} {v:+.3f}" for f, v in sorted(means.items())))

    print("\n" + "-" * 74)
    print("STEP 2: C2 decision rule")
    print("-" * 74)
    if not applicable:
        print("  C2 INAPPLICABLE as pre-registered: within-family variance exceeds")
        print("  between-family variance, so architecture class is not the operative")
        print("  variable. Reporting per-model deltas instead of family claims.")
    else:
        wins = 0
        for K in KS:
            fam = defaultdict(list)
            for m, d in per_model.items():
                if K in d:
                    fam[FAM[m]].append(d[K][3])
            pooled = {f: [x for v in vs for x in v] for f, vs in fam.items()}
            a3 = sum(pooled["A3"]) / len(pooled["A3"])
            beats = []
            for other in ("A1", "A2"):
                o = sum(pooled[other]) / len(pooled[other])
                rng = random.Random(11)
                diffs = sorted(boot(pooled["A3"], rng) - boot(pooled[other], rng) for _ in range(B))
                lo, hi = diffs[int(.025 * B)], diffs[int(.975 * B)]
                beats.append((other, a3 - o, lo, hi, hi < 0))
            allbeat = all(b[4] for b in beats)
            wins += allbeat
            print(f"  K={K:>2}: A3 Δ={a3:+.3f}  " +
                  "  ".join(f"vs {o}: {d:+.3f} [{lo:+.2f},{hi:+.2f}]{' <' if s else ''}"
                            for o, d, lo, hi, s in beats))
        print(f"\n  A3 beat BOTH families with CI separation at {wins}/3 K values "
              f"(rule: >=2 => SUPPORTED)")
        print(f"  VERDICT: {'SUPPORTED' if wins >= 2 else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
