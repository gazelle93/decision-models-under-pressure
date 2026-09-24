"""Pre-registered analysis for RQ1: accuracy vs K AND vs text length."""
from __future__ import annotations

import json
import math
import pathlib
import random
from collections import defaultdict

FAM = {"deberta-v3-base-zeroshot-v2.0": "A1", "deberta-v3-large-zeroshot-v2.0": "A1",
       "bge-large-en-v1.5": "A2", "gte-large": "A2",
       "laya": "A3", "gliclass-large-v3.0": "A3"}
KS = [2, 4, 8, 16, 32, 64, 128, 256]
B = 1000


def load():
    rec = defaultdict(list)          # (model,K) -> [(correct, chars, domain, leaked, trunc)]
    trunc = defaultdict(int)
    for f in pathlib.Path("results/v3").glob("rq1__*.jsonl"):
        _, model, dom, tier = f.stem.split("__")
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("phase") != "acc" or "probs" not in r:
                continue
            ok = float(r["pred_idx"] == r["gold_idx"])
            rec[(model, r["K"])].append((ok, r["text_chars"], r["domain"],
                                         r["leaked"], r.get("truncated", False)))
            trunc[(model, r["K"])] += bool(r.get("truncated"))
    return rec, trunc


def ci(v, seed=3):
    if not v:
        return (float("nan"),) * 3
    rng = random.Random(seed); n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(v) / n, bs[int(.025 * B)], bs[int(.975 * B)]


def main():
    rec, trunc = load()
    models = sorted({k[0] for k in rec}, key=lambda m: (FAM[m], m))
    done = [m for m in models if all((m, K) in rec for K in KS)]

    print("=" * 84)
    print("RQ1 — accuracy vs candidate-set size K   (ext pool, 4 domains, n=800/cell)")
    print("=" * 84)
    print(f"\n{'model':30s}{'fam':5s}" + "".join(f"{('K='+str(k)):>7s}" for k in KS) + "   slope/2x")
    for m in models:
        row, xs, ys = [], [], []
        for K in KS:
            v = [x[0] for x in rec.get((m, K), [])]
            row.append(f"{sum(v)/len(v):>7.3f}" if v else "    ---")
            if v:
                xs.append(math.log2(K)); ys.append(sum(v) / len(v))
        if len(xs) > 2:
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            slope = sum((a-mx)*(b-my) for a, b in zip(xs, ys)) / sum((a-mx)**2 for a in xs)
        else:
            slope = float("nan")
        print(f"{m:30s}{FAM[m]:5s}" + "".join(row) + f"   {slope:+.4f}")

    print("\n" + "=" * 84)
    print("RQ1 second factor — TEXT LENGTH. Accuracy by length quartile, dbpedia only")
    print("(dbpedia is the only domain with a within-domain length spread: 10x)")
    print("=" * 84)
    db = [x for (m, K), v in rec.items() for x in v if x[2] == "dbpedia"]
    if db:
        chars = sorted({x[1] for x in db})
        qs = [chars[int(len(chars)*q)] for q in (.25, .5, .75)]
        print(f"\nquartile cuts (chars): <={qs[0]}, <={qs[1]}, <={qs[2]}, >")
        print(f"\n{'model':30s}{'K':>5s}" + "".join(f"{('Q'+str(i+1)):>9s}" for i in range(4)))
        for m in done:
            for K in (16, 64, 256):
                v = [x for x in rec.get((m, K), []) if x[2] == "dbpedia"]
                if not v:
                    continue
                buckets = [[], [], [], []]
                for ok, c, *_ in v:
                    b = 0 if c <= qs[0] else 1 if c <= qs[1] else 2 if c <= qs[2] else 3
                    buckets[b].append(ok)
                cells = "".join(f"{(sum(b)/len(b) if b else float('nan')):>9.3f}" for b in buckets)
                print(f"{m:30s}{K:>5d}" + cells)

    print("\n" + "=" * 84)
    print("TRUNCATION (Amendment 2: logged, never inferred)")
    print("=" * 84)
    any_t = False
    for m in models:
        row = [(K, trunc[(m, K)]) for K in KS if trunc.get((m, K))]
        if row:
            any_t = True
            print(f"  {m:30s} " + "  ".join(f"K={k}: {n}/800" for k, n in row))
    if not any_t:
        print("  no truncation on any call, any model, any K — the K-curves are")
        print("  cardinality effects, not context-overflow artifacts")

    print("\n" + "=" * 84)
    print("C3 — K-curve slope per FAMILY, bootstrap CI on pairwise differences")
    print("=" * 84)
    fam_items = defaultdict(lambda: defaultdict(list))
    for (m, K), v in rec.items():
        fam_items[FAM[m]][K] += [x[0] for x in v]

    def slope_of(per_k):
        xs = [math.log2(k) for k in KS if per_k.get(k)]
        ys = [sum(per_k[k]) / len(per_k[k]) for k in KS if per_k.get(k)]
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        return sum((a-mx)*(b-my) for a, b in zip(xs, ys)) / sum((a-mx)**2 for a in xs)

    def boot_slope(per_k, rng):
        res = {}
        for k, v in per_k.items():
            n = len(v)
            res[k] = [v[rng.randrange(n)] for _ in range(n)]
        return slope_of(res)

    fams = sorted(fam_items)
    base = {f: slope_of(fam_items[f]) for f in fams}
    print("\n  slope per doubling of K:  " + "   ".join(f"{f} {base[f]:+.4f}" for f in fams))
    print()
    for i, a in enumerate(fams):
        for b in fams[i+1:]:
            rng = random.Random(17)
            d = sorted(boot_slope(fam_items[a], rng) - boot_slope(fam_items[b], rng)
                       for _ in range(400))
            lo, hi = d[int(.025*400)], d[int(.975*400)]
            sep = "SEPARATED" if (lo > 0 or hi < 0) else "overlaps 0"
            print(f"  {a} vs {b}: {base[a]-base[b]:+.4f} [{lo:+.4f},{hi:+.4f}]  {sep}")

    print("\n" + "=" * 84)
    print("LEAKAGE STRATIFICATION (clinc carries 25% verbatim-gold items)")
    print("=" * 84)
    print(f"\n{'model':30s}{'K':>5s}{'leaked':>9s}{'clean':>9s}{'gap':>8s}")
    for m in done:
        for K in (16, 256):
            v = [x for x in rec.get((m, K), []) if x[2] == "clinc"]
            lk = [x[0] for x in v if x[3]]
            cl = [x[0] for x in v if not x[3]]
            if lk and cl:
                a, b = sum(lk)/len(lk), sum(cl)/len(cl)
                print(f"{m:30s}{K:>5d}{a:>9.3f}{b:>9.3f}{a-b:>+8.3f}")


if __name__ == "__main__":
    main()
