"""Fold the Jev arm into the three headline analyses."""
from __future__ import annotations

import json
import pathlib
import random
from collections import defaultdict

from ..dataset import load_rq

B = 2000


def ledger(rq):
    p = pathlib.Path(f"results/jev/{rq}_ledger.jsonl")
    out = {}
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("probs") is not None:
            out[r["key"]] = r
    return out


def ci(v, seed=3):
    rng = random.Random(seed); n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(v) / n, bs[int(.025 * B)], bs[int(.975 * B)]


def fit_slope(ks, accs):
    import math
    xs = [math.log2(k) for k in ks]
    mx, my = sum(xs) / len(xs), sum(accs) / len(accs)
    return sum((a - mx) * (b - my) for a, b in zip(xs, accs)) / sum((a - mx) ** 2 for a in xs)


def open_slopes(ks):
    """Open-model K-slopes fitted on the SAME K range as Jev.

    Jev's API refuses more than 255 options, so its curve stops at K=128 while
    the open models run to 256. Comparing a 2-128 slope against 2-256 slopes
    flatters whichever model stopped earlier, because the curves steepen at the
    top. Everything is refitted on the shared range here.
    """
    S = json.loads(pathlib.Path("results/published/rq1_summary.json").read_text())
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for key, cells in S.items():
        model = key.split("|")[0]
        for k in ks:
            c = cells.get(str(k))
            if c:
                agg[model][k][0] += c["acc"] * c["n"]
                agg[model][k][1] += c["n"]
    out = {}
    for model, byk in agg.items():
        got = [k for k in ks if byk[k][1]]
        if len(got) == len(ks):
            out[model] = fit_slope(got, [byk[k][0] / byk[k][1] for k in got])
    return out


def main():
    print("=" * 76)
    print("JEV ARM — 29,600 paid calls, $1.04, zero failures")
    print("=" * 76)

    # ---------- RQ1: K-curve
    items, spec = load_rq("rq1")
    gold = {it["uid"]: it["gold"] for it in items}
    L = ledger("rq1")
    print("\nRQ1 — accuracy vs K (ext pool; Jev's API caps options at 255, so no K=256)")
    per_k = defaultdict(list)
    for k, r in L.items():
        _, _, uid, tier, K = k.split("|")[:5]
        per_k[int(K)].append(float(r["choice"] == gold[uid]))
    ks = sorted(per_k)
    print("  " + "".join(f"{('K='+str(k)):>9s}" for k in ks))
    print("  " + "".join(f"{sum(per_k[k])/len(per_k[k]):>9.3f}" for k in ks))
    ys = [sum(per_k[k])/len(per_k[k]) for k in ks]
    slope = fit_slope(ks, ys)
    print(f"\n  slope per doubling of K, all models refitted on K={ks[0]}..{ks[-1]}:")
    ranked = sorted(open_slopes(ks).items(), key=lambda kv: -kv[1])
    print(f"    {'Jev':<32s} {slope:+.4f}")
    for m, s in ranked:
        print(f"    {m:<32s} {s:+.4f}")

    # ---------- RQ2: flip rate
    items2, _ = load_rq("rq2")
    L2 = ledger("rq2")
    byitem = defaultdict(dict)
    for k, r in L2.items():
        parts = k.split("|")
        if parts[1] != "ord":
            continue
        _, _, uid, tier, K, s = parts
        byitem[(uid, tier, int(K))][int(s)] = r["choice"]
    print("\nRQ2 — order sensitivity (5 permutations, identical option set)")
    flips = defaultdict(list)
    for (uid, tier, K), d in byitem.items():
        if len(d) == 5:
            flips[K].append(1.0 if len(set(d.values())) > 1 else 0.0)
    for K in sorted(flips):
        mu, lo, hi = ci(flips[K])
        print(f"  K={K:>2}: flip {mu:.4f} [{lo:.4f},{hi:.4f}]  n={len(flips[K])}")
    print("  open A3: laya .206/.494 | gliclass .157/.282   open A1/A2: 0.0000")
    print("  published independent figure for Jev on uncleaned sets: 0.13")

    # per-domain detail
    dom = {it["uid"]: it["domain"] for it in items2}
    dd = defaultdict(list)
    for (uid, tier, K), d in byitem.items():
        if len(d) == 5:
            dd[(dom[uid], tier, K)].append(1.0 if len(set(d.values())) > 1 else 0.0)
    print("  by domain/tier at K=64: " + "  ".join(
        f"{d[:4]}/{t[0]} {sum(v)/len(v):.3f}" for (d, t, K), v in sorted(dd.items())
        if K == 64))

    # ---------- RQ3: near vs far
    items3, _ = load_rq("rq3")
    gold3 = {it["uid"]: it["gold"] for it in items3}
    L3 = ledger("rq3")
    got = defaultdict(dict)
    for k, r in L3.items():
        _, _, uid, tier, K = k.split("|")[:5]
        got[(uid, int(K))][tier] = float(r["choice"] == gold3[uid])
    print("\nRQ3 — Delta = acc(far) - acc(near), paired per item")
    for K in (16, 32, 64):
        pairs = [v["far"] - v["near"] for (uid, kk), v in got.items()
                 if kk == K and "far" in v and "near" in v]
        if pairs:
            mu, lo, hi = ci(pairs)
            print(f"  K={K:>2}: {mu:+.3f} [{lo:+.3f},{hi:+.3f}]  n={len(pairs)}")
    print("  open families at K=64: A1 +0.256 | A2 +0.282 | A3 +0.351")

    # rounding-limited flag
    allr = list(L.values()) + list(L2.values()) + list(L3.values())
    z = sum(1 for r in allr if min(r["probs"]) == 0)
    print(f"\nProbability rounding: {z}/{len(allr)} calls ({z/len(allr):.0%}) have at least one")
    print("option at exactly 0.00 — NLL/Brier/ECE for Jev are rounding-limited.")


if __name__ == "__main__":
    main()
