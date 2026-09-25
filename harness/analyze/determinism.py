"""RQ2 determinism control: how much of the flip rate is the API, not the order.

Reports the shuffled-order rate beside the fixed-order rate per cell. The
fixed-order rate is a NOISE FLOOR, not a term to subtract: noise and order can
flip the same item, so the two do not decompose additively. What it bounds is
how much of the headline could be order at most.
"""
from __future__ import annotations

import json
import pathlib
import random
import statistics
from collections import defaultdict

from ..dataset import load_rq

LEDGER = pathlib.Path("results/jev/rq2_ledger.jsonl")


def ci(v, n=5000, seed=11):
    rng = random.Random(seed)
    s = sorted(statistics.mean(rng.choices(v, k=len(v))) for _ in range(n))
    return statistics.mean(v), s[int(.025 * n)], s[int(.975 * n)]


def load(arm, K):
    by = defaultdict(dict)
    for line in LEDGER.open():
        r = json.loads(line)
        p = r["key"].split("|")
        if p[1] != arm or int(p[4]) != K:
            continue
        by[(p[2], p[3])][int(p[5])] = r.get("choice")
    return {k: (1.0 if len(set(v.values())) > 1 else 0.0)
            for k, v in by.items() if len(v) == 5 and None not in v.values()}


def main(K=64):
    items, _ = load_rq("rq2")
    dom = {it["uid"]: it["domain"] for it in items}
    ordr, det = load("ord", K), load("det", K)
    both = sorted(set(ordr) & set(det))
    print(f"Jev RQ2 determinism control, K={K}, {len(both)} items with both arms complete\n")
    print(f"{'domain':12s}{'tier':6s}{'n':>5}{'shuffled':>10}{'fixed':>8}{'gap':>8}  "
          f"{'95% CI on gap':>20}")
    cells = defaultdict(lambda: ([], []))
    for k in both:
        d, t = dom[k[0]], k[1]
        cells[(d, t)][0].append(ordr[k])
        cells[(d, t)][1].append(det[k])
    for (d, t), (o, e) in sorted(cells.items()):
        diffs = [a - b for a, b in zip(o, e)]
        _, lo, hi = ci(diffs)
        print(f"{d:12s}{t:6s}{len(o):>5}{statistics.mean(o):>10.3f}{statistics.mean(e):>8.3f}"
              f"{statistics.mean(diffs):>+8.3f}  [{lo:>+7.3f},{hi:>+7.3f}]")
    o = [ordr[k] for k in both]
    e = [det[k] for k in both]
    diffs = [a - b for a, b in zip(o, e)]
    _, lo, hi = ci(diffs)
    print(f"\n{'POOLED':18s}{len(o):>5}{statistics.mean(o):>10.3f}{statistics.mean(e):>8.3f}"
          f"{statistics.mean(diffs):>+8.3f}  [{lo:>+7.3f},{hi:>+7.3f}]")
    agree = sum(1 for a, b in zip(o, e) if a == b) / len(o)
    print(f"\nitems flipping under BOTH arms: {sum(1 for a,b in zip(o,e) if a and b)}")
    print(f"items flipping only when shuffled: {sum(1 for a,b in zip(o,e) if a and not b)}")
    print(f"items flipping only when fixed:    {sum(1 for a,b in zip(o,e) if b and not a)}")
    print(f"same verdict under both arms: {agree:.3f}")


if __name__ == "__main__":
    main()
