"""Post-hoc temperature scaling over logged probability distributions.

log(p) is a valid logit reparameterization (softmax-invariant), so a single
temperature can be fitted from the JSONL alone, no model re-runs. Per run:
50/50 calibration/test split (seed 7 shuffle), T fitted by grid+refine on
calibration NLL, pre/post metrics reported on the test half only.
"""
from __future__ import annotations

import json
import math
import pathlib
import random

from . import metrics


def apply_T(probs, T):
    z = [math.log(max(p, 1e-12)) / T for p in probs]
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    s = sum(e)
    return [v / s for v in e]


def nll_at(records, T):
    tot = 0.0
    for r in records:
        tot += -math.log(max(apply_T(r["probs"], T)[r["gold"]], 1e-12))
    return tot / len(records)


GRID_LO, GRID_HI = math.exp(-24 / 8.0), math.exp(48 / 8.0)  # ~0.05 .. ~403


def fit_T(cal):
    """Returns (T, railed). Post-review fix: the old grid topped out at
    exp(4)*1.3 = 71, and two rows silently railed there (calibrated uniform
    guessing). Wider grid + explicit rail flag."""
    grid = [math.exp(x / 8.0) for x in range(-24, 49)]
    best = min(grid, key=lambda T: nll_at(cal, T))
    lo, hi = best / 1.3, best * 1.3
    for _ in range(20):
        mid1, mid2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll_at(cal, mid1) < nll_at(cal, mid2):
            hi = mid2
        else:
            lo = mid1
    T = (lo + hi) / 2
    railed = T >= GRID_HI * 0.95 or T <= GRID_LO * 1.05
    return T, railed


def main():
    out = pathlib.Path("results")
    summary = []
    for f in sorted(out.glob("*.jsonl")):
        name = f.stem
        if "__" not in name:
            continue
        dataset, model = name.split("__")[0], name.split("__")[1]
        records = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if len(records) < 60:
            continue
        idx = list(range(len(records)))
        random.Random(7).shuffle(idx)
        half = len(idx) // 2
        cal = [records[i] for i in idx[:half]]
        test = [records[i] for i in idx[half:]]
        T, railed = fit_T(cal)
        pre = metrics.summarize(test)
        post_records = [{**r, "probs": apply_T(r["probs"], T)} for r in test]
        post = metrics.summarize(post_records)
        confs = [max(r["probs"]) for r in post_records]
        conf_std = (sum((c - sum(confs) / len(confs)) ** 2 for c in confs) / len(confs)) ** 0.5
        row = {"dataset": dataset, "model": model, "n_test": len(test), "T": round(T, 3),
               "T_railed": railed, "post_conf_std": round(conf_std, 4),
               "ece_pre": pre["ece_15bin"], "ece_post": post["ece_15bin"],
               "nll_pre": pre["nll"], "nll_post": post["nll"],
               "brier_pre": pre["brier"], "brier_post": post["brier"],
               "accuracy": post["accuracy"]}
        summary.append(row)
        rail = " RAILED" if railed else ""
        flat = " CONST-CONF" if conf_std < 0.01 else ""
        print(f"{model:32s} {dataset:22s} T={T:7.2f}{rail}{flat}  ece {pre['ece_15bin']:.3f} -> {post['ece_15bin']:.3f}"
              f"  nll {pre['nll']:.3f} -> {post['nll']:.3f}")
    (out / "calibration_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
