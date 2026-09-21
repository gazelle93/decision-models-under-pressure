"""Metrics for the decision-model eval harness.

Per the research brief (Section 6 protocol): accuracy, macro-F1, Brier,
NLL, and 15-bin equal-width ECE on the top-label confidence.
Debiased ECE is a tracked TODO (README).
"""
from __future__ import annotations

import math
from collections import defaultdict


def accuracy(preds, golds):
    return sum(p == g for p, g in zip(preds, golds)) / len(golds)


def macro_f1(preds, golds):
    labels = sorted(set(golds) | set(preds))
    f1s = []
    for c in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, golds) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, golds) if p != c and g == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def brier(probs, golds):
    """Multiclass Brier: mean squared distance between distribution and one-hot gold."""
    total = 0.0
    for dist, g in zip(probs, golds):
        total += sum((p - (1.0 if i == g else 0.0)) ** 2 for i, p in enumerate(dist))
    return total / len(golds)


def nll(probs, golds, eps=1e-12):
    return -sum(math.log(max(dist[g], eps)) for dist, g in zip(probs, golds)) / len(golds)


def ece(probs, golds, n_bins=15):
    """Equal-width ECE on top-label confidence (Guo et al. 2017 convention)."""
    bins = defaultdict(list)
    for dist, g in zip(probs, golds):
        conf = max(dist)
        pred = dist.index(conf)
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, 1.0 if pred == g else 0.0))
    n = len(golds)
    total = 0.0
    table = []
    for b in range(n_bins):
        rows = bins.get(b, [])
        if not rows:
            continue
        avg_conf = sum(c for c, _ in rows) / len(rows)
        avg_acc = sum(a for _, a in rows) / len(rows)
        total += (len(rows) / n) * abs(avg_conf - avg_acc)
        table.append({"bin": b, "n": len(rows), "conf": round(avg_conf, 4), "acc": round(avg_acc, 4)})
    return total, table


def summarize(records):
    """records: dicts with keys probs (list[float]), gold (int), latency_ms (float)."""
    probs = [r["probs"] for r in records]
    golds = [r["gold"] for r in records]
    preds = [dist.index(max(dist)) for dist in probs]
    lat = sorted(r["latency_ms"] for r in records)
    q = lambda x: lat[min(int(x * len(lat)), len(lat) - 1)]
    ece_val, reliability = ece(probs, golds)
    return {
        "n": len(records),
        "accuracy": round(accuracy(preds, golds), 4),
        "macro_f1": round(macro_f1(preds, golds), 4),
        "brier": round(brier(probs, golds), 4),
        "nll": round(nll(probs, golds), 4),
        "ece_15bin": round(ece_val, 4),
        "reliability": reliability,
        "latency_ms_p50": round(q(0.50), 1),
        "latency_ms_p95": round(q(0.95), 1),
    }
