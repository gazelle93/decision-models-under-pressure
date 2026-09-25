"""Generate the README figures from results/published/.

Nothing here is hardcoded: every number is read from the published aggregates,
so the charts cannot drift from the tables. Re-run after any new result.

Usage: .venv/bin/python -m harness.analyze.figures
"""
from __future__ import annotations

import collections
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB = pathlib.Path("results/published")
OUT = pathlib.Path("docs/figures")

INK, MUTED, GRID = "#6b7280", "#9ca3af", "#d8dad8"
JEV, LAYA, FIELD, ZERO = "#2a78d6", "#eb6834", "#b6bbc0", "#1baf7a"
JEV_T = "#93bdec"      # tint of JEV: same entity, second measurement
KS = [2, 4, 8, 16, 32, 64, 128]
SHORT = {
    "deberta-v3-large-zeroshot-v2.0": "deberta-large",
    "deberta-v3-base-zeroshot-v2.0": "deberta-base",
    "gliclass-large-v3.0": "gliclass",
    "bge-large-en-v1.5": "bge-large",
    "gte-large": "gte-large",
    "laya": "Laya",
}


def load(name):
    return json.loads((PUB / name).read_text())


def pooled_acc():
    """Per-model accuracy at each K, pooled over the RQ1 domains by item count."""
    S = load("rq1_summary.json")
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    for key, cells in S.items():
        model = key.split("|")[0]
        for k in KS:
            c = cells.get(str(k))
            if c:
                agg[model][k][0] += c["acc"] * c["n"]
                agg[model][k][1] += c["n"]
    return {SHORT.get(m, m): {k: t / n for k, (t, n) in ks.items()} for m, ks in agg.items()}


def pooled_flip(K):
    """Per-model flip rate at one K, pooled over RQ2 domains and tiers."""
    S = load("rq2_summary.json")
    agg = collections.defaultdict(lambda: [0.0, 0])
    for key, cells in S.items():
        model = key.split("|")[0]
        f = cells.get(f"flip_K{K}")
        if f and f["n_complete"]:
            agg[model][0] += f["flip_rate"] * f["n_complete"]
            agg[model][1] += f["n_complete"]
    return {SHORT.get(m, m): t / n for m, (t, n) in agg.items()}


def pooled_delta(K):
    """Per-model acc(far) - acc(near) at one K, pooled over RQ3 domains."""
    S = load("rq3_summary.json")
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    for key, cells in S.items():
        model, _, tier = key.split("|")
        c = cells.get(str(K))
        if c:
            agg[model][tier][0] += c["acc"] * c["n"]
            agg[model][tier][1] += c["n"]
    out = {}
    for m, t in agg.items():
        far = t["far"][0] / t["far"][1]
        near = t["near"][0] / t["near"][1]
        out[SHORT.get(m, m)] = far - near
    return out


def style(ax):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=.7, alpha=.6)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  {OUT/name}")


def kcurve(curves):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style(ax)
    for name, ys in curves.items():
        xs = [k for k in KS if k in ys]
        vs = [ys[k] for k in xs]
        if name == "Jev":
            ax.plot(xs, vs, color=JEV, lw=2.6, marker="o", ms=5, zorder=5)
        elif name == "Laya":
            ax.plot(xs, vs, color=LAYA, lw=2.6, marker="o", ms=5, zorder=4)
        else:
            ax.plot(xs, vs, color=FIELD, lw=1.3, alpha=.85, zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS); ax.set_xticklabels(KS)
    ax.set_xlabel("number of candidate options", color=MUTED, fontsize=9.5)
    ax.set_ylabel("accuracy", color=MUTED, fontsize=9.5)
    ax.set_ylim(.31, .95)
    ax.annotate("Jev", (128, curves["Jev"][128]), xytext=(9, 0),
                textcoords="offset points", color=JEV, fontsize=10.5, weight="bold")
    ax.annotate("Laya", (128, curves["Laya"][128]), xytext=(9, -4),
                textcoords="offset points", color=LAYA, fontsize=10.5, weight="bold")
    ax.annotate("four open models", (64, .450), xytext=(-46, -30),
                textcoords="offset points", color=MUTED, fontsize=9)
    ax.set_title("Accuracy falls as the candidate list grows", color=INK,
                 fontsize=11.5, weight="bold", loc="left", pad=12)
    save(fig, "k-curve.png")


def bars(data, title, xlabel, name, fmt="{:.1%}", xmax=None, zero_label=None):
    """data: [(label, value, colour)], drawn top to bottom in the order given."""
    fig, ax = plt.subplots(figsize=(7.4, .46 * len(data) + 1.35))
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=.7, alpha=.6)
    names = [d[0] for d in data][::-1]
    vals = [d[1] for d in data][::-1]
    cols = [d[2] for d in data][::-1]
    lim = xmax or max(vals) * 1.22
    ax.barh(names, vals, color=cols, height=.6)
    for i, v in enumerate(vals):
        # an exact zero has no bar, so the label has to carry the information
        txt = (zero_label or fmt.format(v)) if v == 0 else fmt.format(v)
        ax.text(max(v, 0) + lim * .015, i, txt, va="center", fontsize=9.5,
                color=INK, weight="bold" if v > 0 else "normal")
    ax.set_xlim(0, lim)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=11.5, weight="bold", loc="left", pad=12)
    save(fig, name)


def main():
    jev = load("jev_summary.json")

    curves = pooled_acc()
    curves["Jev"] = {int(k): v for k, v in jev["rq1_acc_by_k"].items()}
    kcurve(curves)

    # Order flips at K=64. Every model gets its own row: rounding three models
    # into one "never flips" bar hid gte-large's 2%, which is not zero.
    flips = pooled_flip(64)
    flips["Jev"] = jev["rq2_flip_by_k"]["64"]
    rows = sorted(flips.items(), key=lambda kv: -kv[1])
    bars([(n, v, JEV if n == "Jev" else LAYA if n == "Laya" else (ZERO if v == 0 else FIELD))
          for n, v in rows],
         "Answers that change when only the option order changes",
         "share of decisions, 64 candidates", "order-flips.png",
         fmt="{:.1%}", xmax=.60, zero_label="0.0% (exact)")

    # What the flip rate is actually made of. Shuffling the order varies two
    # things at once for a hosted API: the order, and the fact that it is a
    # second call. The fixed-order bars separate them.
    det = load("rq2_determinism.json")
    bars([("Laya, shuffled order", det["laya_control"]["shuffled"], LAYA),
          ("Laya, order held fixed", det["laya_control"]["fixed"], ZERO),
          ("Jev, shuffled order", det["pooled"]["shuffled"], JEV),
          ("Jev, order held fixed", det["pooled"]["fixed"], JEV_T)],
         "How much of the flip rate is the order, and how much is the model",
         "share of decisions, 64 candidates", "flip-decomposition.png",
         fmt="{:.1%}", xmax=.60, zero_label="0.0% (exact)")

    # Near vs far distractors at K=64, every model shown individually.
    deltas = pooled_delta(64)
    deltas["Jev"] = jev["rq3_by_k"]["64"]["delta"]
    rows = sorted(deltas.items(), key=lambda kv: -kv[1])
    bars([(n, v, JEV if n == "Jev" else LAYA if n == "Laya" else FIELD) for n, v in rows],
         "Accuracy lost when the wrong options are nearly right",
         "drop in accuracy, 64 candidates", "near-distractors.png",
         fmt="{:.3f}", xmax=.42)


if __name__ == "__main__":
    main()
