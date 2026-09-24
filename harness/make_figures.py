"""Generate the README figures from the recorded results."""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#6b7280", "#9ca3af", "#d8dad8"
JEV, LAYA, FIELD = "#2a78d6", "#eb6834", "#b6bbc0"
KS = [2, 4, 8, 16, 32, 64, 128]
CURVES = {
    "Jev":            [.890, .801, .782, .769, .721, .672, .600],
    "Laya":           [.866, .777, .733, .691, .641, .521, .385],
    "deberta-large":  [.856, .777, .733, .691, .619, .521, .410],
    "deberta-base":   [.829, .729, .677, .639, .561, .469, .385],
    "gliclass":       [.802, .704, .676, .625, .569, .476, .394],
    "bge-large":      [.681, .618, .593, .561, .507, .435, .357],
    "gte-large":      [.670, .586, .554, .527, .475, .425, .354],
}


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
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  {OUT/name}")


def kcurve():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style(ax)
    for name, ys in CURVES.items():
        xs = [k for k, y in zip(KS, ys) if y is not None]
        vs = [y for y in ys if y is not None]
        if name == "Jev":
            ax.plot(xs, vs, color=JEV, lw=2.6, marker="o", ms=5, zorder=5, label=name)
        elif name == "Laya":
            ax.plot(xs, vs, color=LAYA, lw=2.6, marker="o", ms=5, zorder=4, label=name)
        else:
            ax.plot(xs, vs, color=FIELD, lw=1.3, alpha=.85, zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS); ax.set_xticklabels(KS)
    ax.set_xlabel("number of candidate options", color=MUTED, fontsize=9.5)
    ax.set_ylabel("accuracy", color=MUTED, fontsize=9.5)
    ax.set_ylim(.31, .95)
    ax.annotate("Jev", (128, .600), xytext=(9, 0), textcoords="offset points",
                color=JEV, fontsize=10.5, weight="bold")
    ax.annotate("Laya", (128, .385), xytext=(9, -4), textcoords="offset points",
                color=LAYA, fontsize=10.5, weight="bold")
    ax.annotate("four open models", (64, .450), xytext=(-46, -30),
                textcoords="offset points", color=MUTED, fontsize=9)
    ax.set_title("Accuracy falls as the candidate list grows", color=INK,
                 fontsize=11.5, weight="bold", loc="left", pad=12)
    save(fig, "k-curve.png")


def bars(data, title, xlabel, name, fmt="{:.1%}", xmax=None):
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    style(ax)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=.7, alpha=.6)
    names = [d[0] for d in data][::-1]
    vals = [d[1] for d in data][::-1]
    cols = [d[2] for d in data][::-1]
    ax.barh(names, vals, color=cols, height=.6)
    for i, v in enumerate(vals):
        ax.text(v + (xmax or max(vals)) * .015, i, fmt.format(v), va="center",
                fontsize=9.5, color=INK, weight="bold")
    ax.set_xlim(0, xmax or max(vals) * 1.22)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=11.5, weight="bold", loc="left", pad=12)
    save(fig, name)


if __name__ == "__main__":
    kcurve()
    bars([("Laya", .494, LAYA), ("gliclass", .282, FIELD), ("Jev", .146, JEV),
          ("deberta / bge / gte", .0001, "#1baf7a")],
         "Answers that change when only the option order changes",
         "share of decisions, 64 candidates", "order-flips.png", xmax=.56)
    bars([("Laya", .351, LAYA), ("embedding scorers", .282, FIELD),
          ("pair cross-encoders", .256, FIELD), ("Jev", .105, JEV)],
         "Accuracy lost when the wrong options are nearly right",
         "drop in accuracy, 64 candidates", "near-distractors.png",
         fmt="{:.3f}", xmax=.40)
