"""Chance and majority-class reference baselines (review fix 2).

Every accuracy table must carry these; DailyDialog's majority class (~0.81)
beats every wave-1 model, so skewed sets are reported as lift over majority.
Usage: .venv/bin/python -m harness.baselines_ref
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

from .registry import REGISTRY

CAPS = {"enron_spam": 500, "daily_dialog_emotion": 500, "twitter_fin_topic": 500,
        "sst5": 500, "clinc150_oos": 300, "banking77": 300}
SEED = 42


def main():
    out = {}
    for key, cap in CAPS.items():
        spec = REGISTRY[key]
        ex = spec.loader(spec, cap, SEED)
        golds = Counter(e.gold for e in ex)
        n = len(ex)
        out[key] = {
            "n": n,
            "n_options": len(ex[0].options),
            "chance": round(1 / len(ex[0].options), 4),
            "majority_acc": round(golds.most_common(1)[0][1] / n, 4),
            "majority_class": ex[0].options[golds.most_common(1)[0][0]],
        }
        print(f"{key:22s} chance {out[key]['chance']:.3f}  majority {out[key]['majority_acc']:.3f} "
              f"({out[key]['majority_class']!r})")
    pathlib.Path("results/reference_baselines.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
