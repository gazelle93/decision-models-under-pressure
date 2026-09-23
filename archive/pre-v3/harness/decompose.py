"""Wave-2.5 decomposition analysis (review fix 3): 2x2 factorial over
universe (288 vs 410) x filters (off vs on), same 50 items in every arm.

  filter effect   = mean(B-A, C-D)   ambiguity removal at fixed universe
  dilution effect = mean(D-A, C-B)   universe growth at fixed filtering
  interaction     = C - B - D + A

Usage: .venv/bin/python -m harness.decompose
"""
from __future__ import annotations

import json
import pathlib

ARMS = {
    "A(288,nofilt)": "results/ksweep_pilot.json",
    "B(288,filters)": "results/ksweep_pilot_armB.json",
    "C(410,filters)": "results/ksweep_pilot_v3raw.json",
    "D(410,nofilt)": "results/ksweep_pilot_armD.json",
}
MODELS = ["laya", "gliclass-large-v3.0", "bge-large-en-v1.5", "deberta-v3-base-zeroshot-v2.0"]
KS = [32, 64, 128, 256]


def main():
    data = {k: json.loads(pathlib.Path(v).read_text()) for k, v in ARMS.items()}
    out = {}
    for m in MODELS:
        out[m] = {}
        print(f"\n=== {m}")
        print(f"{'K':>4} {'A':>6} {'B':>6} {'C':>6} {'D':>6} | {'filter':>7} {'dilute':>7} {'interact':>8}")
        for K in KS:
            try:
                a, b, c, d = (data[arm]["models"][m][str(K)]["acc"] for arm in ARMS)
            except (KeyError, TypeError):
                print(f"{K:>4}  missing arm data")
                continue
            filt = ((b - a) + (c - d)) / 2
            dil = ((d - a) + (c - b)) / 2
            inter = c - b - d + a
            out[m][K] = {"A": a, "B": b, "C": c, "D": d,
                         "filter_effect": round(filt, 3), "dilution_effect": round(dil, 3),
                         "interaction": round(inter, 3)}
            print(f"{K:>4} {a:>6.2f} {b:>6.2f} {c:>6.2f} {d:>6.2f} | {filt:>+7.3f} {dil:>+7.3f} {inter:>+8.3f}")
    pathlib.Path("results/decomposition.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/decomposition.json  (n=50 pilot: effects are directional)")


if __name__ == "__main__":
    main()
