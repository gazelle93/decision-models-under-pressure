"""Build the v3 dataset and run every pre-inference gate.

Usage: .venv/bin/python -m harness.build [--n 200]
"""
from __future__ import annotations

import json
import pathlib
import time

from . import gates, tiers
from .items import build_items

QUESTION = "Which label applies here?"
TEMPLATE = "This example is labeled {}."
KS = [2, 4, 8, 16, 32, 64]          # tier-contrast grid (RQ3); see tiers.MAX_K
KS_FAR_ONLY = [128, 256]            # RQ1 K-curve extension, far tier only


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    import sys
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 200
    universe, sources, conflicts, alias_of = tiers.load_universe("v3")
    log(f"universe_v3: {len(universe)} options, "
        f"{sum(len(v) for v in conflicts.values())//2} conflict pairs")
    log("building items (nested by construction)")
    domains = build_items(n, alias_of, log=log)
    log("building tier pools (banded + surface-matched)")
    allitems = [it for d in domains.values() for it in d]
    tiers.build_pools(allitems, universe, sources, conflicts, alias_of, log=log)
    # build_pools may drop items (thin gold stratum); keep domains in sync
    built = {it["uid"] for it in allitems if "tiers" in it}
    domains = {d: [it for it in items if it["uid"] in built] for d, items in domains.items()}
    log("  after construction: " + ", ".join(f"{d} n={len(v)}" for d, v in domains.items()))

    ctx = {"universe": universe, "sources": sources, "conflicts": conflicts,
           "domains": domains, "question": QUESTION, "template": TEMPLATE, "ks": KS}
    ok = gates.run_all(ctx, log=log)

    out = pathlib.Path("results/dataset_v3.json")
    out.write_text(json.dumps({
        "version": "v3", "n_per_domain": n, "ks": KS,
        "question": QUESTION, "template": TEMPLATE,
        "gates_passed": ok,
        "items": {d: [{k: v for k, v in it.items()} for it in items]
                  for d, items in domains.items()},
    }, indent=None))
    log(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)  gates_passed={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
