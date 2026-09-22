"""Phase A freeze: apply adjudication rulings to the nominated pairs and emit
universe_v1.json (canonical options + conflict matrix).

Rulings pre-adjudicated by Claude 2026-09-22, pending Mingyou spot-check.
Policy: same-source pairs are exempt (source datasets pre-adjudicate their own
label sets); cross-source nominations default to CONFLICT unless they are true
string variants (MERGE). Bias-to-conflict is deliberate: a false CONFLICT costs
a few distractor-pool entries; a missed one corrupts the metric.

Usage: .venv/bin/python -m harness.freeze_universe
"""
from __future__ import annotations

import json
import pathlib

# canonical <- alias (true synonyms / plural / word-order variants only)
MERGES = {
    "definition": ["definitions"],
    "change pin": ["pin change"],
    "insurance": ["insurances"],
}
# 'approval' (emotion) vs 'approvals' (legal clause) is deliberately NOT a
# merge: different meanings, near-identical strings -> CONFLICT below.


def main():
    out = pathlib.Path("results")
    draft = json.loads((out / "universe_draft.json").read_text())
    noms = [l for l in (out / "universe_cross_nominations.md").read_text().splitlines()
            if l.startswith("- [")]
    pairs = []
    for l in noms:
        body = l.split("]", 1)[1]
        body = body.replace("CONTAIN", "", 1)
        a = body.split("'")[1]
        b = body.split("  vs  ")[1].split("'")[1]
        pairs.append((a, b))

    alias_of = {}
    for canon, aliases in MERGES.items():
        for a in aliases:
            alias_of[a] = canon

    options = {}
    for k, v in draft["options"].items():
        canon = alias_of.get(k, k)
        if canon not in options:
            options[canon] = {"sources": set()}
        options[canon]["sources"].update(v["sources"])

    conflicts = set()
    for a, b in pairs:
        ca, cb = alias_of.get(a, a), alias_of.get(b, b)
        if ca != cb:
            conflicts.add(tuple(sorted((ca, cb))))

    v1 = {
        "version": "v1",
        "frozen": "2026-09-22",
        "adjudication": "Claude pre-adjudicated; pending Mingyou spot-check",
        "policy": {
            "same_source_exempt": True,
            "nomination_channels": ["mpnet cosine >= 0.60 (cross-source)",
                                     "token containment (cross-source)"],
            "filter_model": "sentence-transformers/all-mpnet-base-v2",
            "default_ruling": "CONFLICT",
        },
        "n_options": len(options),
        "n_merges": sum(len(v) for v in MERGES.values()),
        "n_conflicts": len(conflicts),
        "merges": MERGES,
        "options": {k: sorted(v["sources"]) for k, v in sorted(options.items())},
        "conflicts": sorted(list(p) for p in conflicts),
    }
    (out / "universe_v1.json").write_text(json.dumps(v1, indent=1))
    print(f"universe_v1: {v1['n_options']} canonical options, "
          f"{v1['n_merges']} aliases merged, {v1['n_conflicts']} conflict pairs")


if __name__ == "__main__":
    main()
