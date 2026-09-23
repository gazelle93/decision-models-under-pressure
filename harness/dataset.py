"""Load a frozen dataset version. Experiments read this, never rebuild."""
from __future__ import annotations

import json
import pathlib


def load(version="v3", root="dataset"):
    d = pathlib.Path(root) / version
    items = [json.loads(l) for l in (d / "items.jsonl").read_text().splitlines() if l.strip()]
    universe = json.loads((d / "universe.json").read_text())
    manifest = json.loads((d / "manifest.json").read_text())
    return items, universe, manifest


def by_domain(items, domains=None):
    out = {}
    for it in items:
        if domains and it["domain"] not in domains:
            continue
        out.setdefault(it["domain"], []).append(it)
    return out


def load_rq(rq, version="v3", root="dataset"):
    """Load exactly the items, tiers and K grid a research question may use.

    Experiments call this instead of filtering by hand — the fin-topic
    exclusion from RQ1/RQ3 is then enforced by the data, not by remembering.
    Returns (items, spec).
    """
    import json, pathlib
    key = {"rq1": "rq1_kscaling", "rq2": "rq2_order", "rq3": "rq3_hardness"}.get(rq.lower(), rq)
    spec = json.loads((pathlib.Path(root) / version / "splits" / f"{key}.json").read_text())
    items, _, _ = load(version, root)
    keep = set(spec["uids"])
    sel = [it for it in items if it["uid"] in keep]
    for it in sel:
        it["distractors"] = {k: v for k, v in it["distractors"].items() if k in spec["tiers"]}
    return sel, spec
