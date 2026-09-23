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
