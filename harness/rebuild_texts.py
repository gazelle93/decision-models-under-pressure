"""Restore item texts into the public dataset from their original sources.

Run once after cloning:  .venv/bin/python -m harness.rebuild_texts
Verifies every restored text against the published SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from .items import build_items
from .tiers import load_universe


def main():
    d = pathlib.Path("dataset/v3-public")
    rows = [json.loads(l) for l in (d / "items.jsonl").read_text().splitlines()]
    _, _, _, alias_of = load_universe("v3")
    print("re-drawing items from upstream sources with the published seed...")
    domains = build_items(200, alias_of, log=lambda m: None)
    by_uid = {it["uid"]: it["text"] for items in domains.values() for it in items}

    ok = bad = miss = 0
    out = []
    for r in rows:
        t = by_uid.get(r["uid"])
        if t is None:
            miss += 1; out.append(r); continue
        if hashlib.sha256(t.encode()).hexdigest() != r["text_sha256"]:
            bad += 1; out.append(r); continue
        ok += 1
        out.append({**{"uid": r["uid"], "domain": r["domain"], "text": t}, **r})
    with (d / "items.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"restored {ok} texts, {bad} hash mismatches, {miss} not found")
    if bad or miss:
        raise SystemExit("rebuild incomplete — upstream data may have changed; "
                         "the hashes pin exactly what this study used")
    print("all texts verified against the published hashes")


if __name__ == "__main__":
    main()
