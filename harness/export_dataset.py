"""Export the built dataset to a versioned, reusable, self-describing folder.

dataset/<version>/
  items.jsonl     canonical: one JSON object per item, including its tier pools
  items.csv       flat view for eyeballing (no nested pools)
  universe.json   the option universe, conflicts, merges, adjudication policy
  gates.json      gate results at freeze time
  manifest.json   build config, counts, sha256 of every file
  README.md       datacard: provenance, construction, limits, how to load

Usage: .venv/bin/python -m harness.export_dataset [--version v3]
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import shutil
import sys
from collections import Counter


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    version = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else "v3"
    src = json.loads(pathlib.Path("results/dataset_v3.json").read_text())
    uni = json.loads(pathlib.Path(f"results/universe_{version}.json").read_text())
    gates = json.loads(pathlib.Path("results/gates.json").read_text())

    out = pathlib.Path("dataset") / version
    out.mkdir(parents=True, exist_ok=True)

    # ---- items.jsonl (canonical)
    rows = []
    with (out / "items.jsonl").open("w") as f:
        for domain, items in src["items"].items():
            for it in items:
                r = {
                    "uid": it["uid"], "domain": domain, "text": it["text"],
                    "gold": it["gold"], "source": it["source"],
                    "leaked": it["leaked"],
                    "max_gold_sim_near": round(it["max_gold_sim_near"], 4),
                    "max_gold_sim_far": round(it["max_gold_sim_far"], 4),
                    "distractors": {"near": it["tiers"]["near"], "far": it["tiers"]["far"]},
                }
                rows.append(r)
                f.write(json.dumps(r) + "\n")

    # ---- items.csv (flat, for inspection)
    with (out / "items.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "domain", "text", "gold", "source", "leaked",
                    "n_near", "n_far", "max_gold_sim_near", "max_gold_sim_far"])
        for r in rows:
            w.writerow([r["uid"], r["domain"], r["text"], r["gold"], r["source"],
                        int(r["leaked"]), len(r["distractors"]["near"]),
                        len(r["distractors"]["far"]),
                        r["max_gold_sim_near"], r["max_gold_sim_far"]])

    shutil.copy(f"results/universe_{version}.json", out / "universe.json")
    (out / "gates.json").write_text(json.dumps(gates, indent=1))

    by_dom = Counter(r["domain"] for r in rows)
    manifest = {
        "version": version,
        "frozen": uni["frozen"],
        "n_items": len(rows),
        "n_items_by_domain": dict(by_dom),
        "n_options": uni["n_options"],
        "n_conflicts": uni["n_conflicts"],
        "seed_items": 101,
        "question": src["question"],
        "template": src["template"],
        "k_grid_tier_contrast": src["ks"],
        "max_k": max(src["ks"]),
        "gates_passed": src["gates_passed"],
        "gates": {k: v["pass"] for k, v in gates.items()},
        "rq_scope": {
            "RQ1_kcurve": ["clinc", "goemotions"],
            "RQ2_order": ["clinc", "goemotions", "fintopic"],
            "RQ3_tiers": ["clinc", "goemotions"],
            "fintopic_note": "excluded from RQ1/RQ3 (Amendment 1); retained for RQ2",
        },
        "failed_sources": uni.get("failed_sources", []),
        "files": {},
    }
    for name in ("items.jsonl", "items.csv", "universe.json", "gates.json"):
        p = out / name
        manifest["files"][name] = {"sha256_16": sha(p), "bytes": p.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"exported dataset/{version}/")
    for name, meta in manifest["files"].items():
        print(f"  {name:16s} {meta['bytes']/1024:8.1f} KB  sha {meta['sha256_16']}")
    print(f"  items: {manifest['n_items']} ({dict(by_dom)})")


if __name__ == "__main__":
    main()
