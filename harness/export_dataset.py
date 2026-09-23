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
                    "text_chars": it["text_chars"],
                    **({"mtop_domain": it["mtop_domain"]} if "mtop_domain" in it else {}),
                    "max_gold_sim_near": round(it["max_gold_sim_near"], 4),
                    "max_gold_sim_far": round(it["max_gold_sim_far"], 4),
                    "distractors": {k: v for k, v in it["tiers"].items()},
                }
                rows.append(r)
                f.write(json.dumps(r) + "\n")

    # ---- items.csv (flat, for inspection)
    with (out / "items.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "domain", "text", "gold", "source", "leaked", "text_chars",
                    "n_near", "n_far", "max_gold_sim_near", "max_gold_sim_far"])
        for r in rows:
            w.writerow([r["uid"], r["domain"], r["text"], r["gold"], r["source"],
                        int(r["leaked"]), r["text_chars"], len(r["distractors"]["near"]),
                        len(r["distractors"]["far"]),
                        r["max_gold_sim_near"], r["max_gold_sim_far"]])

    # ---- per-RQ splits: uid indexes, not copies, so there is exactly one
    # canonical item record and no way for the splits to drift from it.
    SPLITS = {
        "rq1_kscaling": {
            "domains": ["clinc", "mtop", "goemotions", "dbpedia"],
            "tiers": ["ext"], "k_grid": [2, 4, 8, 16, 32, 64, 128, 256],
            "factors": ["K", "text_length"],
            "why": "Analysed on TWO factors: candidate-set size K and text length. "
                   "DBpedia is included UNTRUNCATED and supplies the length range "
                   "(p10 128 -> p90 1300 chars, a 10x spread INSIDE one domain) "
                   "against clinc/mtop ~34-38 median. Its text-free-picker residual "
                   "(0.245-0.260) inflates the accuracy LEVEL, but is uncorrelated "
                   "with text length within the domain (r=-0.089, p=0.21), so the "
                   "length factor is unconfounded by it. Long texts plus 255 options "
                   "will truncate some models' context: that is a measurable "
                   "interaction, not a bug, and truncation must be logged per call. "
                   "fin-topic excluded (Amendment 1).",
        },
        "rq2_order": {
            "domains": ["clinc", "goemotions", "fintopic", "mtop"],
            "tiers": ["near", "far"], "k_grid": [16, 64],
            "why": "fin-topic RETAINED: flip rate compares permutations of one "
                   "identical option set, so a format shortcut is constant within "
                   "the item and cannot manufacture or mask order sensitivity.",
        },
        "rq3_hardness": {
            "domains": ["clinc", "mtop"],
            "tiers": ["near", "far"], "k_grid": [2, 4, 8, 16, 32, 64],
            "why": "CLINC retained; MTOP added. MTOP has the strongest measured "
                   "tier separation in the suite (+0.342 vs clinc +0.239) because "
                   "its 102 intents cluster into 11 real domains, so near-band "
                   "siblings are same-object/different-verb (create/delete/get/"
                   "snooze alarm) -- the dense structure CLINC's flat multi-domain "
                   "taxonomy lacks. Its far tier is also the cleanest in the suite "
                   "(text-free picker 0.060, under the 0.125 gate). Caveat to "
                   "report: MTOP is G3 for BOTH model families (intent is trained "
                   "for deberta via MASSIVE/Banking77 and for Laya via support "
                   "triage), so it adds no rung diversity; GoEmotions can be "
                   "re-added if a G4*-for-Laya domain is wanted.",
        },
    }
    splits_dir = out / "splits"
    splits_dir.mkdir(exist_ok=True)
    split_meta = {}
    for name, spec in SPLITS.items():
        uids = [r["uid"] for r in rows if r["domain"] in spec["domains"]]
        payload = dict(spec)
        payload["n_items"] = len(uids)
        payload["uids"] = uids
        (splits_dir / f"{name}.json").write_text(json.dumps(payload, indent=1))
        split_meta[name] = {"n_items": len(uids), "domains": spec["domains"],
                            "tiers": spec["tiers"], "k_grid": spec["k_grid"]}

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
        "splits": split_meta,
        "failed_sources": uni.get("failed_sources", []),
        "files": {},
    }
    for name in ("items.jsonl", "items.csv", "universe.json", "gates.json",
                 "splits/rq1_kscaling.json", "splits/rq2_order.json",
                 "splits/rq3_hardness.json"):
        p = out / name
        manifest["files"][name] = {"sha256_16": sha(p), "bytes": p.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"exported dataset/{version}/")
    for name, meta in manifest["files"].items():
        print(f"  {name:16s} {meta['bytes']/1024:8.1f} KB  sha {meta['sha256_16']}")
    print(f"  items: {manifest['n_items']} ({dict(by_dom)})")
    for k, v in split_meta.items():
        print(f"  split {k:16s} n={v['n_items']:>3}  domains={v['domains']}  "
              f"tiers={v['tiers']}  K={v['k_grid'][0]}..{v['k_grid'][-1]}")


if __name__ == "__main__":
    main()
