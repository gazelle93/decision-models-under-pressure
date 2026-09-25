"""Determinism control for RQ2.

The RQ2 flip rate sends five calls that differ in TWO ways: the option order
changed, and they are five separate calls. This arm removes the first
difference and keeps the second — same item, same option set, same ORDER,
five calls — so whatever still flips is the system disagreeing with itself.

Interpretation: this is the noise floor, not a term to subtract. If both
effects are live they can flip the same item, so they do not decompose.

Usage:
  .venv/bin/python -m harness.run.determinism --models laya
"""
from __future__ import annotations

import json
import pathlib
import random
import time
import traceback
from collections import defaultdict

from ..dataset import load_rq
from .local import N_ORDERS, ORDER_KS, QUESTION, TEMPLATE, build_models, log

OUT = pathlib.Path("results/v3")


def fixed_options(item, tier, K):
    """The SAME list every call: permutation 0 of the RQ2 order arm, so the
    control sits inside the distribution the headline was measured over."""
    d = item["distractors"][tier]
    if len(d) < K - 1:
        return None
    opts = [item["gold"]] + d[: K - 1]
    random.Random(f"order|{item['uid']}|{tier}|{K}|0").shuffle(opts)
    return opts


def run_det(adapter, items, tier, fh):
    out = {}
    for K in ORDER_KS:
        flips = total = fails = 0
        for it in items:
            opts = fixed_options(it, tier, K)
            if opts is None:
                continue
            preds, failed = [], 0
            for _ in range(N_ORDERS):
                try:
                    probs, _ = adapter.decide(it["text"], list(opts), TEMPLATE,
                                              question=QUESTION)
                except Exception:
                    failed += 1
                    continue
                preds.append(opts[max(range(len(probs)), key=probs.__getitem__)])
            fails += failed
            if len(preds) == N_ORDERS:
                total += 1
                flips += len(set(preds)) > 1
            fh.write(json.dumps({"phase": "det", "uid": it["uid"], "tier": tier,
                                 "K": K, "preds": preds, "n_failed": failed}) + "\n")
        out[f"det_K{K}"] = {"n_complete": total, "fails": fails,
                            "flip_rate": round(flips / total, 4) if total else None}
        log(f"    det@K{K}: {out[f'det_K{K}']['flip_rate']} (n={total}, {fails} fails)")
    return out


def main():
    import sys
    a = sys.argv
    names = set(a[a.index("--models") + 1].split(",")) if "--models" in a else {"laya"}
    items, spec = load_rq("rq2")
    OUT.mkdir(parents=True, exist_ok=True)
    sfile = OUT / "rq2_determinism.json"
    summary = json.loads(sfile.read_text()) if sfile.exists() else {}
    log(f"rq2 determinism: {len(items)} items, tiers {spec['tiers']}, K {ORDER_KS}")

    with open(OUT / "rq2_determinism.jsonl", "a") as fh:
        for mname, factory in build_models(names).items():
            todo = [(d, t) for d in sorted({i["domain"] for i in items})
                    for t in spec["tiers"] if f"{mname}|{d}|{t}" not in summary]
            if not todo:
                log(f"=== {mname}: complete (resume)")
                continue
            log(f"=== {mname} ({len(todo)} cells)")
            try:
                adapter = factory()
            except Exception:
                log(f"LOAD FAILED {mname}\n{traceback.format_exc(limit=2)}")
                continue
            for dom, tier in todo:
                sub = [i for i in items if i["domain"] == dom and i["tier_ok"].get(tier, True)] \
                    if items and "tier_ok" in items[0] else \
                    [i for i in items if i["domain"] == dom]
                log(f"  {dom}|{tier} ({len(sub)} items)")
                summary[f"{mname}|{dom}|{tier}"] = run_det(adapter, sub, tier, fh)
                sfile.write_text(json.dumps(summary, indent=1))
    log(f"wrote {sfile}")


if __name__ == "__main__":
    main()
