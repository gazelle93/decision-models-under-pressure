"""Experiment runner v3 — reads the FROZEN dataset, never rebuilds.

Fixes carried from the 2026-09-23 review:
  F8  full-precision probabilities in the per-item log (round(p,6) previously
      sent Laya's gold probability to 0.0 on 8/200 items at K=256, giving
      NLL 27.6 instead of the real value)
  H3  a cell where every call raised is no longer indistinguishable from an
      infeasible cell — fails are counted and surfaced
  H4  failed order permutations are counted, not silently skipped (skipping
      biased flip rates DOWN, the direction that made C1 easier to pass)
  A2  truncation is logged per call (Amendment 2)

Usage:
  .venv/bin/python -m harness.run_v3 --rq rq3 --models laya,bge-large-en-v1.5
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import time
import traceback
from collections import defaultdict

from .dataset import load_rq

OUT = pathlib.Path("results/v3")
N_ORDERS = 5
ORDER_KS = [16, 64]
QUESTION = "Which label applies here?"
TEMPLATE = "This example is labeled {}."


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def options_for(item, tier, K, seed=101):
    d = item["distractors"][tier]
    if len(d) < K - 1:
        return None
    opts = [item["gold"]] + d[: K - 1]
    random.Random(f"{seed}|{item['uid']}|{tier}|{K}").shuffle(opts)
    return opts


def build_models(names=None):
    from .models import EmbeddingSim, GLiClassZS, LayaChoice, ZeroShotNLI
    # cheapest first, so usable cells land early and an interrupted run still
    # leaves complete families rather than fragments
    reg = {
        "laya": lambda: (lambda a: (a.set_budgets(4096, 1024), a)[1])(LayaChoice()),
        "bge-large-en-v1.5": lambda: EmbeddingSim(),
        "gte-large": lambda: EmbeddingSim("thenlper/gte-large"),
        "gliclass-large-v3.0": lambda: GLiClassZS(),
        "deberta-v3-base-zeroshot-v2.0": lambda: ZeroShotNLI("MoritzLaurer/deberta-v3-base-zeroshot-v2.0"),
        "deberta-v3-large-zeroshot-v2.0": lambda: ZeroShotNLI("MoritzLaurer/deberta-v3-large-zeroshot-v2.0"),
    }
    return {k: v for k, v in reg.items() if not names or k in names}


def run_cell(adapter, mname, items, tier, ks, fh):
    stats = {}
    for K in ks:
        acc = top5 = nll = 0.0
        n = fails = infeas = trunc = 0
        lat = []
        for it in items:
            opts = options_for(it, tier, K)
            if opts is None:
                infeas += 1
                continue
            gi = opts.index(it["gold"])
            try:
                probs, ms = adapter.decide(it["text"], opts, TEMPLATE, question=QUESTION)
            except Exception:
                fails += 1
                fh.write(json.dumps({"phase": "acc", "uid": it["uid"], "tier": tier,
                                     "K": K, "error": traceback.format_exc(limit=1)[-200:]}) + "\n")
                continue
            pred = max(range(len(probs)), key=probs.__getitem__)
            order5 = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)[:5]
            acc += pred == gi
            top5 += gi in order5
            nll += -math.log(max(probs[gi], 1e-300))
            was_trunc = bool(getattr(adapter, "last_truncated", False))
            trunc += was_trunc
            lat.append(ms)
            n += 1
            fh.write(json.dumps({
                "phase": "acc", "uid": it["uid"], "domain": it["domain"], "tier": tier, "K": K,
                "gold_idx": gi, "pred_idx": pred, "probs": probs,      # FULL precision (F8)
                "text_chars": it["text_chars"], "leaked": it["leaked"],
                "truncated": was_trunc, "latency_ms": round(ms, 3),
            }) + "\n")
        lat.sort()
        stats[str(K)] = {"n": n, "fails": fails, "infeasible": infeas, "truncated": trunc,
                         "acc": round(acc / n, 4) if n else None,
                         "top5": round(top5 / n, 4) if n else None,
                         "nll": round(nll / n, 4) if n else None,
                         "p50_ms": round(lat[len(lat) // 2], 2) if lat else None}
        flag = ""
        if fails:
            flag += f" FAILS={fails}"
        if trunc:
            flag += f" TRUNC={trunc}"
        log(f"    K={K:>3}: acc {stats[str(K)]['acc']} (n={n}, infeas {infeas}){flag}")
    return stats


def run_order(adapter, items, tier, fh):
    out = {}
    for K in ORDER_KS:
        flips = total = perm_fails = 0
        for i, it in enumerate(items):
            preds, failed = [], 0
            for s in range(N_ORDERS):
                d = it["distractors"][tier]
                if len(d) < K - 1:
                    break
                opts = [it["gold"]] + d[: K - 1]
                random.Random(f"order|{it['uid']}|{tier}|{K}|{s}").shuffle(opts)
                try:
                    probs, _ = adapter.decide(it["text"], opts, TEMPLATE, question=QUESTION)
                except Exception:
                    failed += 1
                    continue
                preds.append(opts[max(range(len(probs)), key=probs.__getitem__)])
            perm_fails += failed
            # H4: only count items with the FULL permutation set; a partial set
            # biases the flip rate downward
            if len(preds) == N_ORDERS:
                total += 1
                flips += len(set(preds)) > 1
            fh.write(json.dumps({"phase": "order", "uid": it["uid"], "tier": tier, "K": K,
                                 "preds": preds, "n_failed_perms": failed}) + "\n")
        out[f"flip_K{K}"] = {"n_complete": total, "perm_fails": perm_fails,
                             "flip_rate": round(flips / total, 4) if total else None}
        log(f"    flip@K{K}: {out[f'flip_K{K}']['flip_rate']} "
            f"(n={total} complete, {perm_fails} perm fails)")
    return out


def main():
    import sys
    a = sys.argv
    rq = a[a.index("--rq") + 1] if "--rq" in a else "rq3"
    names = set(a[a.index("--models") + 1].split(",")) if "--models" in a else None
    limit = int(a[a.index("--limit") + 1]) if "--limit" in a else None

    items, spec = load_rq(rq)
    if limit:
        by = defaultdict(list)
        for it in items:
            by[it["domain"]].append(it)
        items = [x for v in by.values() for x in v[:limit]]
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"{rq}: {len(items)} items, domains {sorted({i['domain'] for i in items})}, "
        f"tiers {spec['tiers']}, K {spec['k_grid']}")

    sfile = OUT / f"{rq}_summary.json"
    summary = json.loads(sfile.read_text()) if sfile.exists() else {}
    for mname, factory in build_models(names).items():
        todo = [(d, t) for d in sorted({i["domain"] for i in items}) for t in spec["tiers"]
                if f"{mname}|{d}|{t}" not in summary]
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
            sel = [i for i in items if i["domain"] == dom]
            log(f"  {dom}/{tier} (n={len(sel)})")
            fpath = OUT / f"{rq}__{mname}__{dom}__{tier}.jsonl"
            try:
                with fpath.open("w") as fh:
                    cell = run_cell(adapter, mname, sel, tier, spec["k_grid"], fh)
                    if rq == "rq2":
                        cell.update(run_order(adapter, sel, tier, fh))
                summary[f"{mname}|{dom}|{tier}"] = cell
                sfile.write_text(json.dumps(summary, indent=1))
            except Exception:
                log(f"CELL FAILED {mname}/{dom}/{tier}\n{traceback.format_exc(limit=2)}")
        del adapter
        try:
            import torch
            torch.mps.empty_cache()
        except Exception:
            pass
    log("DONE")


if __name__ == "__main__":
    main()
