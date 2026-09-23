"""Pre-inference gates. Nothing runs until every gate passes.

Each gate exists because a specific defect reached a reported finding
(REVIEW-2026-09-23.md). Gates are assertions, not diagnostics.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter, defaultdict

import numpy as np

# fixed function-word list — NOT derived from the prompt, so the check can fail
FUNCTION_WORDS = {"a","an","the","this","that","these","those","is","are","was","were","be",
                  "of","to","in","for","and","or","not","no","on","at","with","which","what",
                  "here","there","it","its","as","by","from","about","into","up","down"}
GATES = {}


def gate(name):
    def deco(fn):
        GATES[name] = fn
        return fn
    return deco


@gate("G1_prompt_collision")
def g1(ctx):
    """No option may equal or be contained in the prompt's content words."""
    words = {w.strip(".,?!{}").lower() for s in (ctx["question"], ctx["template"]) for w in s.split()}
    content = {w for w in words if w and w not in FUNCTION_WORDS}
    hits = sorted(set(ctx["universe"]) & content)
    soft = sorted(o for o in ctx["universe"]
                  if len(o) > 3 and o in (ctx["question"] + " " + ctx["template"]).lower())
    ok = not hits and not soft
    return ok, f"content words {sorted(content)}; exact hits {hits}; substring hits {soft}"


@gate("G2_format_tell")
def g2(ctx):
    """A surface-feature classifier must not separate FAR from NEAR (AUC <= 0.60)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def feats(o):
        w = o.split()
        return [len(o), len(w), float(np.mean([len(x) for x in w])), max(len(x) for x in w),
                1.0*(" " not in o and len(o) > 12), 1.0*o.endswith("s"),
                1.0*any(c.isdigit() for c in o), 1.0*("|" in o or "&" in o), 1.0*(" or " in o)]
    msgs, worst = [], 0.0
    for d, items in ctx["domains"].items():
        X, y = [], []
        for it in items:
            for o in it["tiers"]["near"][:40]:
                X.append(feats(o)); y.append(1)
            for o in it["tiers"]["far"][:40]:
                X.append(feats(o)); y.append(0)
        auc = cross_val_score(LogisticRegression(max_iter=3000), np.array(X), np.array(y),
                              cv=5, scoring="roc_auc").mean()
        worst = max(worst, auc)
        msgs.append(f"{d} {auc:.3f}")
    return worst <= 0.60, f"far-vs-near surface AUC: {', '.join(msgs)} (max {worst:.3f}, gate <=0.60)"


@gate("G3_textfree_gold_picker")
def g3(ctx):
    """A classifier with NO access to the text must not find the gold above 2x chance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    def feats(o):
        w = o.split()
        return [len(o), len(w), float(np.mean([len(x) for x in w])), max(len(x) for x in w),
                1.0*(" " not in o and len(o) > 12), 1.0*o.endswith("s")]
    msgs, fail = [], False
    for d, items in ctx["domains"].items():
        for tier in ("near", "far"):
            X, y, g = [], [], []
            for it in items:
                opts = ([it["gold"]] + it["tiers"][tier][:15])
                for o in opts:
                    X.append(feats(o)); y.append(1 if o == it["gold"] else 0); g.append(it["gold"])
            X, y, g = np.array(X), np.array(y), np.array(g)
            if len(set(g)) < 3:
                continue
            gkf = GroupKFold(n_splits=min(5, len(set(g))))
            acc = []
            for tr, te in gkf.split(X, y, g):
                clf = LogisticRegression(max_iter=3000).fit(X[tr], y[tr])
                p = clf.predict_proba(X[te])[:, 1]
                # per 16-option block, does the highest-scoring option = gold?
                blocks = len(te) // 16
                for b in range(blocks):
                    sl = slice(b*16, (b+1)*16)
                    acc.append(1.0 if y[te][sl][np.argmax(p[sl])] == 1 else 0.0)
            a = float(np.mean(acc)) if acc else 0.0
            chance = 1/16
            msgs.append(f"{d}/{tier} {a:.3f}")
            if a > 2*chance:
                fail = True
    return not fail, f"text-free picker acc (chance {1/16:.3f}, gate <={2/16:.3f}): {', '.join(msgs)}"


@gate("G4_gold_position_uniform")
def g4(ctx):
    """Gold position must be item-dependent, not a function of the gold label."""
    from .tiers import options_for
    msgs, fail = [], False
    for d, items in ctx["domains"].items():
        for K in (16, 64):
            by_gold = defaultdict(set)
            pos = []
            for it in items:
                o = options_for(it, "far", K)
                if o is None:
                    continue
                p = o.index(it["gold"])
                pos.append(p)
                by_gold[it["gold"]].add(p)
            if not pos:
                continue
            multi = sum(1 for g, s in by_gold.items() if len(s) > 1)
            repeated = max(Counter(pos).values())
            share = repeated / len(pos)
            used = len(set(pos))
            msgs.append(f"{d}K{K}: {used}/{K} slots, max-share {share:.2f}")
            if used < K * 0.5 or share > 3.0 / K:
                fail = True
    return not fail, "gold position: " + "; ".join(msgs)


@gate("G5_tier_separation")
def g5(ctx):
    """NEAR must be measurably nearer to the gold than FAR (>= +0.10 mean cosine)."""
    msgs, worst = [], 9.0
    for d, items in ctx["domains"].items():
        n = float(np.mean([it["max_gold_sim_near"] for it in items]))
        f = float(np.mean([it["max_gold_sim_far"] for it in items]))
        worst = min(worst, n - f)
        msgs.append(f"{d} near {n:.3f} far {f:.3f} sep {n-f:+.3f}")
    return worst >= 0.10, "max gold-distractor similarity: " + "; ".join(msgs)


@gate("G6_pairing_feasible")
def g6(ctx):
    """Every item must be feasible in BOTH tiers at every confirmatory K."""
    from .tiers import options_for
    msgs, fail = [], False
    for d, items in ctx["domains"].items():
        for K in ctx["ks"]:
            n = sum(1 for it in items
                    if options_for(it, "near", K) and options_for(it, "far", K))
            if n < len(items):
                fail = True
                msgs.append(f"{d}K{K} {n}/{len(items)}")
    return not fail, ("all items feasible in both tiers at every K"
                      if not fail else "INFEASIBLE CELLS: " + "; ".join(msgs))


@gate("G7_no_conflict_in_options")
def g7(ctx):
    """No adjudicated-ambiguous option may sit beside its partner as gold."""
    from .tiers import options_for
    bad = 0
    for d, items in ctx["domains"].items():
        for it in items:
            banned = set(ctx["conflicts"].get(it["gold"], ()))
            for tier in ("near", "far"):
                o = options_for(it, tier, 64)
                if o and banned & set(o):
                    bad += 1
    return bad == 0, f"{bad} option sets contain an adjudicated conflict of the gold"


@gate("G8_gold_present_once")
def g8(ctx):
    """Gold appears exactly once; no duplicate options."""
    from .tiers import options_for
    bad = 0
    for d, items in ctx["domains"].items():
        for it in items:
            for tier in ("near", "far"):
                for K in (16, 256):
                    o = options_for(it, tier, K)
                    if o is None:
                        continue
                    if o.count(it["gold"]) != 1 or len(set(o)) != len(o):
                        bad += 1
    return bad == 0, f"{bad} malformed option sets"


def run_all(ctx, log=print):
    log("=" * 72)
    log("PRE-INFERENCE GATES")
    log("=" * 72)
    results = {}
    for name, fn in GATES.items():
        try:
            ok, msg = fn(ctx)
        except Exception as e:
            ok, msg = False, f"ERROR {type(e).__name__}: {e}"
        results[name] = {"pass": bool(ok), "detail": msg}
        log(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    n_fail = sum(1 for r in results.values() if not r["pass"])
    log("=" * 72)
    log(f"{len(results)-n_fail}/{len(results)} gates passed")
    pathlib.Path("results/gates.json").write_text(json.dumps(results, indent=1))
    return n_fail == 0
