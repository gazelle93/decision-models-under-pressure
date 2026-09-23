"""Tier construction v3 (rebuilt after the 2026-09-23 review).

v2 defined tiers by SOURCE membership. That failed four ways:
  - no semantic separation on CLINC (+0.007 mean cosine; CLINC-150 is a
    multi-domain intent set, so "sibling" does not mean "near")
  - a format tell: FAR pools drew from concatenated-label sources, NEAR did
    not, letting a text-free classifier locate the gold (AUC up to 0.94)
  - infeasible NEAR beyond K=16 for any small-label domain (0/200 at K=64)
  - unpaired contrasts where NEAR dropped items (fintopic 97/200, biased)

v3 defines tiers by MEASURED gold-similarity band, drawn from the whole
universe, with FAR sampled to match NEAR's surface strata so format carries
no tier information. Both tiers are always feasible for the same items, so
every contrast is paired.
"""
from __future__ import annotations

import json
import pathlib
import random
from collections import defaultdict

FILTER_MODELS = ("sentence-transformers/all-mpnet-base-v2", "intfloat/e5-large-v2")
NEAR_BAND = 0.15      # top 15% most similar options to the gold
FAR_BAND = 0.35       # far candidates start below the median similarity
MIN_STRATUM = 0       # 0 = keep all items. Dropping thin-stratum items was
                      # tried and REJECTED: it shrinks n without closing gate
                      # G3, and the survivors are a biased (format-typical)
                      # subsample. The residual is disclosed instead.
EXT_K = 256           # RQ1 extended K-curve: a single pool, no tier
                      # contrast, so the near/far matching constraint does
                      # not apply — it only must not leak the gold.
MAX_K = 64            # tier-contrast ceiling. A "near" band must stay near:
                      # 63 distractors is 12% of a 532-option universe, which
                      # is defensible; 255 would be 48% and would not be. The
                      # K-curve beyond 64 runs on the far tier only (RQ1),
                      # where the pool is large. Raising this needs a bigger
                      # universe, not a wider band.


def load_universe(version="v3"):
    v = json.loads(pathlib.Path(f"results/universe_{version}.json").read_text())
    options = sorted(v["options"])
    sources = {k: set(x) for k, x in v["options"].items()}
    conflicts = defaultdict(set)
    for a, b in v["conflicts"]:
        conflicts[a].add(b)
        conflicts[b].add(a)
    alias_of = {a: c for c, als in v["merges"].items() for a in als}
    return options, sources, dict(conflicts), alias_of


def surface_bucket(o):
    """Stratum for format matching.

    Stratify on exactly the features the format gate tests, or the gate finds
    the one that was left out: v3.0 matched only (word count, char bin) and
    fin-topic stayed legible at AUC 0.64 because several of its labels contain
    a conjunction ('company or product news') and no far option was matched on
    that. Keys: word count, char bin, conjunction, trailing -s.
    """
    return (min(len(o.split()), 5), min(len(o) // 6, 5),
            (" or " in o) or ("|" in o) or ("&" in o), o.endswith("s"))


def embed(strings, model_id, device=None):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer(model_id, device=device)
    pre = "passage: " if "e5" in model_id else ""
    out = np.asarray(m.encode([pre + s for s in strings], normalize_embeddings=True, batch_size=64))
    del m
    return out


def goemotions_rater_alternatives():
    """Every emotion at least one human rater assigned to each text.

    The single-label filter removes AGGREGATION disagreement, not ANNOTATOR
    disagreement: the semantic audit found 52% of near-tier items offered a
    distractor a real rater had voted for on that exact text. Those are not
    distractors, they are second correct answers, so they are excluded per item.
    """
    from datasets import load_dataset
    raw = load_dataset("google-research-datasets/go_emotions", "raw", split="train")
    names = [c for c in raw.column_names if c not in
             ("text", "id", "author", "subreddit", "link_id", "parent_id",
              "created_utc", "rater_id", "example_very_unclear")]
    alt = defaultdict(set)
    for r in raw:
        t = r["text"]
        for n in names:
            if r[n] == 1:
                alt[t].add(n.replace("_", " "))
    return alt


def build_pools(items, universe, sources, conflicts, alias_of, seed=101,
                text_exclude_frac=0.03, log=print):
    """items: [{text, gold, source, domain}] -> adds tiers {near, far} nested lists.

    Exclusions, in order:
      1. type-level conflicts of the gold (the adjudicated ambiguity matrix)
      2. per-item rater-attested alternatives (GoEmotions only, objective)
      3. rank-based text-similarity exclusion, applied as the SAME FRACTION of
         each tier's candidate pool (v2 removed a fixed 10 per item, which cost
         NEAR up to 23.7% of its pool and FAR only 2%)
    """
    import numpy as np
    log(f"  embedding {len(universe)} options x {len(FILTER_MODELS)} filter models")
    opt_emb = [embed(universe, m) for m in FILTER_MODELS]
    txt_emb = [embed([it["text"] for it in items], m) for m in FILTER_MODELS]
    idx = {o: i for i, o in enumerate(universe)}
    by_bucket = defaultdict(list)
    for o in universe:
        by_bucket[surface_bucket(o)].append(o)

    rater_alt = {}
    if any(it["domain"] == "goemotions" for it in items):
        rater_alt = goemotions_rater_alternatives()
        log(f"  loaded rater-attested alternatives for {len(rater_alt)} GoEmotions texts")

    # An item can only be constructed format-neutrally if the universe actually
    # populates its gold's surface stratum. Items whose stratum is too thin are
    # dropped and logged, never quietly padded from another stratum.
    supply = defaultdict(int)
    for o in universe:
        supply[surface_bucket(o)] += 1
    thin = [it for it in items if supply[surface_bucket(it["gold"])] < MIN_STRATUM]
    if thin:
        from collections import Counter as _C
        log(f"  dropping {len(thin)} items whose gold stratum has <{MIN_STRATUM} options: "
            f"{dict(_C(it['domain'] for it in thin))}")
        keep = [it for it in items if supply[surface_bucket(it["gold"])] >= MIN_STRATUM]
        items[:] = keep

    missing = [it for it in items if it["gold"] not in idx]
    if missing:
        raise SystemExit(
            f"{len(missing)} items have a gold absent from the universe "
            f"(e.g. {[m['gold'] for m in missing[:3]]}). Golds must never be "
            "silently dropped — fix the universe or the item source.")

    stats = defaultdict(int)
    for i, it in enumerate(items):
        gold = it["gold"]
        gi = idx[gold]
        # similarity to the gold, averaged over both filter models (no single
        # filter family gets to define the bands — v2's single mpnet filter
        # deleted 83-87% of the embedding models' competitors vs 49% of the
        # NLI model's)
        sims = np.mean([e @ e[gi] for e in opt_emb], axis=0)

        banned = {gold} | set(conflicts.get(gold, ()))
        if it["domain"] == "goemotions":
            alts = rater_alt.get(it["text"], set())
            alts = {alias_of.get(a, a) for a in alts}
            n_before = len(banned)
            banned |= (alts & set(universe)) - {gold}
            stats["rater_excluded"] += len(banned) - n_before

        order = np.argsort(-sims)
        cand = [universe[j] for j in order if universe[j] not in banned]
        n_near = max(int(len(cand) * NEAR_BAND), MAX_K)
        near_pool = cand[:n_near]
        far_pool = cand[int(len(cand) * FAR_BAND):]   # bottom half by gold similarity

        # rank-based text-similarity exclusion, equal fraction per tier
        tsim = np.mean([e[i] @ oe.T for e, oe in zip(txt_emb, opt_emb)], axis=0)
        def drop_nearest(pool):
            k = max(1, int(len(pool) * text_exclude_frac))
            worst = sorted(pool, key=lambda o: -tsim[idx[o]])[:k]
            return [o for o in pool if o not in worst], len(worst)
        near_pool, dn = drop_nearest(near_pool)
        far_pool, df = drop_nearest(far_pool)
        stats["near_text_dropped"] += dn
        stats["far_text_dropped"] += df

        # Prefer distractors in the GOLD's own surface stratum. Without this the
        # gold is surface-distinct from its own distractors (87.5% of GoEmotions
        # golds sat in one bucket vs 45% of distractors), so a text-free
        # classifier locates it above chance — review gate G3.
        gb = surface_bucket(gold)
        same_b = [o for o in near_pool if surface_bucket(o) == gb]
        other_b = [o for o in near_pool if surface_bucket(o) != gb]
        near = (same_b + other_b)[:MAX_K - 1]
        stats["near_same_bucket"] += sum(1 for o in near if surface_bucket(o) == gb)
        stats["near_total"] += len(near)
        # FAR: one option per NEAR option, same surface stratum, and within the
        # stratum take the LEAST similar available. Exact format matching and
        # maximum separation at once; a random pick inside the stratum (v3.0)
        # let far drift toward the near cutoff and still missed buckets.
        far_by_bucket = defaultdict(list)
        for o in sorted(far_pool, key=lambda x: sims[idx[x]]):   # ascending similarity
            far_by_bucket[surface_bucket(o)].append(o)
        far_set, used = [], set()
        for o in near:
            b = surface_bucket(o)
            pick = next((c for c in far_by_bucket.get(b, []) if c not in used), None)
            if pick is None:
                for db in sorted(far_by_bucket, key=lambda x: abs(x[0]-b[0]) + abs(x[1]-b[1])):
                    pick = next((c for c in far_by_bucket[db] if c not in used), None)
                    if pick is not None:
                        stats["bucket_fallback"] += 1
                        break
            if pick is not None:
                far_set.append(pick)
                used.add(pick)
        # RQ1 extended pool: 255 distractors spanning the similarity range,
        # preferring the gold's surface stratum so format stays uninformative.
        ext_cand = [o for o in cand if o not in set(near) | set(far_set)]
        ext_same = [o for o in ext_cand if surface_bucket(o) == gb]
        ext_other = [o for o in ext_cand if surface_bucket(o) != gb]
        ext = (ext_same + ext_other)[:EXT_K - 1]
        stats["ext_short"] += 1 if len(ext) < EXT_K - 1 else 0

        it["tiers"] = {"near": near, "far": far_set, "ext": ext}
        it["max_gold_sim_near"] = float(max((sims[idx[o]] for o in near[:63]), default=0))
        it["max_gold_sim_far"] = float(max((sims[idx[o]] for o in far_set[:63]), default=0))
        stats["n"] += 1
    log(f"  pools built for {stats['n']} items | rater-excluded {stats['rater_excluded']} | "
        f"text-dropped near {stats['near_text_dropped']} far {stats['far_text_dropped']} | "
        f"bucket fallbacks {stats['bucket_fallback']} | ext-short {stats['ext_short']} | "
        f"distractors in gold's stratum {stats['near_same_bucket']/max(stats['near_total'],1):.1%}")
    return items


def options_for(item, tier, K, seed=101):
    """Gold + K-1 distractors, shuffled with an ITEM-INDEXED seed.

    v2 seeded on crc32(gold+tier)+K with no item identifier, so every item
    sharing a gold label placed the gold at the same position (review F2).
    """
    d = item["tiers"][tier]
    if len(d) < K - 1:
        return None
    opts = [item["gold"]] + d[: K - 1]
    rng = random.Random(f"{seed}|{item['uid']}|{tier}|{K}")
    rng.shuffle(opts)
    return opts
