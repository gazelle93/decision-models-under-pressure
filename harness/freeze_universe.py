"""Freeze the option universe (v3).

Policy, corrected after the 2026-09-23 review:

CROSS-source nominations -> CONFLICT by default. Two labels from different
  taxonomies that are near-identical were never adjudicated against each other
  by anyone, so they are presumed interchangeable.

SAME-source nominations -> KEEP by default, ban only if listed in AMBIGUOUS.
  v2 exempted same-source pairs entirely (leaving the whole NEAR tier
  unadjudicated); the naive fix of banning them all is equally wrong, because
  high cosine does not mean ambiguous: 'iot hue lighton' vs 'iot hue lightoff'
  are opposite actions, trivially separable from text, and are exactly the
  legitimate difficulty the NEAR tier exists to measure. Conversely the pairs
  that DO create ties ('earnings' vs 'financials') sit at modest cosine.
  Embeddings cannot see this distinction, so it is an explicit list.

AMBIGUOUS is sourced from the semantic audit (2026-09-23), which read items and
  reported per-pair tie rates.
"""
from __future__ import annotations

import json
import pathlib

VERSION = "v3"

# canonical <- aliases (pure surface variants of one concept)
MERGES = {
    "definition": ["definitions"],
    "change pin": ["pin change"],
    "insurance": ["insurances"],
    "weather": ["getweather", "weather query", "get weather"],
    "share location": ["sharecurrentlocation", "share current location"],
    "recipe": ["cooking recipe"],
    "meeting schedule": ["schedule meeting"],
}

# Pairs judged interchangeable-for-some-item by the semantic audit. These may
# never co-occur as gold + distractor. Tier-1 findings, verbatim rates in
# REVIEW-2026-09-23.md.
AMBIGUOUS = [
    # fintopic: undocumented taxonomy conventions, demonstrated inconsistent
    ("earnings", "financials"),
    ("macro", "politics"),
    ("macro", "general news or opinion"),
    ("legal or regulation", "general news or opinion"),
    ("company or product news", "general news or opinion"),
    ("stock commentary", "general news or opinion"),
    # goemotions: intensity/target distinctions with no textual cue
    ("annoyance", "anger"),
    ("annoyance", "disapproval"),
    ("anger", "disapproval"),
    ("curiosity", "confusion"),
    ("approval", "admiration"),
    ("caring", "love"),
    ("realization", "surprise"),
    ("sadness", "grief"),
    ("nervousness", "fear"),
    # cross-source twins the v2 matrix missed
    ("traffic", "gettrafficinformation"),
    ("traffic", "get traffic information"),
    ("shopping list", "lists query"),
    ("todo list", "lists query"),
    ("directions", "transport query"),
]

# Hypernym magnets: true of almost any item in some domain, so they create
# defensible-but-unintended answers. Removed from the universe entirely.
# NOTE: 'general news or opinion' is NOT here — it is a legitimate fin-topic
# gold class. Its hypernym ambiguity is handled by explicit AMBIGUOUS pairs
# instead, so it can be a gold but never a distractor for those partners.
MAGNETS = {"general", "miscellaneous", "business", "world", "general quirky",
           "other", "misc"}


def main():
    out = pathlib.Path("results")
    draft = json.loads((out / "universe_draft.json").read_text())
    noms = [l for l in (out / "universe_cross_nominations.md").read_text().splitlines()
            if l.startswith("- [")]

    pairs = []
    for l in noms:
        body = l.split("]", 1)[1]
        same = body.strip().startswith("SAME")
        body = body.replace("SAME", "", 1).replace("CROSS", "", 1).replace("CONTAIN", "", 1)
        a = body.split("'")[1]
        b = body.split("  vs  ")[1].split("'")[1]
        pairs.append((a, b, same))

    alias_of = {a: c for c, als in MERGES.items() for a in als}
    options = {}
    for k, v in draft["options"].items():
        canon = alias_of.get(k, k)
        if canon in MAGNETS:
            continue
        options.setdefault(canon, set()).update(v["sources"])

    amb = {tuple(sorted((alias_of.get(a, a), alias_of.get(b, b)))) for a, b in AMBIGUOUS}
    conflicts, kept_same = set(), 0
    for a, b, same in pairs:
        ca, cb = alias_of.get(a, a), alias_of.get(b, b)
        if ca == cb or ca not in options or cb not in options:
            continue
        key = tuple(sorted((ca, cb)))
        if same and key not in amb:
            kept_same += 1
            continue          # legitimate sibling difficulty, keep
        conflicts.add(key)
    # explicit ambiguity bans, whether or not the embedding nominated them
    for key in amb:
        if key[0] in options and key[1] in options:
            conflicts.add(key)

    v3 = {
        "version": VERSION,
        "frozen": "2026-09-23",
        "adjudication": "cross-source: default-conflict; same-source: default-keep with an "
                        "explicit AMBIGUOUS list from the 2026-09-23 semantic audit",
        "policy": {
            "cross_source_threshold": 0.60,
            "same_source_threshold": 0.75,
            "same_source_default": "keep",
            "filter_model": "sentence-transformers/all-mpnet-base-v2",
            "magnets_removed": sorted(MAGNETS),
            "normalization": "camelCase/dot-aware split before lowercase (v3 fix)",
        },
        "failed_sources": draft.get("failed_sources", []),
        "n_options": len(options),
        "n_merges": sum(len(v) for v in MERGES.values()),
        "n_conflicts": len(conflicts),
        "n_same_source_kept": kept_same,
        "merges": MERGES,
        "ambiguous_pairs": [list(k) for k in sorted(amb)],
        "options": {k: sorted(v) for k, v in sorted(options.items())},
        "conflicts": sorted(list(p) for p in conflicts),
    }
    (out / f"universe_{VERSION}.json").write_text(json.dumps(v3, indent=1))
    print(f"universe_{VERSION}: {v3['n_options']} options, {v3['n_conflicts']} conflicts "
          f"({len(amb)} explicit ambiguity bans), {kept_same} same-source pairs kept as "
          f"legitimate near difficulty, {len(MAGNETS)} magnets removed, "
          f"failed sources {v3['failed_sources']}")


if __name__ == "__main__":
    main()
