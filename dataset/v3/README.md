# Decision-model evaluation dataset — v3

Frozen 2026-09-23. **1,000 items across 5 domains** (clinc, mtop, goemotions,
dbpedia, fintopic) over a 902-option label universe. Every item comes with two
matched distractor pools, so the same question can be asked with easy
alternatives or hard ones. Built for the *Decision Models Under Pressure*
comparison and reusable on its own.

## Licence and credit

**This dataset is CC BY-SA 4.0** (<https://creativecommons.org/licenses/by-sa/4.0/>).
Not a preference: MTOP is CC BY-SA 4.0, so anything redistributing its text
inherits ShareAlike, and so does anything you derive from this.

Item texts are verbatim and unmodified. `text_sha256` is a SHA-256 of each text,
so you can verify any item against its original source. **Cite the upstream
papers, not this repo** — they did the work the labels rest on.

| domain | source | licence | reference |
|---|---|---|---|
| clinc | [CLINC-150](https://huggingface.co/datasets/clinc/clinc_oos) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | Larson et al. 2019, EMNLP |
| goemotions | [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Demszky et al. 2020, ACL (Google Research) |
| mtop | [MTOP](https://huggingface.co/datasets/WillHeld/mtop) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Li et al. 2021, EACL (Meta) |
| dbpedia | [DBpedia Classes](https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | derived from DBpedia / Wikipedia |
| fintopic | [Twitter Financial News](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic) | [MIT](https://opensource.org/license/mit) | uploader-stated |

Three notes the upstream cards carry and this one should too. GoEmotions texts
are Reddit comments and some are offensive or sensitive. The MTOP mirror above
has no licence metadata of its own; CC BY-SA 4.0 is the licence of the original
Meta release (verified 2026-09-25) and is carried by the other mirrors
(`tasksource/mtop`, `SEACrowd/mtop_intent_classification`). The fintopic texts
are tweets written by third parties, so the MIT grant is the uploader's; this repo relies on that
stated licence and makes no independent claim about the underlying posts.

**What changed from upstream**, identically for all five: a seeded 200-item
subset per domain (`seed_items: 101`, in `manifest.json`); label strings
normalised (lowercased, CamelCase and dot forms split, punctuation stripped,
whitespace collapsed); item texts left untouched; and derived fields added
(`distractors`, `max_gold_sim_near`, `max_gold_sim_far`, `leaked`, `text_chars`,
`text_sha256`).

## What an item is

A real row, `mtop:15`, abridged:

```json
{"uid": "mtop:15", "domain": "mtop", "text": "Set alarm for cooking time at 7pm",
 "gold": "create alarm", "source": "mtop", "leaked": false, "text_chars": 33,
 "max_gold_sim_near": 0.8171, "max_gold_sim_far": 0.396,
 "distractors": {
   "near": ["snooze alarm", "update alarm", "delete alarm", "silence alarm", "...63 total"],
   "far":  ["lacrosse player", "figure skater", "declined transfer", "...63 total"],
   "ext":  ["...255 total, spanning the range"]}}
```

| pool | size | purpose |
|---|---|---|
| `near` | 63 | the gold's most-similar band: hard distractors |
| `far` | 63 | below-median similarity, surface-matched to `near` |
| `ext` | 255 | one pool spanning the range, for candidate lists up to 256 |

To pose a question at cardinality K, take the gold plus the first K-1
distractors of the chosen pool and shuffle
(`harness/build/tiers.py:options_for`, seeded on the item uid). Pools are
**nested**: the K=8 options are a subset of the K=16 options, so a curve across K
is a within-item measurement.

## Files

| file | format | role |
|---|---|---|
| `items.jsonl` | JSON Lines | **canonical.** One object per item, pools included |
| `items.csv` | CSV | flat view for eyeballing in Excel/Sheets; no pools |
| `universe.json` | JSON | the 902-option label universe, conflicts, merges, policy |
| `gates.json` | JSON | the eight validation checks and their per-cell numbers |
| `manifest.json` | JSON | counts, seeds, prompt, split summary, sha256 of every file |
| `splits/rq*.json` | JSON | **per-question item indexes** — which items, pools and K each one may use |

## Loading

```python
from harness.dataset import load_rq
items, spec = load_rq("rq3")   # 400 items, clinc+mtop, near/far, K<=64
# spec carries the K grid, the permitted pools, and why the scope is what it is

from harness.dataset import load
items, universe, manifest = load("v3")   # everything, unfiltered
```

**Use `load_rq`, not `load`.** The splits are uid indexes over the one canonical
`items.jsonl`, so they cannot drift from it, and they make the fin-topic
exclusion a property of the data rather than something you have to remember.

## Construction

- **Universe**: 902 label strings from 17 loaded sources (CLINC-150, Banking77,
  MASSIVE + MASSIVE scenarios, GoEmotions, dair-emotion, DBpedia-14 + DBpedia
  L2 + L3, LEDGAR, 20-News, SIB-200, AG-News, SNIPS, TREC-QC, MTOP, fin-topics).
  Two declared sources failed to load (`trec_fine`, `yahoo`) and are listed in
  `universe.json:failed_sources` rather than silently omitted. Normalisation
  splits CamelCase and dots *before* lowercasing.
- **Pools** are defined by **measured similarity to the gold**, not by source
  membership: `near` comes from the gold's most-similar band, `far` from below
  the median. Each `far` distractor is matched one-for-one into the surface
  stratum (word count, char length, conjunction, plural) of a `near` distractor,
  and within that stratum the least-similar option is taken. This exists so label
  *formatting* cannot signal which pool an option came from.
- **Ambiguity control**: an adjudicated conflict matrix. Cross-source pairs
  default to conflict; same-source pairs default to keep, with an explicit
  ambiguity list, because high cosine is not ambiguity (`iot hue lighton` vs
  `lightoff` are opposite actions and are exactly the difficulty `near` should
  contain). GoEmotions additionally excludes, per item, every emotion a real
  rater voted for on that text.
- **Positions** are seeded on the item uid, so where the gold sits does not
  depend on what the gold is.

Eight checks run as assertions at build time; results and per-cell numbers are in
`gates.json`. Seven pass. G3 does not, and limit 1 below says what that costs.

## Known limits — read before using

1. **G3 is open.** A classifier with no access to the item text finds the gold
   above the 0.125 gate in six of ten cells, against a chance rate of 0.0625.
   Worst is **dbpedia/near at 0.260**, then fintopic/far 0.250, dbpedia/far
   0.245, fintopic/near 0.190, clinc/far 0.175, clinc/near 0.155. Cleanest is
   mtop/far at 0.060, under chance. Report the per-cell picker accuracy beside
   any accuracy table and discount accordingly. The residual inflates the level
   of a curve, not its shape, and it applies to every model equally.
2. **fin-topic is excluded from two of the three questions**: worst format
   residual, a self-contradictory taxonomy, and a hypernym class (`general news
   or opinion`) that is also a legitimate gold. It is kept for the order test,
   where a format shortcut is constant across permutations of one item.
3. **K ≤ 64 for the near/far contrast.** 63 distractors is 7% of the 902-option
   universe and can honestly be called near; 255 would be 28% and could not. The
   `ext` pool carries 255 for longer candidate lists, but it is a single pool, so
   no near/far contrast exists above K=64.
4. **Gold-string leakage.** The gold string appears verbatim in the text for
   CLINC 25.0%, DBpedia 14.5%, MTOP 6.0%, fin-topic 5.5%, GoEmotions 3.0% of
   items. The `leaked` flag is on every item — stratify by it. CLINC and DBpedia
   carry the longest candidate lists, so it matters there most.
5. **No domain is clean for every model family.** CLINC is a trained family for
   Laya and not for the deberta-zeroshot line; GoEmotions is the reverse. Grade
   per (model, domain) and never report a bare average across them.
6. **Two domains for the near/far contrast** (CLINC, MTOP). Thin, and stated as a
   limitation rather than padded with a domain I did not trust.

## Provenance

Built by `harness/build/`: `universe.py` + `nominate.py` + `freeze.py` make the
option universe, `items.py` draws the items, `tiers.py` builds the pools,
`gates.py` runs the checks, `export.py` writes this folder. Reproduce with
`python -m harness.build` then `python -m harness.build.export`.

Design decisions and the rules fixed before any model ran are in
`../../PLAN.md`; the runs themselves, including deviations, are in
`../../EXPERIMENTS.md`.
