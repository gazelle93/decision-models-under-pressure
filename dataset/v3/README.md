# Decision-model evaluation dataset — v3

Frozen 2026-09-23. **1,000 items across 5 domains** (clinc, mtop, goemotions, dbpedia, fintopic); 902-option universe. Built and validated for the comparative study *Decision
Models Under Pressure* (K-scaling, order sensitivity, distractor hardness).
Reusable as-is; the Jev arm will be run against this exact freeze.

## What an item is

Each item is a text plus a gold label, and **two matched distractor pools** so
the same item can be asked with easy or hard alternatives:

```json
{"uid": "clinc:0", "domain": "clinc", "text": "can i share my location with david",
 "gold": "share location", "source": "clinc150", "leaked": false,
 "max_gold_sim_near": 0.68, "max_gold_sim_far": 0.44,
 "distractors": {"near": ["...63 options..."], "far": ["...63 options..."]}}
```

To pose a question at cardinality K, take the gold plus the first K-1
distractors of the chosen tier and shuffle (see `harness/tiers.py:options_for`,
which seeds on the item uid). Pools are **nested**: the K=8 options are a
subset of the K=16 options, so a K-curve is a within-item measurement.

## Files

| file | format | role |
|---|---|---|
| `items.jsonl` | JSON Lines | **canonical.** One object per item, pools included |
| `items.csv` | CSV | flat view for eyeballing in Excel/Sheets; no pools |
| `universe.json` | JSON | the 802-option label universe, conflicts, merges, policy |
| `gates.json` | JSON | validation results at freeze time |
| `manifest.json` | JSON | counts, seeds, prompt, split summary, sha256 of every file |
| `splits/rq*.json` | JSON | **per-RQ item indexes** — which items, tiers and K each question may use |

## Loading

```python
from harness.dataset import load_rq
items, spec = load_rq("rq3")   # 400 items, clinc+goemotions, near/far, K<=64
# spec carries the K grid, the permitted tiers, and why the scope is what it is

from harness.dataset import load
items, universe, manifest = load("v3")   # everything, unfiltered
```

**Use `load_rq`, not `load`, in experiments.** The splits are uid indexes over
the one canonical `items.jsonl` (no duplication, so they cannot drift), and
they make the fin-topic exclusion a property of the data rather than something
an analyst has to remember. Each item carries three pools:

| pool | size | used by | purpose |
|---|---|---|---|
| `near` | 63 | RQ2, RQ3 | most-similar band — hard distractors |
| `far` | 63 | RQ2, RQ3 | below-median band, surface-matched to `near` |
| `ext` | 255 | RQ1 | single pool spanning the range, for K up to 256 |

### RQ1 is a two-factor design: K **and** text length

DBpedia is included **untruncated** on purpose. Its texts run 128 → 1,300 chars
(p10 → p90) — a **10× spread inside one domain** — against clinc/mtop at ~34–38
median and goemotions at 60. Every item carries `text_chars`, so the K-curve can
be cut by length. Two notes that make this valid:

- The text-free-picker residual on DBpedia (0.245–0.260) is **uncorrelated with
  text length within the domain** (r = −0.089, p = 0.21). It inflates the
  accuracy *level*, and cannot manufacture a length effect.
- Long texts plus 255 options **will** exhaust some models' context. That is a
  measurable K × length interaction, not a defect — but truncation must be
  logged per call, or it will be mistaken for a capability finding (exactly the
  error that produced the false Laya cliff, defect F7).

### RQ3 domains: CLINC + MTOP

MTOP has the **strongest tier separation in the suite (+0.342** vs CLINC's
+0.239) because its 102 intents cluster into 11 real domains, so near-band
siblings are same-object/different-verb (`create alarm` / `delete alarm` /
`snooze alarm`) — the dense structure CLINC's flat multi-domain taxonomy lacks.
Its far tier is also the cleanest measured (text-free picker 0.060, under the
gate). Two caveats carried in the split file: MTOP is **G3 for both model
families** (intent is trained for deberta via MASSIVE/Banking77 and for Laya via
support triage), so it adds no rung diversity; and its `IN:` label prefix is
stripped at universe build — unstripped, all 102 labels share a leading token
and become a perfect tier signal. Each MTOP item also carries `mtop_domain`,
which supports a stricter *hierarchical* near tier as a robustness check on the
similarity-banded construction.

## Construction

- **Universe**: 802 label strings from 12 datasets (CLINC-150, Banking77,
  MASSIVE, GoEmotions, DBpedia L2+L3, LEDGAR, 20-News, SIB-200, AG-News,
  SNIPS, TREC-QC, fin-topics). Six further sources failed to load and are
  listed in `universe.json:failed_sources` rather than silently omitted.
  Normalization splits CamelCase and dots *before* lowercasing.
- **Tiers** are defined by **measured similarity to the gold**, not by source
  membership: NEAR is drawn from the gold's most-similar band, FAR from below
  the median. Each FAR distractor is matched one-for-one into the surface
  stratum (word count, char length, conjunction, plural) of a NEAR distractor,
  and within that stratum the least-similar option is taken. This exists so
  label *formatting* cannot signal which tier an option belongs to.
- **Ambiguity control**: an adjudicated conflict matrix (cross-source pairs
  default to conflict; same-source pairs default to keep, with an explicit
  ambiguity list — high cosine is not ambiguity: `iot hue lighton` vs
  `lightoff` are opposite actions). GoEmotions additionally excludes, per
  item, every emotion a real rater voted for on that text.
- **Positions** are seeded on the item uid, so gold position is independent of
  the gold label.

## Validation (`gates.json`)

| gate | result |
|---|---|
| G1 prompt-option collision | PASS |
| G2 format tell (far-vs-near surface AUC ≤ 0.60) | PASS — max 0.573 |
| G3 text-free gold picker (≤ 2× chance) | **FAIL** — max 0.280 |
| G4 gold-position uniformity | PASS — 60–63 of 64 slots used |
| G5 tier separation (≥ +0.10 cosine) | PASS — +0.211 to +0.238 |
| G6 both tiers feasible for every item at every K | PASS |
| G7 no adjudicated conflict in any option set | PASS |
| G8 gold present exactly once, no duplicates | PASS |

## Known limits — read before using

1. **G3 is open.** A classifier with no access to the item text finds the gold
   above 2× chance in some cells, worst at **fintopic/far (0.280 vs 0.0625
   chance)**; CLINC 0.145–0.185, GoEmotions 0.080–0.150. Report the per-cell
   picker accuracy beside any accuracy table and discount accordingly.
2. **fin-topic is excluded from RQ1 and RQ3** (pre-registration Amendment 1):
   worst format residual, a self-contradictory taxonomy, and a hypernym class
   (`general news or opinion`) that is also a legitimate gold. It is kept for
   RQ2, where a format shortcut is constant across permutations of one item.
3. **K ≤ 64 for tier contrasts** (`near`/`far`). 63 distractors is 12% of the
   universe and can honestly be called "near"; 255 would be 32% and could not.
   The `ext` pool carries 255 distractors for RQ1's K-curve to 256, but it is a
   single pool — no near/far contrast exists above K=64.
4. **Gold-string leakage**: CLINC 25%, fin-topic 5.5%, GoEmotions 3% of items
   contain the gold verbatim. The `leaked` flag is on every item — stratify
   K-curves by it.
5. **No domain is clean for every model family.** CLINC is G3 for Laya (intent
   is a trained family) and G4 for the deberta-zeroshot line; GoEmotions is the
   reverse. Annotate rungs per (model, domain); never report a bare average.
6. **Two domains for RQ3** (CLINC, GoEmotions) after Amendment 1. Thin, and
   stated as a limitation rather than padded with an untrusted domain.

## Provenance

Built by `harness/build_v3.py` (items → `harness/items.py`, pools →
`harness/tiers.py`, universe → `harness/universe.py` + `nominate.py` +
`freeze_universe.py`), gated by `harness/gates.py`, exported by
`harness/export_dataset.py`. Every defect this version fixes is documented in
`../../REVIEW-2026-09-23.md`; the two earlier attempts and why they failed are
in `../../DEVIATIONS.md`.
