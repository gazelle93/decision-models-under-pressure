# What I decided to measure, before measuring it

Written 2026-09-23, before a single model ran against dataset v3. Fixed from
that point: anything I changed afterwards is logged as a dated deviation in
`EXPERIMENTS.md` rather than edited quietly into this file.

The point of writing it down first is that it stops me choosing the comparison
that flatters the conclusion I already liked. Two of the rules below fired
against me, which is the only reason they were worth having.

Data: the frozen `dataset/v3/`, loaded through `harness.dataset.load_rq`.
Checksums in `dataset/v3/manifest.json`. Gate status in
`dataset/v3/gates.json` — **G3 is open** and its per-cell magnitude is
reported beside every accuracy table.

## Roster

Two models per family, so no class claim rests on one checkpoint.

- **A1** pair cross-encoders: `deberta-v3-base-zeroshot-v2.0`,
  `deberta-v3-large-zeroshot-v2.0` (large runs full at K ≤ 64 and a fixed
  150-item prefix above, on the same prefix at every K so slopes stay paired)
- **A2** embedding scorers: `bge-large-en-v1.5`, `thenlper/gte-large`
- **A3** option-conditioned: `convaiinnovations/laya`, `gliclass-large-v3.0`
- **Jev** appended to A3 when credentials exist: K ≤ 64, hosted latency in its
  own column, calibration flagged rounding-limited (2-decimal API), rung
  ungradable.

## RQ1 — cardinality × text length

Domains clinc, mtop, goemotions, dbpedia. Pool `ext`. K ∈ {2,4,8,16,32,64,128,256}.

DBpedia is included **untruncated** and supplies the length range (128 → 1300
chars p10→p90, a 10× spread *within* one domain) against clinc/mtop ~34–38.
Every item carries `text_chars`.

Report accuracy vs log₂K, vs length quartile, and the K × length interaction.

Pre-recorded validity notes:
- DBpedia's text-free-picker residual (0.245–0.260) is uncorrelated with text
  length within the domain (r = −0.089, p = 0.21), so it inflates the accuracy
  *level* and cannot manufacture a length effect.
- **Truncation is logged per call.** Long texts with 255 options will exhaust
  some context budgets; that is a real interaction, but unlogged it is
  indistinguishable from a capability finding.
- CLINC's gold-string leakage (25%) interacts with K, so its curves are
  reported split by the `leaked` flag.

## RQ2 — order sensitivity

Domains clinc, goemotions, fintopic, mtop. Tiers near and far. K ∈ {16, 64},
5 permutations per item.

fin-topic is retained here alone: flip rate compares permutations of one
identical option set, so a format shortcut is constant within the item and can
neither manufacture nor mask order sensitivity.

**C1.** Per A3 model, claim "residual order tax under 5%" only if the 95%
bootstrap CI upper bound on flip rate is below 0.05. A1 and A2 are structurally
order-invariant; their measured rate is a harness check, and any non-zero value
must be explained (exact score ties are a known, benign cause).

Items whose permutation set is incomplete are excluded from the rate, and the
count of failed permutations is reported — dropping them silently biases flip
rates downward, the direction that makes this contrast easier to pass.

## RQ3 — distractor hardness

Domains clinc, mtop. Tiers near and far, paired on the same items.
K ∈ {2,4,8,16,32,64}.

MTOP carries this question: strongest measured tier separation in the suite
(+0.342 vs CLINC's +0.239), because its 102 intents cluster into 11 real
domains. Pre-recorded caveat: MTOP is **G3 for both families** (intent is
trained for deberta via MASSIVE/Banking77 and for Laya via support triage), so
it adds no rung diversity. GoEmotions may be re-added only as a dated amendment.

**C2 — does option–option attention help against near distractors?**
Δ_family = acc_far − acc_near at K ∈ {16, 32, 64}.
- **SUPPORTED** iff Δ_A3 < Δ_A1 and Δ_A3 < Δ_A2, with 95% CIs on the
  differences excluding 0, at ≥2 of 3 K values.
- **FALSIFIED** iff Δ_A3 ≥ Δ_A2 at all three K with CI support.
- Anything else is reported as mixed.

Pre-recorded counter-signal: in pilot work the option-conditioned models were
*more* near-sensitive, not less. Both outcomes are publishable; neither is a
surprise to be explained away afterwards.

**Within-family variance is reported before any between-family claim.** If the
two A3 models differ more from each other than the families differ, C2 is
reported as inapplicable — the architecture class would not be the operative
variable.

**C3 — K-curve separation.** Slope of accuracy vs log₂K per family, far tier,
with bootstrap CIs on pairwise slope differences. Descriptive; no binary rule.

## Metrics and analysis

Accuracy, top-3/5, NLL, Brier, 15-bin ECE **reported beside accuracy, never
alone**, plus chance and majority baselines on every table; skewed domains are
reported as lift over majority. Per-item JSONL at full precision. Bootstrap
2,000 resamples over items; paired comparisons on identical items; McNemar for
pairwise accuracy at fixed cells. Latency: local batch-1 and hosted end-to-end
never share a column, and wrapper overhead is separated from model time.

## Ladder and stopping

n=200 per domain first, then 500. Items nest by construction (the draw is taken
once at max_n and truncated). A contrast may be dropped only when its CI
excludes the interesting effect; otherwise complete to 500.

## What would make this boring

Stated in advance so it cannot be rationalised later: if every family lands
within CI of the others on all three questions, the finding is that
architecture does not matter at this scale for these tasks — and that is the
result I post.
