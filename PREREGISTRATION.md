# Stage-2 pre-registration — Decision Models Under Pressure

STATUS: APPROVED by Mingyou 2026-09-22 (in-session) — BINDING. Jev arm deferred (no API credentials). Written 2026-09-22, before any
stage-2 inference. Deviations go in a deviations log, not in silent edits.

## Scope

Confirmatory runs for RQ1 (K-scaling), RQ2 (order sensitivity), RQ3
(distractor hardness) over as-shipped systems. Pilot evidence (waves 1-3.5,
n=50) generated the hypotheses; none of it counts toward confirmation.

## Roster (two models per class)

- A1 pair cross-encoders: deberta-v3-base-zeroshot-v2.0; deberta-v3-large-
  zeroshot-v2.0 (large runs full at K<=64; 150-item subsample at K in
  {128,256} — cost, pre-registered).
- A2 embedding scorers: bge-large-en-v1.5; thenlper/gte-large.
- A3 option-conditioned: laya (extended ctx 2048/1024, direct-forward
  adapter); gliclass-large-v3.0 (raw-distribution adapter).
- Jev appended when API credentials exist (K capped at its 255-option API
  limit; hosted latency in its own column; rung ungradable).
- bart-large-mnli dropped: HF-default reference only, below majority on 2/6
  pilot sets; keeping it adds cost, not information.

## AMENDMENT 1 (2026-09-23): fin-topic excluded from RQ3; Jev-arm constraints

**fin-topic is dropped from the confirmatory tier contrast (C2/RQ3).** It is
retained in the built dataset and reported as exploratory only. Reasons, all
measured before the decision:
- Worst text-free-gold-picker cell in the suite: 0.280 at far tier vs 0.0625
  chance (4.5x), against CLINC's 0.185 and GoEmotions' 0.150. The cause is
  structural and unfixable with a natural label universe: fin-topic's label
  grammar ('gold or metals or materials', 'treasuries or corporate debt') has
  no dense surface stratum to match a far tier into. Both candidate remedies
  were tried and rejected (coarser strata reopen the format gate at AUC 0.681;
  dropping thin-stratum items costs 90/200 items and makes the picker worse).
- The semantic audit found the taxonomy self-contradictory on its own items
  (two same-shape items labelled 'legal or regulation' and 'general news or
  opinion'), 14% strict near-tier ambiguity, and unanswerable items ('#OOTT').
- 'general news or opinion' is a residual hypernym that is ALSO a legitimate
  gold class, so it cannot be removed as a magnet; everything leaks into it.

RQ3 therefore runs on CLINC + GoEmotions. Two domains is thin, and that is
stated as a limitation rather than papered over with a domain we do not trust.
Note the two have COMPLEMENTARY contamination, which is itself worth
reporting: CLINC is G3 for Laya (intent is a trained family) and G4 for the
deberta line; GoEmotions is G3 for the deberta line (three emotion datasets in
its v1.1 mixture) and G4* for Laya. No domain in this space is clean for every
family at once.

fin-topic REMAINS in RQ2 (order sensitivity) as a supporting domain, since
flip rate is measured across permutations of an identical option set, where a
format shortcut is constant within the item. It is excluded from RQ1 headline
K-curves for the same inflation reason as RQ3.

**Jev-arm constraints (when credentials exist).** The dataset is frozen before
any Jev call, because Jev calls are not re-runnable, cost money, and burn the
items into the reserve list the moment they are sent. Consequences:
- Send only cells we trust: CLINC + GoEmotions, K <= 64 (well inside Jev's
  255-option API cap).
- Jev rounds probabilities to 2 decimals, so its NLL/Brier/ECE are
  rounding-limited and flagged as such; accuracy and flip rate are unaffected.
- Hosted latency stays in its own column and is never compared to local bs=1.
- Rung: ungradable (training data undisclosed).

## Items

- Gold domains x 500 items each, fresh draws (seed 101): CLINC-150 intents
  (test split MINUS the 50 pilot items), GoEmotions single-label rows,
  twitter-financial-news-topic (validation).
- Rung annotation per (model, domain); CLINC is G3 for Laya (trained family)
  and stated as such wherever cited.
- Universe: frozen universe_v2.json (537 canonical options, 100 conflicts,
  8 aliases). Per-item exclusion: conflict matrix + UNION of top-10
  text-nearest options from TWO filter models (all-mpnet-base-v2,
  intfloat/e5-large-v2 with query/passage prefixes) — neither in the roster.

## Conditions

- K grid (confirmatory): {2, 4, 8, 16, 32, 64, 128, 256}. K=512+ exploratory
  only, pending universe v3.
- Distractor tiers: FAR (outside the gold's source dataset), NEAR (same-source
  siblings minus conflicts), MIXED (uniform draw, pilot-comparable). Each tier
  reports its distractor-to-gold cosine distribution (computed by BOTH filter
  models; committed script).
- Order: 5 permutations per item at K in {16, 64}; CRC32-stable seeds.
- NLI accuracy cells use internal pair-batching (batch_size 32); bs=1 latency
  comes only from dedicated probes.

## Pre-registered contrasts and decision rules

C1 — Residual order tax (RQ2). Pooled 1,500 items x 5 orders at K=16, clean
near+far sets. Per A3 model: claim "residual tax under 5%" iff the 95%
bootstrap CI upper bound on flip rate < 0.05. Structural-zero classes (A1,
A2) reported as harness checks.

C2 — H3, option-option attention vs near distractors (RQ3). Delta_c =
acc_far - acc_near per class at K in {16, 32, 64} (below the A3 context
ceiling). H3 SUPPORTED iff Delta_A3 < Delta_A2 AND Delta_A3 < Delta_A1 with
95% CIs on the differences excluding 0 at >=2 of 3 K values. H3 FALSIFIED iff
Delta_A3 >= Delta_A2 at all three K with CI support. Anything else: mixed,
reported as such. (Pilot counter-signal on record: Laya was the most
near-density-sensitive model at K>=128, ceiling-confounded.)

C3 — K-curve class separation (RQ1). Slope of accuracy vs log2 K per class,
FAR tier, K 16->256; bootstrap CIs on pairwise slope differences. Descriptive:
no binary rule, CIs speak.

Secondary (reported, not confirmatory): macro-F1, top-3/5, NLL, Brier,
ECE-15 beside accuracy; calibration-vs-K; gold-in-support rates; latency
(direct-forward and wrapper separately for laya); chance + majority baselines
on every table, skewed sets as lift.

## Analysis

Bootstrap 2,000 resamples over items; paired comparisons (identical items
across models); McNemar for pairwise accuracy at fixed cells. Per-item JSONL
retained and versioned.

## Ladder and stopping

N=200/domain first; a contrast may be dropped only when its CI excludes the
interesting effect; otherwise complete to 500/domain. Estimated compute:
1-2 days background on this machine.

## Sign-off

- [x] Mingyou approval (date): 2026-09-22
