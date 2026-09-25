# Experiment log

Running record for the v3 runs. Governed by `PLAN.md`. The design came out of
an audit that found the previous dataset could not support the question it was
built for; what it found is summarised under "How the dataset was checked" in
the README, and the fixes are the eight gates below.

Nothing above a pilot label may be cited as a result. Every entry states n, the
dataset version, and any deviation from the pre-registration.

---

## Dataset v3 — frozen 2026-09-23

1,000 items across 5 domains (clinc, mtop, goemotions, dbpedia, fintopic) over
a 902-option universe from 17 loaded sources (2 declared sources failed). Per-RQ splits in `dataset/v3/splits/`.

**Gates: 7 of 8 pass.**

| gate | result |
|---|---|
| G1 prompt-option collision | PASS |
| G2 format tell (far-vs-near surface AUC ≤ 0.60) | PASS — max 0.584 |
| G3 text-free gold picker (≤ 2× chance) | **FAIL** — max 0.260 (dbpedia) |
| G4 gold-position uniformity | PASS — 60–63 of 64 slots |
| G5 tier separation (≥ +0.10 cosine) | PASS — +0.212 to +0.342 |
| G6 both tiers feasible, every item, every K | PASS |
| G7 no adjudicated conflict in any option set | PASS |
| G8 gold present exactly once | PASS |

Tier separation by domain: mtop **+0.342**, dbpedia +0.264, clinc +0.239,
fintopic +0.230, goemotions +0.212.

G3 residual by domain (chance 0.0625): mtop far **0.060**, goemotions near
0.080, mtop near 0.140, clinc 0.155–0.175, fintopic 0.190–0.250, dbpedia
0.245–0.260. Reported per cell beside every accuracy table.

## Harness readiness — 2026-09-23

Verified before any run:

- **Laya option rendering fixed.** Options now pass as a list; the dict form
  rendered `"transfer: transfer"`, doubling option tokens and truncating
  content at high K while the marker count stayed correct. Verified on the
  longest DBpedia item (585 text tokens): K=256 builds a 1,293-token sequence
  with full 256/256 support and no truncation.
- **Full-precision probabilities** in the per-item log.
- **Truncation logged per call** for laya and gliclass (required by RQ1).
- **Failure counters** for both accuracy calls and order permutations; a cell
  where everything raised is no longer indistinguishable from an infeasible one.
- Runner reads the frozen dataset via `load_rq`, never a rebuild.

## Runs

### RQ3 / C2 — near vs far distractors — 2026-09-23, n=400 (clinc 200 + mtop 200)

Dataset v3, 6 models x 2 domains x 2 tiers x K in {2..64} = 28,800 decisions,
47 min wall clock. Paired per item, bootstrap 2,000 resamples.
Delta = acc(far) - acc(near); a LARGER Delta means near distractors hurt more.

| model | fam | K=16 | K=32 | K=64 |
|---|---|---|---|---|
| deberta-v3-base-zeroshot | A1 | +0.205 [.16,.25] | +0.220 [.17,.27] | +0.278 [.23,.33] |
| deberta-v3-large-zeroshot | A1 | +0.140 [.09,.19] | +0.177 [.12,.23] | +0.235 [.18,.29] |
| bge-large-en-v1.5 | A2 | +0.205 [.17,.24] | +0.237 [.20,.28] | +0.307 [.27,.35] |
| gte-large | A2 | +0.205 [.17,.25] | +0.223 [.18,.27] | +0.258 [.22,.30] |
| gliclass-large-v3.0 | A3 | +0.210 [.17,.25] | +0.260 [.21,.30] | +0.357 [.31,.41] |
| laya | A3 | +0.135 [.10,.17] | +0.198 [.16,.24] | +0.345 [.30,.40] |

**The pre-registered applicability gate fired.** Within-family spread exceeds
between-family spread at K=16 (0.075 vs 0.033) and K=32 (0.062 vs 0.031), so
the architecture class is NOT the operative variable at those cardinalities and
C2 is reported inapplicable as a family-level claim, exactly as written. At
K=64 the families do cohere (within 0.050 < between 0.095).

**H3 is not supported.** The hypothesis was that option-option attention should
help against near distractors, so A3 should show the SMALLEST Delta. Observed:
at K=64, where the family comparison is admissible, A3 has the LARGEST penalty
(+0.351) against A2 (+0.282) and A1 (+0.256). The formal FALSIFIED condition
(Delta_A3 >= Delta_A2 at all three K) is not met either, because A3 is below A2
at K=16 — so the verdict is MIXED, with the admissible cell pointing opposite
to H3.

**The cleaner finding is the trend.** Every model is hurt by near distractors
(+0.14 to +0.36), the penalty grows with K for all six, and it grows FASTEST
for the option-conditioned pair: laya +0.135 -> +0.345 and gliclass +0.210 ->
+0.357 across K=16->64, against +0.14->+0.24 for deberta-large. Whatever
option-option attention buys, it does not buy robustness to semantically close
candidates, and the deficit widens as the candidate set grows.

**Caveats.** n=400, one run; the pre-registered ladder continues to 500.
Both domains are G3 for Laya and for the deberta line (intent is a trained
family for both), so no cell here is a G4-grade transfer claim. The open G3
residual differs by domain (mtop far 0.060 is the cleanest cell in the suite,
clinc 0.155-0.175), and models differ in how readily they exploit format, so
the far-tier side of every Delta carries that uncertainty.


### RQ2 / C1 — order sensitivity — 2026-09-24, n=800 (clinc, goemotions, fintopic, mtop)

Dataset v3, 6 models x 4 domains x 2 tiers x K in {16,64} x 5 permutations.
48 cells, ~9h wall clock, n=1600 per (model, K) cell, **0 failed permutations**
so no downward bias. A flip = the argmax changed when only the option ORDER
changed; the option set is identical.

| model | fam | K=16 flip [95% CI] | K=64 flip [95% CI] |
|---|---|---|---|
| deberta-v3-large-zeroshot | A1 | 0.0000 [.0000,.0000] | 0.0000 [.0000,.0000] |
| bge-large-en-v1.5 | A2 | 0.0000 [.0000,.0000] | 0.0000 [.0000,.0000] |
| deberta-v3-base-zeroshot | A1 | 0.0006 [.0000,.0019] | 0.0019 [.0000,.0044] |
| gte-large | A2 | 0.0119 [.0069,.0175] | 0.0200 [.0131,.0275] |
| gliclass-large-v3.0 | A3 | **0.1569** [.1394,.1737] | **0.2819** [.2594,.3044] |
| laya | A3 | **0.2062** [.1869,.2263] | **0.4938** [.4706,.5181] |

**C1 FAILS decisively for both option-conditioned models at both K.** The bar
was a 95% CI upper bound below 0.05; the observed lower bounds are 0.139 and
0.187 at K=16 and 0.259 and 0.471 at K=64. **At K=64 Laya changes its answer on
roughly half of all decisions purely from option ordering**, with the option
set held identical.

**Harness check passes.** A1 and A2 score each option independently, so order
cannot exist for them: deberta-large and bge are exactly 0.0000. The two
non-zero cases are exact ties resolved by position, verified directly —
**32 of 32** gte flipping items at K=64 have an exact top-2 tie (gap < 1e-9) in
their probability vector. Structural invariance holds; ties are the only crack.

#### Amendment, 2026-09-25 — determinism control (not pre-registered)

The flip measure above sends five calls that differ in two ways: the option
order changed, and they are five separate calls. Nothing in the original plan
separated those. This arm holds the order fixed at permutation 0 and calls five
times anyway, so whatever flips is the system disagreeing with itself. Run after
the fact, so it is an amendment and not a pre-registered check; the arm is
`--det` in `harness/run/jev.py` and `harness/run/determinism.py`, K=64 only.

| model | n | shuffled | fixed | gap | 95% CI on gap |
|---|---|---|---|---|---|
| Jev | 1600 | 0.1456 | **0.0431** | +0.1025 | [+0.0862, +0.1187] |
| laya | 3200 | 0.4938 | **0.0000** | +0.4938 | — |

Per cell, Jev's gap clears zero everywhere: clinc far +0.015 [.000,.035], clinc
near +0.070 [.035,.110], mtop far +0.025 [.005,.050], mtop near +0.085
[.050,.125], fintopic far +0.040 [.015,.070], fintopic near +0.130 [.075,.185],
goemotions far +0.230 [.170,.295], goemotions near +0.225 [.155,.295].

**Two conclusions.** The order effect survives at about ten points rather than
fourteen, so C1 still fails for Jev by a wide margin. And Jev is not
reproducible: identical input five times changes the answer on 4.3% of
decisions, where Laya is exactly 0.0000 over 16,000 calls. The C1 framing
assumed order was the only thing a repeat call could vary. It was not.

The effects overlap and do not decompose: of 248 items flipping under either
arm, 179 flipped only when shuffled, 15 only under a fixed order, 54 under
both. The fixed rate is a floor on the noise, not a term to subtract.

Laya's exact zero doubles as a second harness check. Whatever produces Jev's
4.3%, it is not the measurement.

**Where flips concentrate.** Near tier > far tier everywhere, and the rate grows
with K. Laya at K=64: clinc/far 0.090 -> mtop/near 0.745. The cheapest cell for
Laya (clinc/far, its trained family against unrelated distractors) is 0.000 at
K=16 — which is exactly the cell an earlier pilot sampled, and why that pilot
reported "order flips nearly vanish". Across the full suite that reading was
wrong by an order of magnitude.

**Interpretation.** Presentation order is not a second-order nuisance for
option-conditioned decision models; at realistic candidate counts it is a
primary determinant of the answer. The same joint attention that lets these
models compare candidates makes position part of the input. Combined with the
RQ3 result (A3 also degrades fastest under near distractors), the mechanism
that defines this architecture family is carrying two measurable costs and, so
far, no measured benefit.

**Caveats.** Flip rate counts argmax changes, not correctness; a model can flip
between two defensible options. Mitigations exist and are untested here:
order-shuffling augmentation during training, and test-time permutation
ensembling, which restores exact invariance at M x latency.


### RQ1 / C3 — cardinality x text length — 2026-09-24, n=800 (clinc, mtop, goemotions, dbpedia)

Dataset v3, 6 models x 4 domains x K in {2..256} on the `ext` pool, 24 cells,
38,400 decisions, ~4.5h.

**Accuracy vs K** (slope = accuracy change per doubling of K):

| model | fam | K=2 | K=16 | K=64 | K=256 | slope |
|---|---|---|---|---|---|---|
| *chance (1/K)* | | *0.500* | *0.062* | *0.016* | *0.004* | |
| laya | A3 | **0.866** | 0.691 | 0.521 | **0.276** | **-0.0807** |
| deberta-v3-large-zeroshot | A1 | 0.856 | 0.691 | 0.521 | 0.343 | -0.0731 |
| deberta-v3-base-zeroshot | A1 | 0.829 | 0.639 | 0.469 | 0.328 | -0.0706 |
| gliclass-large-v3.0 | A3 | 0.802 | 0.625 | 0.476 | 0.328 | -0.0658 |
| bge-large-en-v1.5 | A2 | 0.681 | 0.561 | 0.435 | 0.310 | -0.0527 |
| gte-large | A2 | 0.670 | 0.527 | 0.425 | **0.320** | **-0.0482** |

**C3 verdict — families separate by SLOPE, not by level.** Bootstrap CIs on
pairwise slope differences: A2 vs A1 **-0.0214 [-0.0259,-0.0164] SEPARATED**,
A2 vs A3 **+0.0228 [+0.0181,+0.0280] SEPARATED**, A1 vs A3 +0.0014
[-0.0029,+0.0065] **overlaps 0**. Embedding scorers degrade measurably more
slowly than either family that reads the options jointly with the text;
cross-encoders and option-conditioned models are indistinguishable from each
other in slope.

The ranking inverts across the range. Laya is the best model at K=2 (0.866)
and the WORST at K=256 (0.276); gte-large is the worst at K=2 (0.670) and
finishes ahead of it (0.320). Any single-K benchmark of this category reports
a different winner depending on the K it happened to pick.

**Truncation: zero on every call, every model, every K.** So the curves are
cardinality effects, not context-overflow artifacts — the distinction that a
harness bug previously erased. Laya's K=256 collapse is real model behaviour.

**Second factor — text length is weak.** Within DBpedia's 10x internal spread
(128 -> 1300 chars), accuracy is close to flat across quartiles: deberta-large
at K=64 runs 0.623 / 0.680 / 0.558 / 0.644 from shortest to longest quartile,
with no monotone penalty for any model. What interaction exists is small and
concentrated at K=256. **Candidate-set size dominates text length by a wide
margin in this range** — worth stating because the opposite is often assumed.

**Leakage stratification.** CLINC's 25% verbatim-gold items score +0.13 to
+0.39 above clean items (gliclass at K=256: 0.900 leaked vs 0.507 clean). Any
CLINC figure quoted without this split is inflated.

---

**Deviation D6 (2026-09-24, post-inference): baselines and metrics.** The
pre-registration asks for "chance and majority baselines on every table" and for
Brier and 15-bin ECE beside accuracy. Two corrections.

The majority baseline does not transfer to this design. It was written for the
earlier fixed-label-set setup, where one class could dominate a column. Here
every item carries its own K-option set, the gold appears exactly once (G8), and
gold position is uniform (G4), so the only no-information floor is chance = 1/K.
That row is now on the accuracy table above. Nothing is lost; the statistic
simply had no referent in v3.

Brier and ECE were genuinely missing and are now reported below, recomputed from
the existing per-item logs without re-running any model.

### Calibration vs K — 2026-09-24, derived from the RQ1 logs (no re-runs)

The pre-registration asks for Brier and 15-bin ECE beside accuracy, and the
first write-up shipped accuracy alone. Both are recomputable from the per-item
logs, which carry full-precision distributions, so no model was re-run.
`results/published/rq1_calibration.json`; same 800 items, pooled over the four
RQ1 domains.

**15-bin ECE** (top-label confidence; lower is better):

| model | K=2 | K=4 | K=8 | K=16 | K=32 | K=64 | K=128 | K=256 |
|---|---|---|---|---|---|---|---|---|
| deberta-v3-large-zeroshot | 0.064 | 0.076 | 0.065 | 0.045 | 0.054 | 0.049 | 0.031 | 0.070 |
| deberta-v3-base-zeroshot | 0.066 | 0.080 | 0.070 | 0.052 | 0.041 | 0.046 | 0.044 | 0.075 |
| bge-large-en-v1.5 | 0.099 | 0.035 | 0.084 | 0.152 | 0.192 | 0.201 | 0.188 | 0.197 |
| gte-large | 0.048 | 0.135 | 0.250 | 0.332 | 0.357 | 0.356 | 0.315 | 0.299 |
| gliclass-large-v3.0 | 0.150 | 0.220 | 0.207 | 0.238 | 0.265 | 0.292 | 0.307 | 0.327 |
| laya | **0.032** | 0.071 | 0.174 | 0.262 | 0.305 | 0.407 | 0.514 | **0.573** |

**Multiclass Brier**: deberta-large 0.222 -> 0.810 across K=2 -> 256,
deberta-base 0.266 -> 0.833, bge 0.395 -> 0.876, gliclass 0.338 -> 0.949, gte
0.390 -> 0.973, laya 0.201 -> **1.256**.

**A fourth axis where the families separate, and it is not the same split.**
The pair cross-encoders are flat: deberta-large moves 0.064 -> 0.070 over a
128-fold increase in candidates, so its confidence stays interpretable
everywhere. Everything else degrades, and Laya degrades most, going from the
best-calibrated model in the set at K=2 to the worst by a factor of eight at
K=256. Note this is not the option-conditioned split: gte-large is an embedding
scorer and it degrades nearly as badly (0.048 -> 0.299), while gliclass, which
reads options jointly like Laya, sits between them. Architecture predicts order
sensitivity cleanly; it does not predict calibration drift.

Operationally this is the number that decides whether a confidence gate works.
A Laya deployment tuned on a short candidate list and then pointed at a long one
keeps reporting the same confidences while being wrong far more often.

Jev is absent: its API rounds probabilities to two decimals and 98% of responses
contain at least one option at exactly 0.00, which is too coarse to bin.

## Status 2026-09-24: all three questions answered

| | Result |
|---|---|
| **RQ1 / C3** | Families separate by slope. A2 degrades slowest (-0.048/-0.053); A1 and A3 indistinguishable (-0.066 to -0.081). Rank order inverts between K=2 and K=256. Text length is a weak second factor. |
| **RQ2 / C1** | FAILS for option-conditioned models: 16-49% of answers change from option order alone. A1/A2 structurally invariant (verified: exact 0.0000, with ties the only crack). Amended 2026-09-25: Jev's 14.6% is 10.2 points order and a 4.3% floor of the model not repeating itself; Laya's 49.4% is all order. |
| **RQ3 / C2** | Applicability gate fired at K=16/32 (within-family > between-family). H3 NOT supported; at K=64 A3 carries the LARGEST near-distractor penalty (+0.351 vs +0.256 A1). |

**Converging conclusion.** Across three independent questions, the
option-conditioned architecture shows no measured advantage and two measured
costs: it degrades fastest under near distractors and it is the only family
whose answer depends on the order the options happen to be listed in. Its
genuine advantage is elsewhere and nothing here contests it: single-pass
latency that is flat in K, where cross-encoders pay one forward pass per option.

---

## Jev arm — failure-mode verification before spending (2026-09-24)

Jev is reachable via OpenRouter at the undocumented `/api/v1/systemone`
endpoint (it is absent from the 458-model catalog; the chat endpoint rejects it
with a message naming the right one). Resolves to `typesafe/jev-1.13-20260917`.

**Measured cost scaling** (real calls on real dataset items): input tokens =
**284 + 13·K**, i.e. 284 tokens of FIXED overhead per call, at exactly
$0.042/M with no OpenRouter markup. That makes the full arm **$0.96 for 29,600
calls** — 4.5x my earlier token-only estimate, which ignored the overhead.
Budget is $5.00, so the arm uses 19%.

**Drills run against the live API (total spend $0.00006):**

| # | Failure | Result |
|---|---|---|
| 1 | SIGKILL mid-run | ledger survived (fsync per call); resume re-paid **0 of 2** completed calls |
| 2 | Spend cap | initially FAILED — only blocked once already over. Fixed to a pre-flight projection; now blocks the *crossing* call with no leak |
| 3 | Option set > 255 (API cap) | recorded as an error, $0 spent, run continues |
| 4 | Torn final ledger line | replay tolerates truncated JSON, recovers all valid records |
| 5 | Permanent 4xx | recorded in 0.1s, no retry storm |

**Two further gaps found by review, not by drill:**
- A transient failure (network blip exhausting retries) was being cached as
  "done" forever, so resume would never retry it. Now only records that
  produced probabilities, or that COST money, are settled; free failures are
  retried, since re-sending them cannot double-spend.
- Credit exhaustion (401/402/403) would have been recorded 29,600 times.
  It now aborts the run immediately, resumably.

**Standing protections for the run:** per-call append-only ledger with fsync,
free resume, pre-flight spend cap, bounded exponential backoff on 408/429/5xx
only, and response validation (the returned choice must be one of the options
sent and the probability keys must match exactly, or the call is recorded as a
failure rather than silently scored).

Confirmed as predicted: Jev returns probabilities at **2 decimals**, so its
NLL/Brier/ECE will be rounding-limited. Accuracy and flip rate — which carry
all three findings here — are unaffected.


### Jev arm — 2026-09-24 — 29,600 paid calls, $1.04, zero failures

Via OpenRouter `/api/v1/systemone`, model `typesafe/jev-1.13-20260917`, against
the identical frozen dataset and protocol.

**Deviation D4 (2026-09-24, pre-inference): Jev K range.** The pre-registration
caps the Jev arm at K <= 64. It ran to K=128 on RQ1 instead, because the API
accepts up to 255 options and the extra cell is free at this cost. Permissive
direction, recorded here rather than left silent. RQ2 and RQ3 ran at their
pre-registered K values.

**Deviation D5 (2026-09-24, post-inference): slope comparison range.** The first
write-up compared Jev's K=2..128 slope against open-model slopes fitted on
K=2..256. Curves steepen at the top, so that comparison flatters whichever model
stopped earlier. All slopes are now refitted on the shared K=2..128 range in
`analyze_jev.py`; the ordering is unchanged and Jev is still shallowest
(-0.0425 against gte-large -0.0482).

| | Jev | best open | worst open |
|---|---|---|---|
| **RQ1** K-slope (K=2..128, all models) | **-0.0425** | gte -0.0482 | laya -0.0731 |
| **RQ1** acc @K=128 | **0.600** | deberta-large 0.410 | gte 0.354 |
| **RQ2** flip @K=16 | **0.070** | A1/A2 0.0000 | laya 0.206 |
| **RQ2** flip @K=64 | **0.146** | A1/A2 0.0000 | laya 0.494 |
| **RQ3** Delta @K=64 | **+0.105** | A1 +0.256 | A3 +0.351 |

**This inverts the conclusion about the architecture.** Jev shows the
option-conditioned SIGNATURE — it is order-sensitive, which the pair and
embedding families structurally cannot be — but at roughly a third of Laya's
magnitude, while simultaneously being the MOST robust model tested to near
distractors (+0.105 against +0.256 to +0.351 for every open family) and having
the FLATTEST K-curve of anything measured, embedding scorers included.

Revised reading: the two costs found in the open models are **properties of
those checkpoints, not of the architecture**. Option-conditioning does not
require a 16-49% order tax or the steepest near-distractor decay; Jev
demonstrates an implementation with neither at that magnitude. Our RQ2/RQ3
conclusions must be restated as being about the open A3 implementations, not
about A3 as a class.

**The confound that prevents a stronger claim.** Jev's training data is
undisclosed, so under our own rung ladder it is UNGRADABLE — and CLINC, MTOP
and GoEmotions are all public datasets it may have trained on. Its advantage is
therefore consistent with either better training or contamination, and this
experiment cannot separate them. A weak signal favouring familiarity: Jev's flip
rate at K=64 is lowest on the intent domains and highest on GoEmotions, the
domain least likely to appear in a decision-routing training mix. Both tiers,
because the far tier alone understates every cell by a factor of three to six:

| domain | far | near |
|---|---|---|
| clinc | 0.015 | 0.090 |
| mtop | 0.030 | 0.125 |
| fintopic | 0.050 | 0.220 |
| goemotions | 0.305 | 0.330 |

The near column is the one a deployment sees, since real candidate sets contain
plausible competitors. An earlier draft of this section and of the README quoted
only the far column and reported Jev's intent-domain flip rate as "1.5% to 3%";
the honest range across both tiers is 1.5% to 12.5%.

**Rounding, as predicted.** 28,915 of 29,600 calls (98%) return at least one
option at exactly 0.00, so Jev's NLL/Brier/ECE are rounding-limited and are not
reported. Accuracy and flip rate are unaffected and carry the findings above.

**Operational note.** Every failure protection held: per-call fsync'd ledger,
free resume (the 72 smoke calls were correctly skipped as already paid), a
pre-flight spend cap, and response validation. 23 calls/sec at 5 workers,
238ms mean latency, and not one call had to be retried or re-paid.

---

## Appendix — deviations from the superseded stage-2 plan

These are deviations
from an earlier plan, against a dataset and pipeline (`universe_v2`,
`stage2.py`) that the 2026-09-23 audit discarded. Current deviations are D4
onward, above.

**D1 (pre-inference): near-tier K feasibility.** The near tier was bounded by
source size (CLINC ~149 sibling options, GoEmotions ~27, fin-topic ~19), so the
tier contrast ran at K in {16,32,64} on CLINC and K=16 only elsewhere. Cells
whose per-item pool was smaller than K-1 were skipped and logged as infeasible,
never filled from another source. Superseded: v3 builds tiers from measured
similarity rather than source membership, and every item is feasible in both
tiers at every K (gate G6).

**D2 (pre-inference): deberta-large subsampling.** Full N at K<=64, a fixed
150-item prefix above. Carried forward into v3 unchanged.

**D3 (post-inference, data discarded and rerun): prompt-option collision.**
The one worth keeping. The stage-2 question ("What is the intent or category of
this text?") and template ("The intent or category of this text is {}.") both
contained the word *text*, and CLINC-150 ships an intent named exactly `text`
(send a text message), which was in the universe. At K=64 on the CLINC near
tier, with `text` offered in 93 of 200 items:

| model | errors that were `text` |
|---|---|
| laya | 50 of 53 (94%) |
| gliclass-large-v3.0 | 0 of 29 (0%) |
| bge-large-en-v1.5 | 0 of 22 (0%) |

Laya's near-tier collapse in that run was mostly this artifact rather than
semantic difficulty. Thirty-one completed cells were discarded, the prompt was
reworded to "Which label applies here?" / "This example is labeled {}.", and a
collision check was added that refuses to start if any option equals a prompt
content word. It is now gate G1.

At the time I read the asymmetry as a robustness difference between
architectures. The audit withdrew that (finding S2): `GLiClassZS.decide` accepts
a template and a question and uses neither, passing bare option strings, so
GLiClass was immune because it never saw the colliding word. The embedding
adapter applies the template to the options rather than to the input, so its
exposure was different too. The asymmetry was in my adapters, not in the models.

What survives is narrower and still worth knowing: a single word shared between
your prompt and one of your candidate labels can account for 94% of one model's
errors, and nothing in a normal evaluation would show you that. It is now gate
G1, which refuses to start a run if any option equals a prompt content word.
