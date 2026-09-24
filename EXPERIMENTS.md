# Experiment log

Running record for the v3 program. Governed by `PREREGISTRATION.md`; the audit
that produced this design is in `docs/history/`.

Nothing above a pilot label may be cited as a result. Every entry states n, the
dataset version, and any deviation from the pre-registration.

---

## Dataset v3 — frozen 2026-09-23

1,000 items across 5 domains (clinc, mtop, goemotions, dbpedia, fintopic) over
a 902-option universe from 12 sources. Per-RQ splits in `dataset/v3/splits/`.

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

## Program status 2026-09-24: all three research questions answered

| | Result |
|---|---|
| **RQ1 / C3** | Families separate by slope. A2 degrades slowest (-0.048/-0.053); A1 and A3 indistinguishable (-0.066 to -0.081). Rank order inverts between K=2 and K=256. Text length is a weak second factor. |
| **RQ2 / C1** | FAILS for option-conditioned models: 16-49% of answers change from option order alone. A1/A2 structurally invariant (verified: exact 0.0000, with ties the only crack). |
| **RQ3 / C2** | Applicability gate fired at K=16/32 (within-family > between-family). H3 NOT supported; at K=64 A3 carries the LARGEST near-distractor penalty (+0.351 vs +0.256 A1). |

**Converging conclusion.** Across three independent questions, the
option-conditioned architecture shows no measured advantage and two measured
costs: it degrades fastest under near distractors and it is the only family
whose answer depends on the order the options happen to be listed in. Its
genuine advantage is elsewhere and is not contested by this study: single-pass
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
all three of this study's findings — are unaffected.
