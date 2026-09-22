# Experiment log — decision-model research

Running record of every experimental wave. Governing brief (framing, ladder,
experimental program): https://claude.ai/code/artifact/030ca553-fc0a-4200-863d-70ac0103862b
Companion results page (charts): see "Lab Log" artifact linked from each wave's PR-of-record.

**Program question:** can a ~400M encoder acquire a general-purpose decision
capability (state + question + candidate options -> probability distribution)
that transfers to unseen tasks, candidate sets, and schemas with little or no
task-specific supervision?

---

## Wave 1 — zero-shot baselines (2026-09-21)

**Setup.** 5 models x 6 datasets, fixed seeded subsets (seed 42; n=500, or 300
for CLINC-150 and Banking77). Metrics: accuracy, macro-F1, Brier, NLL, 15-bin
ECE; full per-decision probability distributions logged to JSONL. Latency: true
bs=1 for single-pass models (GLiClass, Laya); batched accuracy runs for NLI
cross-encoders plus a separate 25-example bs=1 probe at 2/20/77 labels.
Results: `baselines/wave1/`.

**Accuracy (suite):**

| model | enron(2) | dd-emotion(7) | fin-topic(20) | sst5(5) | clinc(151) | b77(77) | mean |
|---|---|---|---|---|---|---|---|
| laya | .982 | .642 | .430 | .468 | .560* | .473* | .593 |
| deberta-large-zs | .834 | .772 | .350 | .450 | .527 | .507 | .573 |
| gliclass-large | .748 | .568 | .314 | .502 | .523 | .493 | .525 |
| deberta-base-zs | .778 | .520 | .438 | .450 | .457 | .497 | .523 |
| bart-large-mnli | .586 | .112 | .454 | .466 | .397 | .323 | .390 |

*Laya at extended context (2048/1024), vendor-flagged unvalidated.

**bs=1 latency (MPS):** pass-per-label collapse measured: deberta-large
140ms@2 -> 719@20 -> 2,214@77 labels; Laya 32-68ms at <=20 labels; GLiClass
135 -> 581ms (1 -> 151 labels; not flat at bs=1, contra the paper's A6000
throughput claim).

**Findings.** (1) Laya's suite lead is entirely its trained spam family
(Enron .982; off-family it majority-collapses, DailyDialog F1 .270).
(2) deberta-large-zs is the strongest honest zero-shot baseline. (3) GLiClass
is pathologically overconfident raw (probs ~1.0, NLL 7-14, ECE ~= 1-acc).
(4) Ordinal (SST-5) is uniformly weak, .45-.50 for all. (5) bart-large-mnli
(the HF pipeline default) is weak everywhere.

---

## Wave 2 — rungs, calibration, A2, K-sweep (2026-09-22)

### Generalization rungs (`results/wave2_rungs.json`)

Family-level strictness (brief Section 5) applied to wave-1 zero-shot rows:

- deberta-zeroshot (base & large): **zero defensible G4 rows** — trained on
  spam, emotion, topic, sentiment, and intent families, plus Banking77 itself
  (that row is G0).
- bart-large-mnli: only full-disclosure G4 model on all six rows; mean acc .390.
- GLiClass: five G4* rows (.529), Laya: three G4* rows (.513) — partial
  disclosure caps both at "G4-with-asterisk".

**Consequence:** no public checkpoint demonstrates strong, honestly-G4
zero-shot decision capability on this suite. And a breadth mixture will
blanket the same families, so our own G4 measurement requires
leave-one-family-out mixtures (brief ablation I).

### Temperature scaling (`results/calibration_summary.json`)

Single temperature per (model, dataset), fitted on log-probs from the existing
JSONL (no re-runs), 50/50 cal/test split. Mean ECE across the suite, pre -> post:

| model | pre | post | fitted T range |
|---|---|---|---|
| gliclass-large | .476 | **.046** | 5.3 - 27.0 |
| laya | .251 | .100 | 0.5 - 4.9 |
| bge-large (A2) | .229 | .066 | 0.4 - 71 |
| bart-large-mnli | .208 | .081 | 0.6 - 71 |
| deberta-base-zs | .171 | .109 | 0.6 - 2.5 |
| deberta-large-zs | .154 | .077 | 0.8 - 3.0 |

The <=0.10 zero-shot engineering target is broadly reachable with cheap
post-hoc temperature alone — the bar RLCD-style training must beat (ablations
F/G). Degenerate case worth remembering: bart on DailyDialog reaches ECE .018
at 11% accuracy via T=71 (calibrated uniform guessing) — never report ECE
without accuracy.

### A2: embedding + label similarity (bge-large-en-v1.5)

Suite mean .447 (weak overall; DailyDialog .136), but **best of all models on
Banking77 (.643** vs deberta-large .507) at flat 26-40ms regardless of label
count. The BTZSC claim (embeddings own fine-grained intent) reproduces.

### K-sweep pilot (`results/ksweep_pilot.json`)

n=50 CLINC gold items, nested distractors from a 288-option universe built
from six real label sets; K in {2..256}; same items at every K. Pilot = curve
shape, not headline numbers.

| K | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | p50 @256 |
|---|---|---|---|---|---|---|---|---|---|
| laya (option-cond.) | .98 | .98 | 1.00 | .94 | .90 | .80 | .64 | **.24** | 277ms |
| gliclass (option-cond.) | 1.00 | .98 | .88 | .90 | .82 | .88 | .78 | .56 | 1,096ms |
| bge (embedding) | .96 | .92 | .86 | .82 | .80 | .74 | .64 | **.62** | 703ms |
| deberta-base (NLI) | 1.00 | .96 | .94 | .90 | .80 | .74 | .60 | .50 | 2,829ms |

**Design audit (2026-09-22, on review):** setup is sound as a pilot (nested
distractors, forced choice, paired items), with three recorded flaws.
(1) Universe merges CLINC and Banking77 intents, so large-K cells can draw
near-synonym distractors ("transfer" gold vs "transfer timing"); absolute
accuracy at K>=128 is pessimistic for all models equally — fix via universe
dedup or near/far distractor tiers in the extensive run. (2) Default option
order used salted hash() — irreproducible across processes; fixed to CRC32
(orders in the pilot were consistent within-run). (3) bge latency is
cold-cache (re-embeds options per item); real serving caches option vectors,
so its true K=256 latency is ~30ms, not 703ms. n=50 resolves curve shape only:
acc gaps <~11 pts and the 8%-vs-4% flip contrast are NOT separable; extensive
run = 300-500 items x 3-4 gold domains, deduped universe, 5 order seeds +
3 phrasings at selected K, bootstrap CIs + paired tests, NLI capped at K<=64
on the full grid (subsampled above). ~1-2 days background compute.

Architectures trade places along K: option-conditioned dominates small-to-mid
K (Laya perfect at K=8), GLiClass holds best through K=64-128, embeddings win
at K=256. **Laya's context ceiling is a measured cliff** (.64@128 -> .24@256,
NLL 15). Caveat: gold items are CLINC intents = a trained family for Laya, so
its small-K brilliance is G3-grade.

### Order sensitivity (flip test @ K=16, 5 permutations)

| model | flip rate | why |
|---|---|---|
| gliclass | **8%** | option-option attention: options see each other |
| laya | **4%** | same mechanism, milder |
| bge | 0% | order-invariant by construction (per-option scoring) |
| deberta-NLI | 0% | order-invariant by construction (per-option scoring) |

This is an architectural property, not noise: pair and embedding scorers
process each option independently, so option order cannot exist for them
(the measured 0.00 doubles as a harness sanity check). Option-conditioned
models buy joint state-question-option attention at the cost of sensitivity
to presentation order — the same axis where Jev flips 13% and the worst
constrained LLM 37% (nibzard). Mitigations to test: order-shuffling
augmentation (already in the mixture plan) and test-time permutation
ensembling (average probabilities over M orders — restores invariance at
M x cost). Worst-case reporting = min over permutations, per protocol.

---

## Caveats (both waves)

- Seeded subsets (n=500/300), not full test sets; dataset revisions not yet pinned.
- K-sweep is n=50, CLINC-gold-only, universe capped at 288 options (K=512/1024 pending a larger universe).
- ECE without debiased estimator so far; single temperature per (model, dataset).
- Laya >20-option runs use vendor-unvalidated extended context.
- All numbers are zero-shot checkpoints as shipped; nothing here is our own trained model yet.

## Wave 2.5 — distractor-cleanup sensitivity (2026-09-22)

Phase A/B of the distractor plan executed: universe_v1 frozen (410 canonical
options from 11 sources, 3 merges, 52 cross-source conflict pairs; same-source
pairs exempt — source datasets pre-adjudicate their own labels; nomination =
mpnet cosine >= 0.60 OR token containment, cross-source only; default ruling
CONFLICT). Per-item Phase-B exclusion: each text's top-10 nearest options
(mpnet) removed from its distractor pool (median 9 exclusions/item).
K-sweep pilot rerun on the clean universe, same 50 items:

| K | 64 | 128 | 256 | | flip@K16 |
|---|---|---|---|---|---|
| laya      | .80 -> .92 | .64 -> .72 | **.24 -> .52** | | .04 -> .00 |
| gliclass  | .88 -> .82 | .78 -> .74 | .56 -> .68 | | .08 -> .02 |
| bge       | .74 -> .82 | .64 -> .80 | .62 -> .76 | | .00 -> .00 |
| deberta-b | .74 -> .66 | .60 -> .56 | .50 -> .56 | | .00 -> .00 |

**Why the 1.0 cells are unremarkable** (measured 2026-09-22): far-tier
small-K items are easy by construction — median nearest-distractor-to-gold
cosine is 0.109 at K=2 / 0.201 at K=8 (vs 0.403 at K=256), and per-item
exclusion removes each text's 10 nearest options; 1.0 on n=50 = ">=0.94 at 95%
confidence" (rule of three; a true-0.97 model goes 50/50 ~22% of the time);
and CLINC is Laya's trained family. In this far-tier design the informative
region starts ~K=64; small-K difficulty only exists in the near tier, whose
small-K cells must NOT be 1.0 (else tier construction failed).

**Findings.** (1) Over half of Laya's K=256 "cliff" was distractor ambiguity
(+.28); a real context-ceiling degradation remains (.92@64 -> .52@256).
(2) **Order flips collapsed after cleanup** (gliclass 8% -> 2%, laya 4% -> 0%):
the pilot's order sensitivity was concentrated on ambiguous, defensible-
either-way candidates. The architectural flip tax is smaller than wave 2
suggested — stage-2 N required before claiming it is near-zero.
(3) Honest bias note: per-item exclusion uses mpnet, which correlates with
bge, so part of bge's gain (+.14-.16 at large K) is filter-model bias; the
extensive run must use two independent filter models or report both filtered
and unfiltered curves. (4) v1 -> v2 deltas also include distractor re-sampling
noise (pools changed, so draws changed); at n=50 only the large moves are
trustworthy. Artifacts: results/universe_v1.json, ksweep_pilot_v2clean.json. Full dataset-engineering write-up: Lab Log artifact section 2 (collection, normalization, dual nomination channels, same-source exemption, adjudication, freeze, per-item exclusion, known biases).

## External review #1 (2026-09-22) — ACCEPTED, fixes queued

Full re-derivation review of artifact + repo. Figures matched the data; four
claims did not survive, all verified against the repo before acceptance
(GLiClass support sparsity re-measured: median/max 1 non-zero option on all
six datasets, gold@0 up to 69%; T-grid ceiling confirmed 70.98; ECE == 1-acc
identity exact on all GLiClass rows; duplicate enron row confirmed in
calibration_summary.json).

**Retracted / under revision:**
1. All GLiClass and Laya calibration numbers (adapters discard the models'
   distributions; GLiClass reaches the harness one-hot -> its .476->.046 is
   the arithmetic floor of a constant-confidence predictor with zero
   resolution; Laya's returned support is sparse, gold@0 up to 22% in-suite).
2. "<=0.10 reachable with temperature alone": per-(model,dataset) fits on
   labeled halves are not zero-shot, fitted T spans 0.4-71 within one model
   (two rows railed at the search ceiling), and TS worsened 6/37
   already-calibrated rows. Revised finding: NO transferable temperature
   exists — raises the bar; ablations F/G now target schema-transferable
   calibration directly.
3. Wave-2.5 ambiguity attribution: confounded with universe dilution
   (288->410 options cuts expected near-domain distractors ~31% at K=256
   before any filter). Decomposition rerun required before any attribution.
4. Flip contrast: 1 item vs 4 items, Fisher p~=0.36. Direction only.

**Also corrected:** conflict matrix nearly inert (text top-10 filter supplies
median 9/9 exclusions; its ambiguity precision unaudited); nomination script
was never committed and universe.py's FLAG_T=0.80 disagrees with the used
0.60+containment; "human adjudication" was model-proposed default-to-conflict
pending spot-check; 8 loaded sources (not 9/11); DailyDialog majority class
0.812 beats all six models (6/36 wave-1 cells below a constant predictor;
lift-ordering happens to preserve rankings); G4 means averaged disjoint row
sets (intersection on dd+fin reverses to laya .536 > gliclass .441 > bart
.283); Laya's K=256 NLL is support-miss counting, not confidence evidence
(harness cannot yet distinguish "ranked gold low" from "gold never reached
the model"); ksweep logs aggregates only; three cosine figures were
session-computed with no committed script; stale smoke-run rows pollute
calibration_summary.json and wave2_rungs.json.

**Fix queue (review's order):**
1. Raw-score extraction in GLiClass + Laya adapters -> rerun calibration.
2. Chance + majority baseline columns everywhere; DailyDialog as lift or drop.
3. Wave-2.5 decomposition: clean filters on the ORIGINAL 288 universe.
4. Per-item JSONL + gold-in-support logging in ksweep (enables bootstrap CIs).
5. Commit nomination script; align FLAG_T; fix adjudication language (done in
   artifact).
6. Raise/flag T-grid ceiling; report Brier/resolution beside ECE.
7. G4 intersection comparison replaces disjoint means (done in artifact).
Plus: purge stale smoke-run artifacts from results/ summaries.

## Wave 3 — raw-distribution reruns (2026-09-22, post-review fixes 1/2/4/6)

GLiClass adapter now inverts the multi-label sigmoid path (exact softmax
reconstruction from the same logits); Laya adapter runs an unrounded direct
forward through laya.common (predict()'s 4-decimal rounding was the entire
"sparse support" — gold ALWAYS reaches the model below the ctx ceiling).
Accuracies unchanged (argmax invariant, consistency check passed).

**Honest calibration (suite-mean ECE pre -> post, fitted T range):**
- gliclass: .262 -> .118 (T 1.7-9.8) — genuinely overconfident, NOT degenerate;
  the retracted .476 -> .046 was one-hot arithmetic.
- laya (unrounded): .251 -> .091 (T 0.5-4.2).
- No-transferable-temperature finding STANDS (within-model T spreads remain
  large; bge's degenerate DailyDialog row persists).

**K-sweep v3 (per-item JSONL + gold-in-support now logged):**
- Accuracy curves unchanged from wave 2.5 (same argmax).
- Laya top-5@256: .62 -> .86 unrounded (rounding had destroyed tie-breaking);
  NLL@256 8.53 honest. The K collapse is top-rank discrimination, not total
  ranking failure.
- LATENCY CORRECTION: Laya direct-forward p50 = 8-21ms flat across K=2-256
  (includes tokenize+collate+forward). Prior 27-277ms figures included
  agent.predict() wrapper overhead. Strongest single-pass latency evidence
  in the study; wrapper-vs-model latency now a disclosed distinction.
- GLiClass top-5@256 .72; flip@K16 still 2% (1 item); laya/bge/deberta 0%.

**Reference baselines (fix 2, results/reference_baselines.json):** majority =
enron .502, dd-emotion .812 (beats all models), fin-topic .232, sst5 .300,
clinc .180 ('out of scope' is the modal label — all models beat it),
banking77 .027. All accuracy tables now report these; dd as lift.

## Wave 3.5 — decomposition of the wave-2.5 delta (2026-09-22, review fix 3)

Full 2x2: universe {288 original, 410 grown} x filters {off, on}, same 50
items, all four models, accuracy at K in {32,64,128,256}. Arms: A=(288,off)
wave-2, B=(288,on) NEW, C=(410,on) wave-3, D=(410,off) NEW.
filter = mean(B-A, C-D); dilution = mean(D-A, C-B). results/decomposition.json.

**Verdict: the review's dilution hypothesis confirmed.** For Laya at K=256 the
+0.28 recovery splits ~+0.16 dilution / ~+0.12 filtering; at K=128 dilution
dominates outright (+0.11 vs -0.03). The retracted "over half was ambiguity"
stays retracted; correct statement: universe growth (fewer near-domain
distractors per draw) was the larger component, ambiguity removal real but
secondary. Laya's clean-universe cliff itself stands (B arm: .40@256 on the
288 universe WITH filters, vs .90@64).

**Filter-model bias validated:** bge shows the largest filter effects
(+0.10 at K=128/256), exactly as predicted from mpnet-filter correlation —
even bge's filter effect is partly self-serving. Two independent filters
mandatory for stage 2 (already in the design).

**Early RQ3 signal (directional, n=50):** Laya is the MOST near-density-
sensitive model (dilution effect +0.11/+0.16 at 128/256 vs <=+0.06 for all
others) — a counter-signal to hypothesis H3 (option-option attention helps
with near distractors), though confounded with its context ceiling at these
K. Stage 2's near/far tiers at K<=64 (below the ceiling) will separate them.

Effects at other cells are within noise (se ~0.07); only Laya@256 and bge's
filter column stand above it. deberta shows ~no effects, as expected for a
pass-per-label scorer.

## Two-stage N protocol (adopted 2026-09-22)

Every experiment runs pilot-first: **Stage 1 at N=50** (hypothesis-generating,
always labeled "pilot", never a headline claim), then **Stage 2 confirmatory**
only for contrasts the pilot flags as interesting. Stage-2 rules, binding:

1. Fresh seeded items (pilot items may be included; analysis is on the full
   fresh draw) — never re-measure the same 50 and call it confirmation.
2. Contrasts + decision rules pre-registered here BEFORE the run; everything
   else in the output is exploratory.
3. N from power, not vibes: flip-rate 8%-vs-4% needs ~550 items; a 5-point
   accuracy gap near 0.7 resolves at ~500 items/domain under the paired
   design. Hence the extensive-run spec of 300-500 items x 3-4 domains.
4. Optional ladder 50 -> 200 -> 500 with a pre-stated stopping rule (drop a
   contrast when its CI excludes the interesting effect) to save the
   expensive NLI arm.

## Next

1. SetFit few-shot curves (baseline matrix few-shot cells, K_labels in {10,25,50,100}/option).
2. typed-decisions + PhishNChips wiring (product-shape + guardrail slots, published-number tie-ins).
3. LOFO mixture design -> first A3 training run (ablations B, C first).
4. Larger option universe for K=512/1024; bootstrap CIs; debiased ECE.
5. **Jev arm of the K-sweep + flip test** (queued; needs TypeSafe API access):
   same 50 nested-distractor items, K in {2..128, 255} (API caps choices at 255,
   so the ceiling is API-enforced, not measurable). Purpose: architecture
   fingerprinting — Jev's training/architecture are undisclosed, and the K-curve
   shape + flip rate discriminate option-conditioned encoder vs pair scorer vs
   autoregressive LLM signatures. Priors from external evals: 13% order-flip
   (nibzard), faster than GLiNER at 72 labels (AbdelStark). Hosted latency
   reported separately; rung = ungradable. Cost < $1 in API calls.
