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

## Next

1. SetFit few-shot curves (baseline matrix few-shot cells, K_labels in {10,25,50,100}/option).
2. typed-decisions + PhishNChips wiring (product-shape + guardrail slots, published-number tie-ins).
3. LOFO mixture design -> first A3 training run (ablations B, C first).
4. Larger option universe for K=512/1024; bootstrap CIs; debiased ECE.
