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
