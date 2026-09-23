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

*(none yet — the first confirmatory run goes here)*
