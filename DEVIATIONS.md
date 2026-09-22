# Deviations log — stage 2

Entries recorded BEFORE the affected inference unless marked otherwise.

## D1 (2026-09-22, pre-inference): NEAR-tier K feasibility

The NEAR tier (same-source siblings minus conflicts) is bounded by source
size: CLINC ~149 sibling options, GoEmotions ~27, fin-topic ~19. Therefore:
- C2 is evaluated at K in {16, 32, 64} on CLINC (the ">=2 of 3 K values"
  rule applies there), and at K=16 only on GoEmotions and fin-topic
  (cross-domain support checks).
- NEAR cells whose per-item pool is smaller than K-1 are skipped and logged
  as infeasible, never silently filled from other sources.
This was discovered during runner implementation, before any stage-2
inference was run.

## D2 (2026-09-22, pre-inference): deberta-large subsampling

As pre-registered: deberta-v3-large runs the full N at K<=64 and a fixed
150-item prefix (of the seeded item order) at K in {128, 256}.
