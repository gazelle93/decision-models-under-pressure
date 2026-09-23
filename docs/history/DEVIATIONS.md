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

## D3 (2026-09-23, POST-inference — data discarded and rerun): prompt-option collision

Discovered while printing real examples for review: the stage-2 question
("What is the intent or category of this text?") and hypothesis template
("The intent or category of this text is {}.") both contain the word "text",
and CLINC-150 ships an intent named exactly "text" (send a text message),
which is in universe_v2.

Measured impact (K=64, CLINC near tier, 'text' offered in 93/200 items):
  laya                 50 of 53 errors were 'text'  (94%)
  gliclass-large-v3.0   0 of 29 errors were 'text'   (0%)
  bge-large-en-v1.5     0 of 22 errors were 'text'   (0%)
Also observed in far-tier cells of non-CLINC domains, since 'text' enters
their far pools as a cross-source option.

Consequence: Laya's interim near-tier collapse (Delta +0.235 at K=64) was
largely this artifact, not semantic near-distractor difficulty. Every cell
collected under the v1 wording is contaminated to the extent 'text' was in
the option pool.

Action taken (before any further inference):
1. The 31 completed cells are DISCARDED for confirmatory purposes and
   archived at results/attic/stage2_v1_collision/ for the record.
2. Prompt reworded to QUESTION="Which label applies here?" /
   TEMPLATE="This example is labeled {}." — no content word of which is an
   option in universe_v2.
3. assert_no_collision() added to the runner: it now refuses to start if any
   option equals a prompt content word, and warns on substring overlap.
4. Stage 2 restarted from zero cells under the new wording.

Note for the write-up: this is also a substantive observation, not only a bug.
Lexical collision between prompt wording and an option string degraded ONE
architecture badly and the others not at all, which is itself a robustness
difference worth reporting (as its own controlled probe, not as RQ3).
