# Decision Models Under Pressure

A comparative study of the three architecture families behind request-time
decision models — the systems that take *state + a question + candidate
options* and return a probability distribution over those options.

**The question.** Nobody has published how these families behave as the
candidate set grows, as option order changes, or as distractors get
semantically closer. The 2026 Jev/Laya wave made the question timely; this
study answers it for as-shipped systems.

## Research questions

| | Question | Domains | Design |
|---|---|---|---|
| **RQ1** | How do accuracy and probability quality degrade with candidate-set size **and text length**? | clinc, mtop, goemotions, dbpedia | two-factor: K ∈ 2…256 × text length (DBpedia supplies a 10× length spread) |
| **RQ2** | How often does a model change its answer when only the option *order* changes? | clinc, goemotions, fintopic, mtop | 5 permutations at K ∈ {16, 64} |
| **RQ3** | How much does accuracy fall when distractors are semantically near rather than far? | clinc, mtop | paired near/far tiers, K ∈ 2…64 |

## Architecture families

| | Mechanism | Roster |
|---|---|---|
| **A1** pair cross-encoders | one pass per option | deberta-v3-{base,large}-zeroshot-v2.0 |
| **A2** embedding scorers | one text pass, options are cached vectors | bge-large-en-v1.5, gte-large |
| **A3** option-conditioned | single pass, all options as marker tokens | laya, gliclass-large-v3.0 |

Jev joins A3 when API credentials exist (K ≤ 64 is well inside its 255-option
cap). It is hosted, un-rerunnable and rounds probabilities to 2 decimals, so
its calibration metrics are flagged rounding-limited.

## Repository

```
dataset/v3/          the frozen dataset — items, universe, gates, splits, datacard
harness/             the live pipeline (see below)
docs/history/        the audit that produced this design; read before changing it
archive/pre-v3/      superseded code and results, kept for provenance only
PREREGISTRATION.md   what will be run and what would falsify it
EXPERIMENTS.md       the running log of what was actually run
```

### Pipeline

| module | role |
|---|---|
| `universe.py` → `nominate.py` → `freeze_universe.py` | build and adjudicate the option universe |
| `items.py`, `tiers.py` | draw items; build near/far/ext distractor pools |
| `gates.py` | eight pre-inference assertions; **nothing runs until they pass** |
| `build_v3.py` | build + gate in one command |
| `export_dataset.py` | freeze to `dataset/<version>/` with checksums |
| `dataset.py` | `load_rq("rq3")` — enforces each RQ's scope from the data |
| `models.py` | the six model adapters |
| `run_v3.py` | the experiment runner |

## Running it

```bash
.venv/bin/python -m harness.build_v3 --n 200        # build + gate
.venv/bin/python -m harness.export_dataset          # freeze to dataset/v3
.venv/bin/python -m harness.run_v3 --rq rq3         # run a research question
```

Experiments read the frozen dataset through `load_rq`, never a rebuild, so an
RQ's domain scope is a property of the data rather than something an analyst
has to remember.

## Method commitments

- **Gates before inference.** Format neutrality, gold-position uniformity, tier
  separation, pairing feasibility and four more are assertions, not diagnostics.
  Current status is in `dataset/v3/gates.json`; one gate (G3) is open and its
  per-cell magnitude is reported beside every accuracy table.
- **Two-stage N.** Pilots at n=50 are labelled as pilots and never headline.
  Confirmatory runs use fresh items, pre-registered contrasts and power-derived N.
- **Generalization rungs.** Every cell is graded by how much is genuinely unseen,
  from the *disclosed* training data only. No domain in this space is clean for
  every model family, and that is reported rather than averaged away.
- **Worst case over perturbations, not reruns.** These models are deterministic
  forward passes, so variation is measured over option orders, phrasings and
  few-shot resamples — never over repeated identical calls.
