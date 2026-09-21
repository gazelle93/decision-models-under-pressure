# decision-model

Research program: a 100-500M encoder decision model (request-time answer
spaces, calibrated softmax, 1,024-2,048 token inputs) whose target is strong
task accuracy from 10-100 labels per decision option.

The governing research brief (framing, recipe evidence, dataset plan,
protocol, resolved decisions) lives here:
https://claude.ai/code/artifact/030ca553-fc0a-4200-863d-70ac0103862b

## Layout

- `harness/registry.py` - the 12-slot held-out eval suite (wave 1 wired,
  rest declared as todo). Nothing in here may ever enter training data.
- `harness/metrics.py` - accuracy, macro-F1, Brier, NLL, 15-bin ECE.
- `harness/models.py` - model adapters (v0: zero-shot NLI pipeline).
- `harness/run.py` - runner; per-decision JSONL + metrics JSON in `results/`.

## Quickstart

    .venv/bin/python -m harness.run \
      --model MoritzLaurer/deberta-v3-base-zeroshot-v2.0 \
      --dataset enron_spam --limit 100

## Protocol notes / TODO

- Pin dataset `revision` fields after the first frozen runs (elcronos-style
  frozen protocol before any headline numbers).
- Debiased ECE estimator not yet implemented (15-bin equal-width only).
- Few-shot sampler (K in {10,25,50,100} per option x 5 seeded resamples)
  lands with the fine-tuning phase.
- Baselines still to wire: GLiClass-large-v3.0, Laya checkpoints, SetFit,
  bart-large-mnli, one constrained LLM ceiling.
- Long-input handling: the NLI pipeline truncates at the model's max length;
  fine for wave-1 short sets, must be revisited for arxiv/ECtHR slots.
