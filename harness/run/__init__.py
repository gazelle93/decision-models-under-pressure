"""Experiment runners.

    python -m harness.run.local --rq rq3    the six local models
    python -m harness.run.jev   --rq rq3    the hosted Jev arm (costs money)

Both read the frozen dataset via `harness.dataset.load_rq` and write per-item
JSONL to results/.
"""
