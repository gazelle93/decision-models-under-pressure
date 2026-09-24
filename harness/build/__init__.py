"""Dataset construction: label universe -> items -> distractor tiers -> gates.

Run the whole pipeline with `python -m harness.build`. The individual stages
are runnable too, and are listed in the order they execute:

    python -m harness.build.universe   collect and normalise label strings
    python -m harness.build.nominate   nominate near-duplicate pairs
    python -m harness.build.freeze     apply rulings, freeze the universe
    python -m harness.build            draw items, build tiers, run the gates
    python -m harness.build.export     write dataset/<version>/

Nothing in here runs at experiment time. Experiments read the frozen dataset
through `harness.dataset.load_rq`.
"""
