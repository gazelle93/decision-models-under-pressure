"""Item construction v3: fresh, nested-by-design, leakage-annotated."""
from __future__ import annotations

import random

SEED = 101


def _pilot_clinc_texts(clinc):
    rows = [r for r in clinc if clinc.features["intent"].names[r["intent"]] != "oos"]
    return {r["text"] for r in random.Random(7).sample(rows, 50)}


def build_items(n, alias_of, max_n=500, log=print):
    """Draws max_n once and takes the first n, so the ladder ALWAYS nests.

    v2 called random.sample(pop, n) per rung; for fintopic (4117 rows) n=500
    crossed CPython's sample-algorithm boundary and only 40/200 items recurred
    (review F9).
    """
    from datasets import load_dataset
    from .universe import norm

    out = {}
    clinc = load_dataset("clinc/clinc_oos", "plus", split="test")
    names = clinc.features["intent"].names
    pilot = _pilot_clinc_texts(clinc)
    rows = [r for r in clinc if names[r["intent"]] != "oos" and r["text"] not in pilot]
    picked = random.Random(SEED).sample(rows, max_n)[:n]
    out["clinc"] = [{"text": r["text"], "gold": norm(names[r["intent"]]), "source": "clinc150"}
                    for r in picked]

    ge = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    gnames = ge.features["labels"].feature.names
    rows = [r for r in ge if len(r["labels"]) == 1]
    picked = random.Random(SEED).sample(rows, max_n)[:n]
    out["goemotions"] = [{"text": r["text"], "gold": norm(gnames[r["labels"][0]]),
                          "source": "go_emotions"} for r in picked]

    from .registry import TWITTER_FIN_TOPICS
    ft = load_dataset("zeroshot/twitter-financial-news-topic", split="validation")
    picked = random.Random(SEED).sample(list(ft), max_n)[:n]
    out["fintopic"] = [{"text": r["text"], "gold": norm(TWITTER_FIN_TOPICS[r["label"]]),
                        "source": "fin_topics"} for r in picked]

    for d, items in out.items():
        keep = []
        for j, it in enumerate(items):
            it["domain"] = d
            it["uid"] = f"{d}:{j}"
            it["gold"] = alias_of.get(it["gold"], it["gold"])
            it["leaked"] = it["gold"] in it["text"].lower()
            keep.append(it)
        out[d] = keep
        log(f"  {d}: n={len(keep)}, leakage {sum(x['leaked'] for x in keep)/len(keep):.1%}")
    return out
