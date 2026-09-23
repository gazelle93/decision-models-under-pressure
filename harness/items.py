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

    from .labels import TWITTER_FIN_TOPICS
    ft = load_dataset("zeroshot/twitter-financial-news-topic", split="validation")
    picked = random.Random(SEED).sample(list(ft), max_n)[:n]
    out["fintopic"] = [{"text": r["text"], "gold": norm(TWITTER_FIN_TOPICS[r["label"]]),
                        "source": "fin_topics"} for r in picked]

    db = load_dataset("DeveloperOats/DBPedia_Classes", split="test")
    picked = random.Random(SEED).sample(list(range(len(db))), max_n)[:n]
    out["dbpedia"] = [{"text": db[i]["text"], "gold": norm(db[i]["l2"]),
                       "source": "dbpedia_l2"} for i in picked]

    mt = load_dataset("WillHeld/mtop", split="test_en")
    picked = random.Random(SEED).sample(list(range(len(mt))), max_n)[:n]
    out["mtop"] = [{"text": mt[i]["utterance"],
                    "gold": norm(mt[i]["intent"].replace("IN:", "")),
                    "source": "mtop", "mtop_domain": mt[i]["domain"]} for i in picked]

    for d, items in out.items():
        keep = []
        for j, it in enumerate(items):
            it["domain"] = d
            it["uid"] = f"{d}:{j}"
            it["gold"] = alias_of.get(it["gold"], it["gold"])
            it["leaked"] = it["gold"] in it["text"].lower()
            it["text_chars"] = len(it["text"])          # RQ1 second factor
            keep.append(it)
        out[d] = keep
        L = sorted(x["text_chars"] for x in keep)
        log(f"  {d}: n={len(keep)}, leakage {sum(x['leaked'] for x in keep)/len(keep):.1%}, "
            f"text chars median {L[len(L)//2]} p90 {L[int(len(L)*0.9)]}")
    return out
