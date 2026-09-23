"""Rung annotation (brief Section 5, ladder G0-G6) for zero-shot results.

Grade = highest rung defensible from DISCLOSED training data, per (model, dataset).
Partial/undisclosed training data caps the grade and is marked. Family-level
strictness: a model trained on ANY dataset of a task family is G3 at best on
that family, even on an unseen dataset.
"""
from __future__ import annotations

import json
import pathlib

FAMILY = {
    "enron_spam": "spam",
    "daily_dialog_emotion": "emotion",
    "twitter_fin_topic": "topic",
    "sst5": "sentiment",
    "clinc150_oos": "intent",
    "banking77": "intent",
}

# What each wave-1/2 model's training disclosure supports.
MODELS = {
    "deberta-v3-base-zeroshot-v2.0": dict(
        disclosure="full",
        datasets={"banking77"},
        families={"spam", "emotion", "topic", "sentiment", "intent", "toxicity", "nli"},
        note="v1.1 list: sms_spam; dair-emotion/emocontext/empathetic; ag_news/yahoo; "
             "rotten/imdb/amazon/yelp; massive/banking77 (banking77 itself => G0 there)",
    ),
    "deberta-v3-large-zeroshot-v2.0": dict(
        disclosure="full",
        datasets={"banking77"},
        families={"spam", "emotion", "topic", "sentiment", "intent", "toxicity", "nli"},
        note="same v1.1 list as base",
    ),
    "bart-large-mnli": dict(
        disclosure="full", datasets=set(), families={"nli"},
        note="MNLI only: every suite row is defensible G4",
    ),
    "gliclass-large-v3.0": dict(
        disclosure="partial", datasets=set(), families={"sentiment", "nli"},
        note="1.2M mix described only as 'classification, sentiment, NLI'; row-level provenance undisclosed",
    ),
    "laya": dict(
        disclosure="partial", datasets=set(), families={"spam", "intent", "toxicity", "nli"},
        note="author prose: support triage, NLI/fact, toxicity, jailbreak, sales; "
             "card's own spam/phishing evals read as trained families",
    ),
    "bge-large-en-v1.5": dict(
        disclosure="partial", datasets=set(), families={"sentiment", "nli"},
        note="C-Pack training partially disclosed; sentiment/NLI-adjacent pairs likely",
    ),
}


def rung(model_name, dataset_key):
    m = MODELS.get(model_name)
    if m is None:
        return "UG", "model not in disclosure map"
    if dataset_key in m["datasets"]:
        return "G0", "trained on this dataset"
    fam = FAMILY.get(dataset_key)
    if fam in m["families"]:
        return "G3", f"family '{fam}' in training"
    if m["disclosure"] != "full":
        return "G4*", "no disclosed overlap, but disclosure is partial"
    return "G4", "family absent from full disclosure"


def main():
    out = pathlib.Path("results")
    rows = json.loads((out / "wave1_summary.json").read_text())
    rows = [r for r in rows if "error" not in r]
    annotated = []
    for r in rows:
        g, why = rung(r["model"], r["dataset"])
        r2 = {k: r[k] for k in ("model", "dataset", "accuracy", "macro_f1", "ece_15bin")}
        r2.update({"rung": g, "why": why})
        annotated.append(r2)
        print(f"{r['model']:32s} {r['dataset']:22s} {g:4s} acc {r['accuracy']:.3f}  ({why})")
    # defensible-G4 suite means
    print()
    for m in sorted({r["model"] for r in annotated}):
        g4 = [r for r in annotated if r["model"] == m and r["rung"].startswith("G4")]
        if g4:
            acc = sum(r["accuracy"] for r in g4) / len(g4)
            print(f"{m:32s} G4-defensible rows: {len(g4)}  mean acc {acc:.3f}  "
                  f"({', '.join(r['dataset'] for r in g4)})")
        else:
            print(f"{m:32s} G4-defensible rows: 0  (all rows G3 or below)")
    (out / "wave2_rungs.json").write_text(json.dumps(annotated, indent=2))


if __name__ == "__main__":
    main()
