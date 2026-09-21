"""Eval-set registry: the 12-slot held-out suite from the research brief (Section 6).

Wave 1 is wired; the rest are declared with status="todo" so the suite's shape
is visible from day one. Never add a training-mixture dataset here, and never
train on anything listed here. Pin `revision` after the first frozen run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Example:
    text: str
    gold: int  # index into options
    options: list


@dataclass
class DatasetSpec:
    key: str
    hf_id: str
    config: Optional[str]
    split: str
    slot: str
    status: str = "ready"  # ready | todo
    hypothesis_template: str = "This example is {}."
    notes: str = ""
    loader: Optional[Callable] = None
    revision: Optional[str] = None


def _clean(name):
    return name.replace("_", " ").strip()


def _load_single_label(spec, limit, text_col, label_col, options=None):
    from datasets import load_dataset

    ds = load_dataset(spec.hf_id, spec.config, split=spec.split, revision=spec.revision)
    opts = options or [_clean(n) for n in ds.features[label_col].names]
    rows = ds if limit is None else ds.select(range(min(limit, len(ds))))
    return [Example(text=r[text_col], gold=int(r[label_col]), options=opts) for r in rows]


def _load_daily_dialog(spec, limit):
    """Flatten dialogues to utterances with emotion labels (frozen-Jev-protocol style)."""
    from datasets import load_dataset

    ds = load_dataset(spec.hf_id, split=spec.split, revision=spec.revision)
    opts = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    out = []
    for row in ds:
        for utt, emo in zip(row["dialog"], row["emotion"]):
            out.append(Example(text=utt.strip(), gold=int(emo), options=opts))
            if limit is not None and len(out) >= limit:
                return out
    return out


REGISTRY = {}


def _register(spec):
    REGISTRY[spec.key] = spec
    return spec


_register(DatasetSpec(
    key="enron_spam", hf_id="SetFit/enron_spam", config=None, split="test",
    slot="binary",
    loader=lambda s, l: _load_single_label(s, l, "text", "label",
                                           options=["a legitimate email", "spam"]),
    hypothesis_template="This email is {}.",
    notes="Clean vs Laurer v1.1 / tasksource / FLAN.",
))
_register(DatasetSpec(
    key="daily_dialog_emotion", hf_id="OpenRL/daily_dialog", config=None, split="test",
    slot="emotion-7", loader=_load_daily_dialog,
    hypothesis_template="The emotion in this utterance is {}.",
    notes="Frozen Jev protocol set; CC-BY-NC (eval only).",
))
_register(DatasetSpec(
    key="twitter_fin_topic", hf_id="zeroshot/twitter-financial-news-topic", config=None,
    split="validation", slot="topic-20",
    loader=lambda s, l: _load_single_label(s, l, "text", "label"),
    hypothesis_template="This financial news tweet is about {}.",
    notes="Frozen Jev protocol set; license unknown (flagged in brief).",
))
_register(DatasetSpec(
    key="sst5", hf_id="SetFit/sst5", config=None, split="test",
    slot="ordinal-short",
    loader=lambda s, l: _load_single_label(s, l, "text", "label",
        options=["very negative", "negative", "neutral", "positive", "very positive"]),
    hypothesis_template="The sentiment of this review is {}.",
    notes="SST-2 sentence overlap with tasksource/FLAN flagged.",
))
_register(DatasetSpec(
    key="clinc150_oos", hf_id="clinc/clinc_oos", config="plus", split="test",
    slot="intent-150+oos",
    loader=lambda s, l: _load_single_label(s, l, "text", "intent"),
    hypothesis_template="The intent of this request is {}.",
    notes="Cardinality stress + out-of-scope detection (oos label).",
))
_register(DatasetSpec(
    key="banking77", hf_id="PolyAI/banking77", config=None, split="test",
    slot="comparability-only",
    loader=lambda s, l: _load_single_label(s, l, "text", "label"),
    hypothesis_template="The customer's banking intent is {}.",
    notes="CONTAMINATED vs Laurer/tasksource/RAFT; literature comparability only.",
))

for key, hf_id, slot in [
    ("amazon_products_106", "FastFit/amazon_products", "cardinality-106"),
    ("asap2_essays", "scrosseye/ASAP_2.0 (github release)", "ordinal-long"),
    ("arxiv_11", "ccdv/arxiv-classification", "long-input-2048"),
    ("ecthr_a", "coastalcph/lex_glue:ecthr_a", "long-legal-multilabel"),
    ("phishnchips", "AreLit/PhishNChips", "guardrail-calibration"),
    ("typed_decisions", "LocalLLaMA/typed-decisions", "product-shape"),
]:
    _register(DatasetSpec(key=key, hf_id=hf_id, config=None, split="test", slot=slot, status="todo"))
