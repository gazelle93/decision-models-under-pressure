"""Phase A of the distractor plan: build the candidate option universe,
normalize, and flag near-duplicate pairs for human adjudication.

Filter model: all-mpnet-base-v2 — local, and deliberately NOT in the eval
roster (filtering with a model under test would inflate its own large-K
scores). Authority for merge/conflict rulings is human; embeddings only
nominate candidates.

Outputs:
  results/universe_draft.json        raw + normalized options with sources
  results/universe_flagged_pairs.md  clusters above threshold w/ proposed rulings

Usage: .venv/bin/python -m harness.universe
"""
from __future__ import annotations

import json
import pathlib
import re
import time
import traceback

OUT = pathlib.Path("results")
FLAG_T = 0.80      # exploratory flagging only; authoritative nomination is harness/nominate.py (cos>=0.60 cross-source + containment)
AUTO_MERGE_T = 0.93  # propose MERGE above this; CONFLICT in between


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(s):
    """Word-boundary-preserving normalizer.

    v2 lowercased BEFORE splitting, destroying CamelCase and dot boundaries:
    'AmusementParkAttraction' -> 'amusementparkattraction',
    'comp.sys.ibm.pc.hardware' -> 'compsysibmpchardware'. 92 of 537 options
    were mangled, and because the mangled sources are structurally FAR-only
    this produced a format tell that let a text-free classifier locate the
    gold (review 2026-09-23, F1). Split first, then lowercase.
    """
    s = str(s)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)       # camelCase -> camel Case
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)     # HTTPServer -> HTTP Server
    for ch in "_-./\\":
        s = s.replace(ch, " ")
    s = s.lower()
    s = re.sub(r"[^\w\s&|]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def collect():
    from datasets import load_dataset
    from .registry import TWITTER_FIN_TOPICS

    options = {}  # norm -> {raw, sources:set}
    def add(names, source):
        n_added = 0
        for raw in names:
            key = norm(str(raw))
            if not key or key == "oos":
                continue
            if key not in options:
                options[key] = {"raw": str(raw), "sources": set()}
                n_added += 1
            options[key]["sources"].add(source)
        log(f"  {source}: +{n_added} (total {len(options)})")

    clinc = load_dataset("clinc/clinc_oos", "plus", split="test")
    add([n for n in clinc.features["intent"].names if n != "oos"], "clinc150")
    b77 = load_dataset("mteb/banking77", split="test")
    add({r["label_text"] for r in b77}, "banking77")
    add(TWITTER_FIN_TOPICS, "fin_topics")
    try:
        ge = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
        add(ge.features["labels"].feature.names, "go_emotions")
    except Exception:
        log("  go_emotions FAILED: " + traceback.format_exc(limit=1).strip().splitlines()[-1])
    try:
        db = load_dataset("fancyzhx/dbpedia_14", split="test")
        add(db.features["label"].names, "dbpedia14")
    except Exception:
        log("  dbpedia FAILED")
    FAILED = []

    STRIP_PREFIX = {"mtop": "IN:"}   # MTOP ships IN:GET_WEATHER; the colon would
                                     # normalize to a shared leading token "in"
                                     # on all 102 labels -> perfect tier signal.

    def add_source(hf_id, cfg, split, src, feat=None):
        """Generic label extraction: ClassLabel names, else label_text uniques,
        else string-label uniques on `feat` or 'label'."""
        try:
            ds = load_dataset(hf_id, cfg, split=split)
            col = feat or ("label_text" if "label_text" in ds.features else "label")
            pre = STRIP_PREFIX.get(src)
            if col in ds.features and hasattr(ds.features[col], "names"):
                names = ds.features[col].names
                add([n.replace(pre, "") for n in names] if pre else names, src)
            elif col in ds.features:
                vals = {str(r[col]).replace(pre, "") if pre else str(r[col]) for r in ds}
                if all(not v.lstrip("-").isdigit() for v in list(vals)[:20]):
                    add(vals, src)
                else:
                    log(f"  {src}: labels are bare ints with no names, skipped")
            else:
                log(f"  {src}: no usable label column, skipped")
        except Exception:
            FAILED.append(src)
            log(f"  {src} FAILED: " + traceback.format_exc(limit=1).strip().splitlines()[-1])

    for hf_id, cfg, split, src, feat in [
        ("coastalcph/lex_glue", "ledgar", "test", "ledgar", "label"),
        ("SetFit/20_newsgroups", None, "test", "20news", "label_text"),
        ("Davlan/sib200", "eng_Latn", "test", "sib200", "category"),
        # stage-2 growth (2026-09-22): toward 600+ canonical options
        ("mteb/amazon_massive_intent", "en", "test", "massive", None),
        ("DeepPavlov/hwu64", None, "test", "hwu64", None),
        ("DeveloperOats/DBPedia_Classes", None, "test", "dbpedia_l2", "l2"),
        ("CogComp/trec", None, "test", "trec_fine", "fine_label"),
        ("fancyzhx/ag_news", None, "test", "ag_news", "label"),
        ("yahoo_answers_topics", None, "test", "yahoo", "topic"),
        ("sonos-nlu-benchmark/snips_built_in_intents", None, "train", "snips", "label"),
        ("WillHeld/mtop", None, "test_en", "mtop", "intent"),
        # v3 growth: bucket supply was the binding constraint on format-neutral
        # sampling (a 532-option universe cannot fill a gold's surface stratum
        # at K=64), so breadth here directly buys gate headroom.
        ("DeveloperOats/DBPedia_Classes", None, "test", "dbpedia_l3", "l3"),
        ("mteb/amazon_massive_scenario", "en", "test", "massive_scenario", None),
        ("dair-ai/emotion", None, "test", "dair_emotion", "label"),
        ("SetFit/TREC-QC", None, "test", "trec_qc", "label_text"),
        ("clinc/clinc_oos", "small", "test", "clinc_domain", "domain"),
    ]:
        add_source(hf_id, cfg, split, src, feat)
    if FAILED:
        log(f"  NOTE {len(FAILED)} sources failed to load and are EXCLUDED: {FAILED}")
        log("  (recorded in the universe manifest; not silently dropped)")
    options["__failed_sources__"] = {"raw": ",".join(FAILED), "sources": set()}
    return options


def main():
    OUT.mkdir(exist_ok=True)
    options = collect()
    failed = options.pop("__failed_sources__", {"raw": ""})["raw"]
    keys = sorted(options)
    log(f"universe candidates after exact dedupe: {len(keys)}")

    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    emb = m.encode(keys, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    import numpy as np
    sims = np.asarray(emb) @ np.asarray(emb).T
    flagged = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if sims[i, j] >= FLAG_T:
                flagged.append((float(sims[i, j]), keys[i], keys[j]))
    flagged.sort(reverse=True)
    log(f"flagged pairs at cosine >= {FLAG_T}: {len(flagged)}")

    # union-find clusters over flagged pairs
    parent = {k: k for k in keys}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for _, a, b in flagged:
        parent[find(a)] = find(b)
    clusters = {}
    for _, a, b in flagged:
        clusters.setdefault(find(a), set()).update([a, b])

    lines = ["# Flagged near-duplicate clusters — adjudication list",
             "",
             f"Filter: all-mpnet-base-v2 cosine >= {FLAG_T}; proposed MERGE at >= {AUTO_MERGE_T}.",
             "Rulings: MERGE = true synonyms, keep first as canonical. CONFLICT = distinct",
             "but confusable; both stay in the universe, pair enters the conflict matrix",
             "(never co-occur as gold + distractor). Edit the ruling column, then freeze.",
             ""]
    cluster_list = sorted(clusters.values(), key=lambda c: -len(c))
    for ci, members in enumerate(cluster_list, 1):
        ms = sorted(members)
        pair_sims = [s for s, a, b in flagged if a in members and b in members]
        top = max(pair_sims)
        ruling = "MERGE" if top >= AUTO_MERGE_T else "CONFLICT"
        srcs = {s for k in ms for s in options[k]["sources"]}
        lines.append(f"## C{ci:03d} [{ruling}] max-cos {top:.3f} ({', '.join(sorted(srcs))})")
        for k in ms:
            lines.append(f"- {k}")
        lines.append("")
    (OUT / "universe_flagged_pairs.md").write_text("\n".join(lines))

    (OUT / "universe_draft.json").write_text(json.dumps({
        "filter_model": "sentence-transformers/all-mpnet-base-v2",
        "flag_threshold": FLAG_T, "auto_merge_threshold": AUTO_MERGE_T,
        "n_options": len(keys),
        "failed_sources": [s for s in failed.split(",") if s],
        "options": {k: {"raw": v["raw"], "sources": sorted(v["sources"])} for k, v in options.items()},
        "flagged_pairs": [{"cos": round(s, 4), "a": a, "b": b} for s, a, b in flagged],
    }, indent=1))
    log(f"clusters to adjudicate: {len(cluster_list)}")
    log("wrote results/universe_draft.json and results/universe_flagged_pairs.md")


if __name__ == "__main__":
    main()
