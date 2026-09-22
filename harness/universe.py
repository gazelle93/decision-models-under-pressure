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
FLAG_T = 0.80      # nominate pairs above this cosine
AUTO_MERGE_T = 0.93  # propose MERGE above this; CONFLICT in between


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(s):
    s = s.lower().replace("_", " ").replace("-", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s&|]", "", s)
    return s.strip()


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
    for hf_id, cfg, feat, src in [
        ("AmazonScience/massive", "en-US", "intent", "massive"),
        ("FastFit/amazon_products", None, "label", "amazon_products"),
        ("coastalcph/lex_glue", "ledgar", "label", "ledgar"),
        ("SetFit/20_newsgroups", None, None, "20news"),
        ("Davlan/sib200", "eng_Latn", "category", "sib200"),
    ]:
        try:
            ds = load_dataset(hf_id, cfg, split="test")
            if src == "20news":
                add({r["label_text"] for r in ds}, src)
            elif feat in ds.features and hasattr(ds.features[feat], "names"):
                add(ds.features[feat].names, src)
            else:
                add({str(r[feat]) for r in ds}, src)
        except Exception:
            log(f"  {src} FAILED: " + traceback.format_exc(limit=1).strip().splitlines()[-1])
    return options


def main():
    OUT.mkdir(exist_ok=True)
    options = collect()
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
        "options": {k: {"raw": v["raw"], "sources": sorted(v["sources"])} for k, v in options.items()},
        "flagged_pairs": [{"cos": round(s, 4), "a": a, "b": b} for s, a, b in flagged],
    }, indent=1))
    log(f"clusters to adjudicate: {len(cluster_list)}")
    log("wrote results/universe_draft.json and results/universe_flagged_pairs.md")


if __name__ == "__main__":
    main()
