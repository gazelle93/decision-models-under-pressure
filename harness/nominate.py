"""Committed reproduction of the conflict-nomination step (review fix 5).

The authoritative nomination rule: CROSS-SOURCE pairs only (same-source label
sets are pre-adjudicated by their dataset authors), nominated by mpnet cosine
>= 0.60 OR token containment after stopword removal. universe.py's internal
flagged-pairs list (cosine >= 0.80, all pairs) is exploratory only and is
superseded by this script's output.

Reads results/universe_draft.json, writes results/universe_cross_nominations.md.
Usage: .venv/bin/python -m harness.nominate
"""
from __future__ import annotations

import json
import pathlib

COS_T = 0.60
STOP = {"the", "a", "an", "of", "by", "to", "in", "for", "and", "or", "not", "no",
        "my", "your", "is", "are", "on", "at", "with", "about", "up", "after",
        "into", "get", "getting"}


def main():
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    d = json.loads(pathlib.Path("results/universe_draft.json").read_text())
    opts = d["options"]
    keys = sorted(opts)
    src = {k: set(opts[k]["sources"]) for k in keys}

    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                            device="mps" if torch.backends.mps.is_available() else "cpu")
    emb = np.asarray(m.encode(keys, normalize_embeddings=True, batch_size=64))
    sims = emb @ emb.T
    toks = {k: set(t for t in k.split() if t not in STOP) for k in keys}

    noms = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if src[a] & src[b]:
                continue  # same-source exempt
            cos = float(sims[i, j])
            contain = bool(toks[a] and toks[b] and (toks[a] <= toks[b] or toks[b] <= toks[a]))
            if cos >= COS_T or contain:
                noms.append({"cos": round(cos, 3), "contain": contain, "a": a, "b": b,
                             "src_a": sorted(src[a]), "src_b": sorted(src[b])})
    noms.sort(key=lambda r: -r["cos"])
    lines = [f"# Cross-source conflict nominations (rule: cos>={COS_T} OR token containment; "
             "same-source pairs exempt — source datasets pre-adjudicate their own labels)", ""]
    for r in noms:
        tag = "CONTAIN" if r["contain"] else "       "
        lines.append(f"- [{r['cos']:.3f}] {tag}  {r['a']!r} ({'/'.join(r['src_a'])})  vs  "
                     f"{r['b']!r} ({'/'.join(r['src_b'])})")
    pathlib.Path("results/universe_cross_nominations.md").write_text("\n".join(lines))
    print(f"{len(noms)} nominations written")


if __name__ == "__main__":
    main()
