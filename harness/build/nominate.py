"""Committed reproduction of the conflict-nomination step (review fix 5).

The authoritative nomination rule: CROSS-SOURCE pairs only (same-source label
sets are pre-adjudicated by their dataset authors), nominated by mpnet cosine
>= 0.60 OR token containment after stopword removal. universe.py's internal
flagged-pairs list (cosine >= 0.80, all pairs) is exploratory only and is
superseded by this script's output.

Reads results/universe_draft.json, writes results/universe_cross_nominations.md.
Usage: .venv/bin/python -m harness.build.nominate
"""
from __future__ import annotations

import json
import pathlib

COS_T = 0.60        # cross-source threshold
COS_T_SAME = 0.75   # same-source threshold: siblings are meant to be close, so
                    # only near-interchangeable ones are nominated (review S3:
                    # v2 exempted same-source entirely, leaving the whole NEAR
                    # tier unadjudicated)
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

    def contains(a, b):
        """Whole-word containment. v2 guarded on `toks[a] and toks[b]`, which
        skipped any option whose tokens are all stopwords — 'no' vs
        'no waivers'/'no defaults'/'no conflicts' all escaped (review, Low)."""
        wa, wb = a.split(), b.split()
        if len(wa) == len(wb):
            return False
        short, long_ = (wa, wb) if len(wa) < len(wb) else (wb, wa)
        return all(w in long_ for w in short)

    noms = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            same = bool(src[a] & src[b])
            cos = float(sims[i, j])
            contain = contains(a, b)
            thr = COS_T_SAME if same else COS_T
            if cos >= thr or contain:
                noms.append({"cos": round(cos, 3), "contain": contain, "same_source": same,
                             "a": a, "b": b, "src_a": sorted(src[a]), "src_b": sorted(src[b])})
    noms.sort(key=lambda r: -r["cos"])
    lines = [f"# Conflict nominations. CROSS-source: cos>={COS_T} or containment. "
             f"SAME-source: cos>={COS_T_SAME} or containment (v3: same-source is no longer "
             "exempt — it is exactly the NEAR tier).", ""]
    for r in noms:
        tag = "CONTAIN" if r["contain"] else "       "
        scope = "SAME" if r["same_source"] else "CROSS"
        lines.append(f"- [{r['cos']:.3f}] {scope:5s} {tag}  {r['a']!r} ({'/'.join(r['src_a'])})  vs  "
                     f"{r['b']!r} ({'/'.join(r['src_b'])})")
    pathlib.Path("results/universe_cross_nominations.md").write_text("\n".join(lines))
    print(f"{len(noms)} nominations written")


if __name__ == "__main__":
    main()
