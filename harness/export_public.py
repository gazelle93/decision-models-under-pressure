"""Export a REDISTRIBUTABLE version of the dataset.

The item texts come from five upstream datasets with incompatible terms — one
of them (twitter-financial-news-topic) is tweet text with no stated licence,
which X's terms restrict from redistribution. So the public release ships
everything EXCEPT the text: uids, golds, distractor lists, every derived field,
plus a SHA-256 of each text so a rebuild can be verified byte-for-byte.

`rebuild_texts.py` restores the texts from the original sources. Nothing is
lost scientifically; nothing encumbered is redistributed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import shutil

SRC = pathlib.Path("dataset/v3")
OUT = pathlib.Path("dataset/v3-public")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with (OUT / "items.jsonl").open("w") as f:
        for line in (SRC / "items.jsonl").read_text().splitlines():
            r = json.loads(line)
            text = r.pop("text")
            r["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            f.write(json.dumps(r) + "\n")
            n += 1
    # csv view, text column replaced by its hash
    with (SRC / "items.csv").open() as fi, (OUT / "items.csv").open("w", newline="") as fo:
        rd = csv.reader(fi); w = csv.writer(fo)
        head = next(rd); ti = head.index("text"); head[ti] = "text_sha256"
        w.writerow(head)
        for row in rd:
            row[ti] = hashlib.sha256(row[ti].encode()).hexdigest()[:16]
            w.writerow(row)
    for name in ("universe.json", "gates.json", "manifest.json", "README.md"):
        shutil.copy(SRC / name, OUT / name)
    shutil.copytree(SRC / "splits", OUT / "splits", dirs_exist_ok=True)

    man = json.loads((OUT / "manifest.json").read_text())
    man["distribution"] = {
        "texts_included": False,
        "reason": "upstream licences differ and one source is tweet text with no "
                  "stated licence; see README",
        "restore_with": "python -m harness.rebuild_texts",
        "verification": "each item carries text_sha256 of its original text",
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {OUT}/ — {n} items, texts replaced by SHA-256")


if __name__ == "__main__":
    main()
