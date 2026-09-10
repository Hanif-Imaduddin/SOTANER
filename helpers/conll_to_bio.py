"""Build the 4-column BIO tree from a local CoNLL-2012 v4 OntoNotes checkout.

This replaces the upstream ``conll-to-bio.py`` (which globbed a flat
``data/<split>/<genre>/`` tree of bare ``*.gold_conll`` files). Here we read the
real, licensed **CoNLL-2012 v4** layout the user supplied:

    conll-2012/v4/data/{train,development,test}/data/english/annotations/<genre>/**/*.v4_gold_conll

The gold ``*.v4_gold_conll`` column layout (0-indexed, whitespace-aligned):

    0 doc-id | 1 part | 2 word-no | 3 TOKEN | 4 POS | 5 parse-bit |
    6 lemma | 7 frameset | 8 sense | 9 speaker | 10 NER-bracket | 11..N-2 SRL | N-1 coref

The NER column uses the ``(TYPE* ... *)`` / ``(TYPE)`` bracket convention, turned
into IOB2 by the same ``flag`` state machine the upstream script used.

Fixes vs upstream: reads ``.v4_gold_conll`` only, honours the deeper CoNLL-2012
path, ignores ``#begin/#end document`` lines, ``encoding="utf-8"`` everywhere,
creates output dirs, emits the ``news`` / ``all6`` / ``everything`` roll-ups plus
a combined train+dev+test file, and writes each file exactly once (no O(n^2)).
"""
from __future__ import annotations

import glob
import os

from .bio_utils import (
    ALL6,
    DOCSTART,
    GENRES,
    NEWS,
    count_entities,
    write_bio,
)

# CoNLL-2012 split dir -> the split name this pipeline uses.
SPLIT_DIRS = {"train": "train", "development": "development", "test": "test"}


def _genre_dir(v4_data_root, split_dir, genre):
    return os.path.join(v4_data_root, split_dir, "data", "english", "annotations", genre)


def _gold_conll_files(v4_data_root, split_dir, genre):
    root = _genre_dir(v4_data_root, split_dir, genre)
    return sorted(glob.glob(os.path.join(root, "**", "*.v4_gold_conll"), recursive=True))


def _file_to_blocks(path):
    """One ``*.v4_gold_conll`` file -> list of sentence-block strings (4-col BIO)."""
    blocks = []
    cur = []
    flag = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            l = line.strip()
            if not l or l.startswith("#begin document") or l.startswith("#end document"):
                if cur:
                    blocks.append("\n".join(cur))
                    cur = []
                flag = None
                continue
            ls = l.split()
            if len(ls) < 11:
                # malformed / not a token line -> sentence break
                if cur:
                    blocks.append("\n".join(cur))
                    cur = []
                flag = None
                continue
            word, pos, cons, ori_ner = ls[3], ls[4], ls[5], ls[10]
            if ori_ner == "*":
                ner = "O" if flag is None else "I-" + flag
            elif ori_ner == "*)":
                ner = "I-" + flag
                flag = None
            elif ori_ner.startswith("(") and ori_ner.endswith("*") and len(ori_ner) > 2:
                flag = ori_ner[1:-1]
                ner = "B-" + flag
            elif ori_ner.startswith("(") and ori_ner.endswith(")") and len(ori_ner) > 2 and flag is None:
                ner = "B-" + ori_ner[1:-1]
            else:
                ner = "O"
            cur.append("\t".join([word, pos, cons, ner]))
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _collect_genre(v4_data_root, split_dir, genre):
    blocks = []
    for f in _gold_conll_files(v4_data_root, split_dir, genre):
        blocks.extend(_file_to_blocks(f))
    return blocks


def build_all(v4_data_root, out_dir, verbose=True):
    """Convert every split/genre under ``v4_data_root`` to the BIO tree at ``out_dir``.

    Writes, per split (train / development / test):
        onto.<genre>.ner   for the 7 genres
        onto.news.ner       (bn + mz + nw)
        onto.all6.ner       (bn, mz, nw, bc, tc, wb -- the standard NER test split)
        onto.everything.ner (== all6; the six NER sources, pt excluded)
    plus a combined ``onto.everything.ner`` at ``out_dir`` root (train+dev+test),
    which ``generate_splits`` consumes, and ``_genre_report.txt``.

    Returns the report as a list of strings.
    """
    report = []
    combined = []  # train ++ dev ++ test of the six NER genres

    for split_dir, split in SPLIT_DIRS.items():
        per_genre = {}
        for g in GENRES:
            per_genre[g] = _collect_genre(v4_data_root, split_dir, g)

        news = [b for g in NEWS for b in per_genre[g]]
        all6 = [b for g in ALL6 for b in per_genre[g]]

        for g in GENRES:
            write_bio(os.path.join(out_dir, split, f"onto.{g}.ner"), per_genre[g])
        write_bio(os.path.join(out_dir, split, "onto.news.ner"), news)
        write_bio(os.path.join(out_dir, split, "onto.all6.ner"), all6)
        write_bio(os.path.join(out_dir, split, "onto.everything.ner"), all6)
        combined.extend(all6)

        for g in GENRES:
            report.append(
                f"{split:12s} {g:4s} sents={len(per_genre[g]):6d}  entities={count_entities(per_genre[g]):6d}"
            )
        report.append(f"{split:12s} news sents={len(news):6d}  entities={count_entities(news):6d}")
        report.append(f"{split:12s} all6 sents={len(all6):6d}  entities={count_entities(all6):6d}")
        report.append("")

    write_bio(os.path.join(out_dir, "onto.everything.ner"), combined)
    report.append(
        f"{'combined':12s} all6 sents={len(combined):6d}  entities={count_entities(combined):6d}"
    )

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "_genre_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")

    if verbose:
        print("\n".join(report))
        print("\nwrote BIO tree under:", out_dir)
    return report
