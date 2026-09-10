"""Build the 4-column BIO tree from the HuggingFace export of OntoNotes 5.0.

Drop-in alternative to ``conll_to_bio.build_all`` for environments without the
licensed LDC ``conll-2012/v4`` checkout (e.g. Google Colab). Pulls the tokenised
CoNLL-2012 shared-task data from the HuggingFace parquet auto-conversion of
``conll2012_ontonotesv5`` (config ``english_v4``) and emits the *exact same*
tree ``conll_to_bio.build_all`` does, so every downstream cell is unchanged:

    <out_dir>/{train,development,test}/onto.{<genre>,news,all6,everything}.ner
    <out_dir>/onto.everything.ner          (train ++ dev ++ test of the six NER genres)
    <out_dir>/_genre_report.txt

Config ``english_v4`` = the CoNLL-2012 shared-task split. Its test set is
222 docs / 8,262 non-``pt`` sentences / **11,257 entities** -- an exact match to
the paper's Table 3 micro support and to the local ``conll_to_bio`` output.

``datasets`` >= 4 refuses the repo's ``.py`` loader script, so we read the
``refs/convert/parquet`` revision directly (it keeps the ``ClassLabel`` features
needed to decode POS / NER ids).
"""
from __future__ import annotations

import os

from .bio_utils import ALL6, GENRES, NEWS, count_entities, write_bio

# HuggingFace split name -> the split name this pipeline uses.
HF_SPLITS = {"train": "train", "validation": "development", "test": "test"}


def _parquet_base(config):
    return f"hf://datasets/conll2012_ontonotesv5@refs/convert/parquet/{config}"


def _load_split(hf_split, config):
    """Load one split of the parquet export. Tries a shard glob, then 0000.parquet."""
    from datasets import load_dataset

    base = _parquet_base(config)
    for pattern in (f"{base}/{hf_split}/*.parquet", f"{base}/{hf_split}/0000.parquet"):
        try:
            return load_dataset("parquet", data_files={hf_split: pattern}, split=hf_split)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"could not load {hf_split} parquet for config {config}: {last}")


def _sentence_block(words, pos_names, ner_names):
    return "\n".join(
        f"{w}\t{p}\t_\t{n}" for w, p, n in zip(words, pos_names, ner_names)
    )


def build_all(out_dir, config="english_v4", limit=None, verbose=True):
    """HF parquet -> the BIO tree at ``out_dir``. Signature mirrors
    ``conll_to_bio.build_all`` (which takes a local data root instead of ``config``).

    ``limit`` caps sentences per split for a smoke test. Returns the report lines.
    """
    report = []
    combined = []  # train ++ dev ++ test of the six NER genres

    for hf_split, split in HF_SPLITS.items():
        ds = _load_split(hf_split, config)
        sfeat = ds.features["sentences"].feature
        ne = sfeat["named_entities"].feature
        pos = sfeat["pos_tags"].feature

        per_genre = {g: [] for g in GENRES}
        nsent = 0
        stop = False
        for doc in ds:
            genre = doc["document_id"].split("/")[0]
            for s in doc["sentences"]:
                block = _sentence_block(
                    s["words"],
                    [pos.int2str(int(x)) for x in s["pos_tags"]],
                    [ne.int2str(int(x)) for x in s["named_entities"]],
                )
                per_genre.setdefault(genre, []).append(block)
                nsent += 1
                if limit and nsent >= limit:
                    stop = True
                    break
            if stop:
                break

        news = [b for g in NEWS for b in per_genre.get(g, [])]
        all6 = [b for g in ALL6 for b in per_genre.get(g, [])]

        for g in GENRES:
            write_bio(os.path.join(out_dir, split, f"onto.{g}.ner"), per_genre.get(g, []))
        write_bio(os.path.join(out_dir, split, "onto.news.ner"), news)
        write_bio(os.path.join(out_dir, split, "onto.all6.ner"), all6)
        write_bio(os.path.join(out_dir, split, "onto.everything.ner"), all6)
        combined.extend(all6)

        for g in GENRES:
            blk = per_genre.get(g, [])
            report.append(
                f"{split:12s} {g:4s} sents={len(blk):6d}  entities={count_entities(blk):6d}"
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
