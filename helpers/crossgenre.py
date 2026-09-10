"""Cross-genre training sets (paper Table 8 + Figure 1).

Adapted from ``SOTANER Windows/make_crossgenre_sets.py``. The paper's 4 genres:
``news`` (= bn+mz+nw), ``bc``, ``tc``, ``wb``.

Outputs under ``<out_dir>`` :
  news_train.ner / news_dev.ner          -- Table 8: train on `news` only
  <g>_test.ner  for g in the 4 genres    -- shared test files
  train_not_<g>.ner / dev_<g>.ner        -- Figure 1: train on the other 3,
                                            dev on the held-out genre g
"""
from __future__ import annotations

import os

from .bio_utils import CROSS_GENRES, NEWS, read_sentence_blocks, write_bio


def _genre_file(bio_dir, split, genre):
    return os.path.join(bio_dir, split, f"onto.{genre}.ner")


def _collect(bio_dir, split, genre):
    if genre == "news":
        out = []
        for g in NEWS:
            out += read_sentence_blocks(_genre_file(bio_dir, split, g))
        return out
    return read_sentence_blocks(_genre_file(bio_dir, split, genre))


def make_sets(bio_dir, out_dir, verbose=True):
    """Build every cross-genre set. ``bio_dir`` must contain the
    ``{train,development,test}/onto.<genre>.ner`` tree from ``conll_to_bio``."""
    os.makedirs(out_dir, exist_ok=True)

    # Table 8: news-only train / dev + the 4 shared test sets
    write_bio(os.path.join(out_dir, "news_train.ner"), _collect(bio_dir, "train", "news"))
    write_bio(os.path.join(out_dir, "news_dev.ner"), _collect(bio_dir, "development", "news"))
    for g in CROSS_GENRES:
        write_bio(os.path.join(out_dir, f"{g}_test.ner"), _collect(bio_dir, "test", g))

    # Figure 1: leave-one-genre-out
    for held in CROSS_GENRES:
        others = [g for g in CROSS_GENRES if g != held]
        train_blocks = []
        for g in others:
            train_blocks += _collect(bio_dir, "train", g)
        write_bio(os.path.join(out_dir, f"train_not_{held}.ner"), train_blocks)
        write_bio(os.path.join(out_dir, f"dev_{held}.ner"), _collect(bio_dir, "development", held))
        if verbose:
            print(f"held-out {held:4s}: train on {'+'.join(others)}  "
                  f"({len(train_blocks)} sents)  dev={held}")

    if verbose:
        print("\nwrote cross-genre sets under", out_dir)
    return out_dir
