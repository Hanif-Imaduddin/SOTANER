"""10 random train/dev/test splits for the retraining experiment (paper Table 7).

Adapted from ``SOTANER Windows/generate_splits.py``. The paper says the splits
were "generated at the sentence level on the entire corpus, keeping the
proportions of the original train/dev/test splits"; we implement that as a seeded
``KFold(10)`` for the test fold (~10%) plus a seeded ``ShuffleSplit`` carving
~10% dev out of the remaining train -- reproducible, and close to the standard
~81/9/10 proportions.

Fixes carried over from the Windows port: ``!= ""`` (not ``is not``), seeded
splitters, and mapping ``ShuffleSplit`` positional indices back through
``temptrain`` before indexing the sentence store (the upstream bug let dev
overlap the test fold).
"""
from __future__ import annotations

import os

from sklearn.model_selection import KFold, ShuffleSplit

from .bio_utils import DOCSTART, read_sentence_blocks


def make_kfold_splits(combined_bio, out_dir, n_splits=10, dev_frac=0.1, seed=42, verbose=True):
    """Read one combined BIO file, write ``fold{k}_{train,dev,test}.ner`` for
    k = 1..n_splits into ``out_dir``. Returns the list of fold sizes."""
    os.makedirs(out_dir, exist_ok=True)
    blocks = read_sentence_blocks(combined_bio)
    n = len(blocks)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ss = ShuffleSplit(n_splits=1, test_size=dev_frac, random_state=seed)

    sizes = []
    for fold, (trainval_idx, test_idx) in enumerate(kf.split(range(n)), start=1):
        tr_pos, dv_pos = next(ss.split(trainval_idx))
        train_idx = [trainval_idx[i] for i in tr_pos]
        dev_idx = [trainval_idx[i] for i in dv_pos]

        for part, idxs in (("train", train_idx), ("dev", dev_idx), ("test", list(test_idx))):
            path = os.path.join(out_dir, f"fold{fold}_{part}.ner")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(DOCSTART + "\n\n")
                for i in idxs:
                    fh.write(blocks[i])
                    fh.write("\n\n")
        sizes.append((fold, len(train_idx), len(dev_idx), len(test_idx)))
        if verbose:
            print(f"fold{fold}: train={len(train_idx)}  dev={len(dev_idx)}  test={len(test_idx)}")
    return sizes
