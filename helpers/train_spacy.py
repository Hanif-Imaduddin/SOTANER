"""spaCy NER training/eval wrappers for the retraining experiments
(paper Tables 7, 8, Figure 1).

Adapted from ``SOTANER Windows/spacy-stanza-scripts/{prep,train,eval}_ner_spacy.py``
(the Windows Python ports of the upstream bash loops). The training config is the
upstream ``config.cfg`` = a **CNN tok2vec transition-based parser** (NOT a
transformer, despite the ``trf`` naming upstream used) -- this is what the paper's
released config trains.

All steps shell out to ``python -m spacy`` via ``sys.executable`` so the notebook
never has to manage spaCy's training loop in-process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(os.path.dirname(_HERE), "config.cfg")


def _run(cmd):
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def prep(bio_path, out_dir):
    """BIO file -> ``<out_dir>/<stem>.spacy``. Returns that path."""
    os.makedirs(out_dir, exist_ok=True)
    _run([sys.executable, "-m", "spacy", "convert", bio_path, out_dir, "-c", "ner"])
    stem = os.path.splitext(os.path.basename(bio_path))[0]
    return os.path.join(out_dir, stem + ".spacy")


def train(train_spacy, dev_spacy, out_model_dir, max_steps=20000, gpu_id=None,
          config=CONFIG):
    """Train an NER model; returns the path to ``model-best``."""
    os.makedirs(out_model_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "spacy", "train", config,
           "--paths.train", train_spacy, "--paths.dev", dev_spacy,
           "--output", out_model_dir,
           "--training.max_steps", str(max_steps)]
    if gpu_id is not None:
        cmd += ["--gpu-id", str(gpu_id)]
    _run(cmd)
    return os.path.join(out_model_dir, "model-best")


def evaluate(model_best, test_spacy, out_json=None):
    """Evaluate a trained model on a test ``.spacy``. Returns entity-level F1 (0-100)."""
    if out_json is None:
        out_json = os.path.join(os.path.dirname(model_best), "eval.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    _run([sys.executable, "-m", "spacy", "evaluate", model_best, test_spacy,
          "--output", out_json])
    with open(out_json, encoding="utf-8") as fh:
        j = json.load(fh)
    f1 = j.get("ents_f")
    if f1 is None:
        f1 = j.get("ner", {}).get("ents_f", 0.0)
    return f1 * 100.0
