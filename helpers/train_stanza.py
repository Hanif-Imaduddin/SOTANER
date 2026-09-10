"""Stanza NER training/eval wrappers (paper Tables 7, 8, Figure 1).

Adapted from ``SOTANER Windows/spacy-stanza-scripts/{prep,train,eval}_ner_stanza.py``.
Uses ``python -m stanza.models.ner_tagger`` directly (the upstream bash loops'
approach) with the OntoNotes wordvec pretrain that ships with
``stanza.download('en')``. The upstream ``prepare_ner_data.py`` module moved in
stanza 1.14, so the BIO -> Stanza-JSON conversion is inlined.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

from seqeval.metrics import classification_report

from .bio_utils import bioes_to_bio

_PRETRAIN_GLOBS = [
    "~/stanza_resources/en/pretrain/*.pt",
    "~/AppData/Local/StanfordNLP/stanza/Cache/*/resources/en/pretrain/*.pt",
]


def find_wordvec_pretrain():
    env = os.environ.get("SOTANER_WORDVEC")
    if env and os.path.exists(env):
        return env
    hits = []
    for g in _PRETRAIN_GLOBS:
        hits += glob.glob(os.path.expanduser(g))
    if not hits:
        raise SystemExit("no Stanza wordvec pretrain .pt found; run: "
                         "python -c \"import stanza; stanza.download('en')\"")
    # prefer 'combined' (paper) then anything else
    hits.sort(key=lambda p: (0 if "combined" in os.path.basename(p) else 1, p))
    return hits[0]


def bio_to_json(bio_path, json_path):
    """4-col tab BIO -> Stanza NER JSON ``[[{"text","ner"}, ...], ...]``."""
    sents, cur = [], []
    with open(bio_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("-DOCSTART-"):
                continue
            if line.strip() == "":
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            cur.append({"text": parts[0], "ner": parts[3]})
    if cur:
        sents.append(cur)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(sents, fh)
    return len(sents)


def _run(cmd):
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def train(train_json, dev_json, shorthand, save_dir, save_name,
          max_steps=5000, use_gpu=False, batch_size=None, wordvec=None):
    """Train a Stanza NER model; returns the saved model path.

    ``shorthand`` must be ``<lang>_<name>`` (e.g. ``en_fold1``) -- stanza 1.14's
    ``ner_tagger`` reads the language from its prefix (there is no ``--lang``).
    ``use_gpu=True`` runs on CUDA when available; ``batch_size`` overrides the
    default (raise it on GPU for throughput).
    """
    os.makedirs(save_dir, exist_ok=True)
    wordvec = wordvec or find_wordvec_pretrain()
    cmd = [sys.executable, "-m", "stanza.models.ner_tagger",
           "--wordvec_pretrain_file", wordvec,
           "--train_file", train_json, "--eval_file", dev_json,
           "--shorthand", shorthand, "--mode", "train",
           "--save_name", save_name, "--save_dir", save_dir,
           "--max_steps", str(max_steps)]
    if batch_size:
        cmd += ["--batch_size", str(batch_size)]
    if not use_gpu:
        cmd.append("--cpu")
    _run(cmd)
    return os.path.join(save_dir, save_name)


def predict_and_score(model_path, test_json, shorthand, out_pred=None,
                      use_gpu=False, wordvec=None, digits=4):
    """Predict with a trained model and score against gold with seqeval.
    Returns ``{"dict": report_dict(0-100), "text": str, "f1": float}``.

    stanza 1.14's ``ner_tagger --mode predict --eval_output_file`` writes a
    3-column TSV ``token \\t gold \\t pred`` in the BIOES scheme; we convert both
    columns to IOB2 and score with seqeval (consistent with the rest of the
    notebook).
    """
    wordvec = wordvec or find_wordvec_pretrain()
    save_dir = os.path.dirname(model_path)
    save_name = os.path.basename(model_path)
    if out_pred is None:
        out_pred = os.path.splitext(test_json)[0] + "_pred.tsv"
    cmd = [sys.executable, "-m", "stanza.models.ner_tagger",
           "--wordvec_pretrain_file", wordvec,
           "--eval_file", test_json, "--shorthand", shorthand, "--mode", "predict",
           "--save_name", save_name, "--save_dir", save_dir,
           "--eval_output_file", out_pred]
    if not use_gpu:
        cmd.append("--cpu")
    _run(cmd)

    y_true, y_pred = [], []
    gcur, pcur = [], []
    with open(out_pred, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if gcur:
                    y_true.append(gcur); y_pred.append(pcur); gcur, pcur = [], []
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            gcur.append(bioes_to_bio(cols[1]))
            pcur.append(bioes_to_bio(cols[2]))
    if gcur:
        y_true.append(gcur); y_pred.append(pcur)

    d = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for k, v in d.items():
        if isinstance(v, dict):
            for m in ("precision", "recall", "f1-score"):
                if v.get(m) is not None:
                    v[m] *= 100.0
    txt = classification_report(y_true, y_pred, digits=digits, zero_division=0)
    return {"dict": d, "text": txt, "f1": d["micro avg"]["f1-score"]}
