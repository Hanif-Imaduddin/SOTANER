"""Off-the-shelf spaCy + Stanza NER evaluation (paper Tables 2-6).

Adapted from ``SOTANER Windows/evaluation_spacy_stanza.py`` + ``run_blackbox.py``.
Turned into importable functions that return seqeval reports as dicts (micro +
per-type in one pass) so the notebook can build every table from one eval run.

Notes:
  * spaCy device: ``load_spacy(gpu=None)`` honours env ``SOTANER_SPACY_GPU`` (the
    local Windows default is CPU -- benchmarked slower than a 4 GB card for
    ``en_core_web_trf``); ``gpu=True`` calls ``spacy.prefer_gpu()`` (graceful
    fallback) -- use it on Colab.
  * Stanza is driven directly through ``stanza.Pipeline`` (the ``spacy-stanza``
    wrapper pins ``stanza<1.7``, broken on torch >= 2.6). Same ``en`` OntoNotes
    NER model, so predictions are equivalent. It uses the GPU automatically when
    one is present (``use_gpu=True``).
  * Both pipelines get a whitespace tokenizer so model tokens == gold tokens.
"""
from __future__ import annotations

import os

from seqeval.metrics import classification_report

from .bio_utils import bioes_to_bio, read_bio_file

_SPACY_GPU_ENV = os.environ.get("SOTANER_SPACY_GPU", "0") == "1"


def load_spacy(gpu=None):
    """Load the best available spaCy English NER pipeline with a whitespace tokenizer.

    ``gpu``: ``None`` -> respect env ``SOTANER_SPACY_GPU``; ``True``/``False`` -> force.
    """
    import spacy
    from spacy.tokenizer import Tokenizer

    want_gpu = _SPACY_GPU_ENV if gpu is None else bool(gpu)
    if want_gpu:
        ok = spacy.prefer_gpu()
        print("spaCy: GPU" if ok else "spaCy: GPU requested but unavailable -> CPU")
    else:
        print("spaCy: CPU")

    for name in ("en_core_web_trf", "en_core_web_lg"):
        try:
            nlp = spacy.load(name)
            nlp.tokenizer = Tokenizer(nlp.vocab)
            print("spaCy model:", name)
            return nlp
        except Exception as e:  # noqa: BLE001
            print("  could not load %s: %s" % (name, e))
    raise SystemExit("no usable spaCy NER model (run: python -m spacy download en_core_web_trf)")


def load_stanza(use_gpu=True):
    """Return a callable: pretokenised sentence string -> list[BIO tag]."""
    import stanza

    pipe = stanza.Pipeline(lang="en", processors="tokenize,ner",
                           tokenize_pretokenized=True, use_gpu=use_gpu, verbose=False)
    print("Stanza NER pipeline loaded (use_gpu=%s)" % use_gpu)

    def tagger(text):
        doc = pipe(text)
        out = []
        for sent in doc.sentences:
            for tok in sent.tokens:
                out.append(bioes_to_bio(getattr(tok, "ner", "O")))
        return out

    return tagger


def _spacy_tags(doc, n_gold):
    tags = []
    for token in doc:
        if token.ent_iob_ and token.ent_type_:
            tags.append(token.ent_iob_ + "-" + token.ent_type_)
        else:
            tags.append(token.ent_iob_ or "O")
    if len(tags) != n_gold:
        tags = (tags + ["O"] * n_gold)[:n_gold]
    return tags


def evaluate_bio(path, nlp=None, stanza_tagger=None, models=("spacy", "stanza"),
                 batch_size=16, digits=4):
    """Evaluate a BIO test file. Returns
    ``{model: {"dict": report_dict, "text": report_str, "n_sent": int}}``.
    ``report_dict`` is ``seqeval.classification_report(output_dict=True)`` scaled
    to 0-100 (so ``dict["micro avg"]["f1-score"]`` is a percentage)."""
    gold_sen, gold_ner = read_bio_file(path)
    texts = [" ".join(s) for s in gold_sen]
    out = {}

    if "spacy" in models:
        if nlp is None:
            nlp = load_spacy()
        preds = []
        for sen, doc in zip(gold_sen, nlp.pipe(texts, batch_size=batch_size)):
            preds.append(_spacy_tags(doc, len(sen)))
        out["spacy"] = _report(gold_ner, preds, digits, len(gold_sen))

    if "stanza" in models:
        if stanza_tagger is None:
            stanza_tagger = load_stanza()
        preds = []
        for sen, text in zip(gold_sen, texts):
            t = stanza_tagger(text)
            if len(t) != len(sen):
                t = (t + ["O"] * len(sen))[:len(sen)]
            preds.append(t)
        out["stanza"] = _report(gold_ner, preds, digits, len(gold_sen))

    return out


def _report(gold, pred, digits, n_sent):
    d = classification_report(gold, pred, output_dict=True, zero_division=0)
    _scale(d)
    txt = classification_report(gold, pred, digits=digits, zero_division=0)
    return {"dict": d, "text": txt, "n_sent": n_sent}


def _scale(d):
    for k, v in d.items():
        if isinstance(v, dict):
            for m in ("precision", "recall", "f1-score"):
                if m in v and v[m] is not None:
                    v[m] = v[m] * 100.0


def run_blackbox(paths, results_dir, tag, nlp=None, stanza_tagger=None,
                 models=("spacy", "stanza")):
    """Load the models once, evaluate every path, tee each text report to
    ``<results_dir>/<tag>/<stem>.txt``. Returns ``{stem: evaluate_bio(...) result}``."""
    outdir = os.path.join(results_dir, tag)
    os.makedirs(outdir, exist_ok=True)
    if nlp is None and "spacy" in models:
        nlp = load_spacy()
    if stanza_tagger is None and "stanza" in models:
        stanza_tagger = load_stanza()

    results = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        res = evaluate_bio(p, nlp=nlp, stanza_tagger=stanza_tagger, models=models)
        results[stem] = res
        with open(os.path.join(outdir, stem + ".txt"), "w", encoding="utf-8") as fh:
            for m in models:
                fh.write("Classification report for %s NER:\n" % m.capitalize())
                fh.write(res[m]["text"] + "\n\n")
        f1s = {m: res[m]["dict"]["micro avg"]["f1-score"] for m in models}
        print(stem, " ".join(f"{m}={f1s[m]:.2f}" for m in models))
    return results, nlp, stanza_tagger
