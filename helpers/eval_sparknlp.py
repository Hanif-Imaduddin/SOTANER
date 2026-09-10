"""Off-the-shelf Spark NLP NER evaluation (paper Tables 2-6, Spark NLP column).

Adapted from ``SOTANER Windows/evaluation_sparknlp.py``. Model: pretrained
``onto_bert_base_cased`` (NerDL) on ``bert_base_cased`` embeddings -- the TF
editions dated 2020, matching the paper's Spark NLP 3.1.2 era. Turned into a
function that returns seqeval reports as dicts.
"""
from __future__ import annotations

import os

from seqeval.metrics import classification_report

from .spark_session import load_bert, load_nerdl, start_spark


def _bio_to_spark_conll(inpath, outpath):
    """4-col tab BIO (with optional -DOCSTART- header) -> space-delimited CoNLL."""
    sentences, cur = [], []
    with open(inpath, encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if line.startswith("-DOCSTART-"):
                continue
            if line.strip() == "":
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            token, pos, _, label = parts
            cur.append((token, pos, label))
    if cur:
        sentences.append(cur)
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as fp:
        fp.write("-DOCSTART- -X- -X- -O-\n\n")
        for sent in sentences:
            for token, pos, label in sent:
                fp.write("{} {} {} {}\n".format(token, pos, pos, label))
            fp.write("\n")
    return len(sentences)


def build_pipeline(bert=None, ner=None):
    from pyspark.ml import Pipeline

    if bert is None:
        bert = load_bert("bert_base_cased")
    bert = bert.setInputCols(["sentence", "token"]).setOutputCol("bert") \
        .setCaseSensitive(True).setMaxSentenceLength(512)
    if ner is None:
        ner = load_nerdl("onto_bert_base_cased")
    ner = ner.setInputCols(["sentence", "token", "bert"]).setOutputCol("ner")
    return Pipeline(stages=[bert, ner])


def evaluate_bio_sparknlp(paths, spark=None, pipeline=None, tmp_dir="data/spark-format",
                          results_dir=None, tag="A", digits=6):
    """Evaluate each BIO file with the pretrained Spark NLP OntoNotes model.
    Returns ``{stem: {"dict": report_dict(0-100), "text": report_str}}``."""
    from sparknlp.training import CoNLL

    if spark is None:
        spark = start_spark(memory="8g")
    if pipeline is None:
        pipeline = build_pipeline()

    os.makedirs(tmp_dir, exist_ok=True)
    out = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        conll_path = os.path.join(tmp_dir, stem + ".conll")
        _bio_to_spark_conll(p, conll_path)

        data = CoNLL().readDataset(spark, conll_path)
        model = pipeline.fit(data)
        rows = model.transform(data).select("sentence", "token", "label", "ner").collect()
        bad = {i for i, r in enumerate(rows) if len(r["label"]) != len(r["ner"])}
        rows = [r for i, r in enumerate(rows) if i not in bad]
        y_true = [[t["result"] for t in r["label"]] for r in rows]
        y_pred = [[t["result"] for t in r["ner"]] for r in rows]

        d = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
        for k, v in d.items():
            if isinstance(v, dict):
                for m in ("precision", "recall", "f1-score"):
                    if v.get(m) is not None:
                        v[m] *= 100.0
        txt = classification_report(y_true, y_pred, digits=digits, zero_division=1)
        out[stem] = {"dict": d, "text": txt, "dropped": len(bad)}
        print(stem, f"micro-F1={d['micro avg']['f1-score']:.2f}  (dropped {len(bad)} rows)")

        if results_dir:
            od = os.path.join(results_dir, tag)
            os.makedirs(od, exist_ok=True)
            with open(os.path.join(od, "sparknlp_" + stem + ".txt"), "w", encoding="utf-8") as fh:
                fh.write("====== %s ======\n" % stem)
                fh.write(txt + "\n")
    return out
