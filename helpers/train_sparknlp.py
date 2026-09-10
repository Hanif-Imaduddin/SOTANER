"""Spark NLP NerDL training/eval on one split (paper Tables 7, 8, Figure 1).

Adapted from ``SOTANER Windows/train_sparknlp.py``. Architecture: BERT
(``bert_base_cased``) features -> ``NerDLApproach`` (CharCNN-BiLSTM-CRF). Spark
NLP has no custom-validation hook, so training carves ``setValidationSplit(0.1)``
from the train set (the paper does the same); the supplied ``dev_ner`` is kept in
the signatures only for parity.

Split into ``fit_nerdl`` (train once) + ``score_nerdl`` (evaluate on any test
file) so the cross-genre experiments can reuse one trained model across the 4
genre test sets. ``run_fold`` is the train+score convenience wrapper used by
Table 7.
"""
from __future__ import annotations

import os
import shutil

from seqeval.metrics import classification_report

from .eval_sparknlp import _bio_to_spark_conll
from .spark_session import load_bert, start_spark


def _bert_stage(bert=None):
    if bert is None:
        bert = load_bert("bert_base_cased")
    return bert.setInputCols(["sentence", "token"]).setOutputCol("bert").setCaseSensitive(True)


def _ready_parquet(spark, bert, bio_path, tmp_dir):
    """BIO -> spark-CoNLL -> BERT-embedded parquet. Returns the parquet path."""
    from sparknlp.training import CoNLL

    os.makedirs(tmp_dir, exist_ok=True)
    stem = os.path.basename(bio_path)
    conll = os.path.join(tmp_dir, stem + ".conll")
    pq = os.path.join(tmp_dir, stem + ".pq")
    _bio_to_spark_conll(bio_path, conll)
    data = CoNLL().readDataset(spark, conll)
    bert.transform(data).write.mode("Overwrite").parquet(pq)
    return pq


def fit_nerdl(train_ner, tmp_dir, spark=None, bert=None, max_epochs=1,
              batch_size=8, seed=0):
    """Train a NerDL model on one BIO file. Returns ``(model, spark, bert)``."""
    from pyspark.ml import Pipeline
    from sparknlp.annotator import NerDLApproach

    if spark is None:
        spark = start_spark(memory="8g")
    bert = _bert_stage(bert)
    train_pq = _ready_parquet(spark, bert, train_ner, tmp_dir)

    ner = (NerDLApproach()
           .setInputCols(["sentence", "token", "bert"]).setLabelColumn("label")
           .setOutputCol("ner").setMaxEpochs(max_epochs).setBatchSize(batch_size)
           .setEnableMemoryOptimizer(True).setRandomSeed(seed).setVerbose(1)
           .setValidationSplit(0.1))
    model = Pipeline(stages=[ner]).fit(spark.read.parquet(train_pq))
    shutil.rmtree(train_pq, ignore_errors=True)
    return model, spark, bert


def score_nerdl(model, test_ner, tmp_dir, spark, bert, digits=6, clean_tmp=True):
    """Evaluate a fitted NerDL model on one BIO file with seqeval.
    Returns ``{"dict": report_dict(0-100), "text": str, "f1": float}``."""
    test_pq = _ready_parquet(spark, bert, test_ner, tmp_dir)
    rows = model.transform(spark.read.parquet(test_pq)) \
        .select("sentence", "token", "label", "ner").collect()
    if clean_tmp:
        shutil.rmtree(test_pq, ignore_errors=True)

    bad = {i for i, r in enumerate(rows) if len(r["label"]) != len(r["ner"])}
    rows = [r for i, r in enumerate(rows) if i not in bad]
    y_true = [[t["result"] for t in r["label"]] for r in rows]
    y_pred = [[t["result"] for t in r["ner"]] for r in rows]

    d = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
    for _k, v in d.items():
        if isinstance(v, dict):
            for m in ("precision", "recall", "f1-score"):
                if v.get(m) is not None:
                    v[m] *= 100.0
    txt = classification_report(y_true, y_pred, digits=digits, zero_division=1)
    return {"dict": d, "text": txt, "f1": d["micro avg"]["f1-score"], "dropped": len(bad)}


def run_fold(train_ner, dev_ner, test_ner, tmp_dir, spark=None, bert=None,
             max_epochs=1, batch_size=8, seed=0, digits=6, clean_tmp=True):
    """Train on ``train_ner`` and evaluate on ``test_ner`` (Table 7)."""
    model, spark, bert = fit_nerdl(train_ner, tmp_dir, spark=spark, bert=bert,
                                   max_epochs=max_epochs, batch_size=batch_size, seed=seed)
    return score_nerdl(model, test_ner, tmp_dir, spark, bert, digits=digits, clean_tmp=clean_tmp)
