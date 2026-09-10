"""Turn seqeval reports + paper values into the notebook's comparison tables and
export them.

Every table has, per library, three columns:
    <lib> Reported on Paper | <lib> Obtained | <lib> Delta
where ``Delta = Obtained - Reported on Paper`` (positive = we scored higher).
"""
from __future__ import annotations

import os
import re

import pandas as pd

from .paper_values import LIB_LABEL, LIBRARIES

_MICRO_RE = re.compile(r"micro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)")


def micro_f1_from_text(text):
    """Pull the micro-avg F1 (0-100) out of a saved seqeval text report, or None.
    Handles both fraction (``0.8824``) and percentage (``88.24``) reports.
    Used by the Colab notebook's RESUME checkpoints."""
    m = _MICRO_RE.search(text or "")
    if not m:
        return None
    v = float(m.group(1))
    return v * 100.0 if v <= 1.0 else v


def f1_of(report, label="micro avg"):
    """F1 (0-100) for a label from an ``evaluate_bio`` / ``predict_and_score``
    result (which wraps ``{"dict": seqeval_output_dict}``) or a raw report dict."""
    d = report.get("dict", report) if isinstance(report, dict) else report
    row = d.get(label)
    if row is None:
        return float("nan")
    return row["f1-score"]


def _round(x, n=2):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return x


def comparison_table(index_name, row_keys, obtained, paper, libs=LIBRARIES,
                     row_labels=None):
    """Build a Reported/Obtained/Delta DataFrame.

    obtained, paper : {row_key: {lib: value}}   (values on a 0-100 scale)
    row_keys        : ordered list of row keys
    row_labels      : optional {row_key: pretty label}
    """
    cols = {}
    data_index = [row_labels.get(k, k) if row_labels else k for k in row_keys]
    for lib in libs:
        L = LIB_LABEL[lib]
        rep, obt, dlt = [], [], []
        for k in row_keys:
            p = paper.get(k, {}).get(lib)
            o = obtained.get(k, {}).get(lib)
            rep.append(_round(p))
            obt.append(_round(o))
            dlt.append(_round(o - p) if (p is not None and o is not None) else None)
        cols[f"{L} Reported"] = rep
        cols[f"{L} Obtained"] = obt
        cols[f"{L} Delta"] = dlt
    df = pd.DataFrame(cols, index=pd.Index(data_index, name=index_name))
    return df


def export_all(tables, out_dir, xlsx_name="SOTANER_reproduction_tables.xlsx"):
    """tables: ordered {sheet_name: DataFrame}. Writes one .xlsx (a sheet per
    table) + one .csv per table into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    xlsx_path = os.path.join(out_dir, xlsx_name)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xl:
        for name, df in tables.items():
            sheet = name[:31]
            df.to_excel(xl, sheet_name=sheet)
            df.to_csv(os.path.join(out_dir, name.replace(" ", "_") + ".csv"))
    print("wrote", xlsx_path, "and", len(tables), "CSVs into", out_dir)
    return xlsx_path


def summarise_folds(fold_f1):
    """fold_f1: {lib: [f1, ...]} -> DataFrame avg/sdev/min/max (pandas sample std)."""
    import numpy as np

    rows = {}
    for lib, vals in fold_f1.items():
        a = np.asarray(vals, dtype=float)
        rows[LIB_LABEL.get(lib, lib)] = {
            "n": len(a),
            "avg": _round(a.mean()),
            "sdev": _round(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": _round(a.min()),
            "max": _round(a.max()),
        }
    return pd.DataFrame(rows).T
