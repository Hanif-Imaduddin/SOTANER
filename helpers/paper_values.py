"""Numbers reported in Vajjala & Balasubramaniam, LREC 2022, plus the detailed
figures from the paper's supplementary workbook ``Results-Details.xlsx``.

Every experiment table in the notebook is rendered as
``Reported on Paper | Obtained | Delta`` -- "Reported on Paper" comes from here.

Sources:
  * TABLE2_*, TABLE3_MAIN, TABLE4/5, TABLE6_*: the paper text (Tables 2-6).
  * TABLE3_ALL18: sheet ``Blackbox-DetailedPerformanceTab`` of Results-Details.xlsx
    (the "full list for all 18 types" the paper's footnote 11 points to).
  * TABLE7_FOLDS, TABLE8, FIGURE1: sheet ``TrainingEval`` of the same workbook.

All F1 values are entity-level, seqeval, on a 0-100 scale.
"""

# ---------------------------------------------------------------------------
# Table 2 -- library sanity check on the standard CoNLL-2012 test split
# ---------------------------------------------------------------------------
# "Reported on Paper" for the notebook's Table 2 = the paper's *Obtained* column
# (what the authors themselves measured with seqeval).
TABLE2_OBTAINED = {"spacy": 89.09, "stanza": 88.71, "sparknlp": 88.60}
# The library-website self-reported number (paper's *Reported* column) -- shown
# only as a side note.
TABLE2_WEBSITE = {"spacy": 90.00, "stanza": 88.80, "sparknlp": 89.97}

# ---------------------------------------------------------------------------
# Table 3 -- per entity-type F1 (off-the-shelf models, standard test)
# ---------------------------------------------------------------------------
# The 8 types actually printed in the paper (4 most frequent + 4 least frequent).
TABLE3_MAIN = {
    "DATE":     {"stanza": 86.55, "spacy": 85.51, "sparknlp": 85.54},
    "GPE":      {"stanza": 95.20, "spacy": 95.36, "sparknlp": 95.61},
    "ORG":      {"stanza": 87.44, "spacy": 90.48, "sparknlp": 87.53},
    "PERSON":   {"stanza": 93.29, "spacy": 93.51, "sparknlp": 93.11},
    "LANGUAGE": {"stanza": 60.61, "spacy": 74.42, "sparknlp": 60.60},
    "LAW":      {"stanza": 64.79, "spacy": 67.50, "sparknlp": 64.71},
    "EVENT":    {"stanza": 64.96, "spacy": 74.42, "sparknlp": 53.22},
    "PRODUCT":  {"stanza": 67.97, "spacy": 71.95, "sparknlp": 71.05},
}

# All 18 types, from Results-Details.xlsx / Blackbox-DetailedPerformanceTab (F1).
TABLE3_ALL18 = {
    "CARDINAL":    {"stanza": 85.87, "spacy": 82.29, "sparknlp": 85.54},
    "DATE":        {"stanza": 86.55, "spacy": 85.63, "sparknlp": 85.54},
    "EVENT":       {"stanza": 64.96, "spacy": 74.42, "sparknlp": 53.23},
    "FAC":         {"stanza": 73.56, "spacy": 74.71, "sparknlp": 74.42},
    "GPE":         {"stanza": 95.20, "spacy": 95.36, "sparknlp": 95.61},
    "LANGUAGE":    {"stanza": 60.61, "spacy": 74.42, "sparknlp": 60.61},
    "LAW":         {"stanza": 64.79, "spacy": 67.50, "sparknlp": 64.71},
    "LOC":         {"stanza": 75.48, "spacy": 75.94, "sparknlp": 79.21},
    "MONEY":       {"stanza": 89.03, "spacy": 89.28, "sparknlp": 87.07},
    "NORP":        {"stanza": 94.35, "spacy": 94.13, "sparknlp": 94.61},
    "ORDINAL":     {"stanza": 86.06, "spacy": 83.46, "sparknlp": 85.92},
    "ORG":         {"stanza": 87.44, "spacy": 90.47, "sparknlp": 87.53},
    "PERCENT":     {"stanza": 89.53, "spacy": 92.00, "sparknlp": 88.31},
    "PERSON":      {"stanza": 93.29, "spacy": 93.51, "sparknlp": 93.11},
    "PRODUCT":     {"stanza": 67.97, "spacy": 71.95, "sparknlp": 71.05},
    "QUANTITY":    {"stanza": 78.34, "spacy": 82.46, "sparknlp": 82.08},
    "TIME":        {"stanza": 63.16, "spacy": 68.15, "sparknlp": 66.67},
    "WORK_OF_ART": {"stanza": 61.68, "spacy": 58.86, "sparknlp": 60.12},
}

# ---------------------------------------------------------------------------
# Table 4 -- by data source ; Table 5 -- by genre (off-the-shelf, standard test)
# ---------------------------------------------------------------------------
TABLE4_SOURCE = {
    "bn": {"stanza": 91.82, "spacy": 91.64, "sparknlp": 90.93},
    "mz": {"stanza": 85.97, "spacy": 88.72, "sparknlp": 87.73},
    "nw": {"stanza": 90.87, "spacy": 86.14, "sparknlp": 90.96},
    "bc": {"stanza": 88.35, "spacy": 91.55, "sparknlp": 87.59},
    "tc": {"stanza": 76.68, "spacy": 71.16, "sparknlp": 78.38},
    "wb": {"stanza": 81.20, "spacy": 82.81, "sparknlp": 80.11},
}
TABLE5_GENRE = {
    "news": {"stanza": 90.41, "spacy": 90.79, "sparknlp": 90.47},
    "bc":   {"stanza": 88.35, "spacy": 88.72, "sparknlp": 87.59},
    "tc":   {"stanza": 76.68, "spacy": 71.16, "sparknlp": 78.37},
    "wb":   {"stanza": 81.20, "spacy": 82.81, "sparknlp": 80.11},
}

# ---------------------------------------------------------------------------
# Table 6 -- adversarial test sets. "All" = overall micro-F1, "PER"/"GPE" = that
# class's F1. Row keys: None, P1..P6.
# ---------------------------------------------------------------------------
TABLE6_ALL = {
    "None": {"stanza": 88.71, "spacy": 89.09, "sparknlp": 88.60},
    "P1":   {"stanza": 88.66, "spacy": 88.29, "sparknlp": 86.21},
    "P2":   {"stanza": 87.80, "spacy": 88.75, "sparknlp": 88.14},
    "P3":   {"stanza": 87.95, "spacy": 88.34, "sparknlp": 88.00},
    "P4":   {"stanza": 85.57, "spacy": 86.74, "sparknlp": 85.24},
    "P5":   {"stanza": 87.88, "spacy": 88.42, "sparknlp": 87.84},
    "P6":   {"stanza": 80.87, "spacy": 82.48, "sparknlp": 80.56},
}
# Per-class F1: PER for None+P1..P5, GPE for None+P6.
TABLE6_CLASS = {
    "None": {"stanza": 93.29, "spacy": 93.51, "sparknlp": 93.11},   # PER
    "P1":   {"stanza": 93.00, "spacy": 91.51, "sparknlp": 82.98},   # PER
    "P2":   {"stanza": 88.18, "spacy": 92.13, "sparknlp": 90.48},   # PER
    "P3":   {"stanza": 90.35, "spacy": 90.84, "sparknlp": 91.22},   # PER
    "P4":   {"stanza": 80.75, "spacy": 83.64, "sparknlp": 80.61},   # PER
    "P5":   {"stanza": 90.15, "spacy": 91.37, "sparknlp": 90.26},   # PER
    "P6":   {"stanza": 68.37, "spacy": 73.47, "sparknlp": 68.84},   # GPE
}
TABLE6_NONE_GPE = {"stanza": 95.20, "spacy": 95.61, "sparknlp": 95.61}
PERTURB_DEFS = {
    "P1": "PERSON -> literal word 'Dodo' (no Faker)",
    "P2": "PERSON -> Faker en_US names",
    "P3": "PERSON -> Faker en_IN names",
    "P4": "PERSON -> Faker en_TH female names",
    "P5": "PERSON -> Faker en_IN female names",
    "P6": "GPE    -> Faker en_IE place names",
}

# ---------------------------------------------------------------------------
# Table 7 -- 10 random splits, models retrained from scratch. Per-fold F1 from
# Results-Details.xlsx / TrainingEval.
# ---------------------------------------------------------------------------
TABLE7_FOLDS = {
    "stanza":   [88.00, 88.25, 88.52, 89.09, 88.61, 87.92, 88.41, 88.26, 87.77, 88.53],
    "spacy":    [90.32, 87.98, 90.79, 90.97, 90.24, 91.82, 92.19, 90.83, 90.37, 92.19],
    "sparknlp": [90.41, 89.61, 90.20, 89.86, 90.07, 90.15, 90.12, 89.90, 90.07, 90.05],
}
TABLE7_SUMMARY = {  # as printed in the paper (Table 7)
    "stanza":   {"avg": 88.34, "sdev": 0.37, "min": 87.77, "max": 89.09},
    "spacy":    {"avg": 90.77, "sdev": 1.17, "min": 87.98, "max": 92.19},
    "sparknlp": {"avg": 90.04, "sdev": 0.20, "min": 89.60, "max": 90.41},
}

# ---------------------------------------------------------------------------
# Table 8 -- train on `news` only (dev = news-dev), test on each genre.
# ---------------------------------------------------------------------------
TABLE8 = {
    "news": {"stanza": 89.18, "spacy": 82.64, "sparknlp": 89.48},
    "bc":   {"stanza": 78.55, "spacy": 65.40, "sparknlp": 78.78},
    "wb":   {"stanza": 76.04, "spacy": 62.57, "sparknlp": 75.63},
    "tc":   {"stanza": 67.19, "spacy": 51.64, "sparknlp": 63.08},
}

# ---------------------------------------------------------------------------
# Figure 1 -- leave-one-genre-out: train on 3 genres, dev on the held-out 4th,
# test on all 4. Keyed [held_out_genre][test_genre][library]. From TrainingEval
# rows 2-17. held-out genre <-> the 3 trained genres:
#   tc   <- news + bc + wb
#   bc   <- news + wb + tc
#   wb   <- news + tc + bc
#   news <- wb + tc + bc
# ---------------------------------------------------------------------------
FIGURE1 = {
    "tc": {
        "news": {"spacy": 79.61, "stanza": 86.14, "sparknlp": 89.06},
        "bc":   {"spacy": 71.39, "stanza": 77.98, "sparknlp": 82.63},
        "wb":   {"spacy": 67.71, "stanza": 76.05, "sparknlp": 76.96},
        "tc":   {"spacy": 60.92, "stanza": 55.08, "sparknlp": 70.82},
    },
    "bc": {
        "news": {"spacy": 81.49, "stanza": 87.69, "sparknlp": 88.26},
        "bc":   {"spacy": 69.26, "stanza": 75.01, "sparknlp": 78.60},
        "wb":   {"spacy": 69.67, "stanza": 77.85, "sparknlp": 78.91},
        "tc":   {"spacy": 62.08, "stanza": 73.61, "sparknlp": 72.73},
    },
    "wb": {
        "news": {"spacy": 82.32, "stanza": 87.61, "sparknlp": 88.44},
        "bc":   {"spacy": 65.96, "stanza": 81.64, "sparknlp": 81.79},
        "wb":   {"spacy": 61.46, "stanza": 75.46, "sparknlp": 77.24},
        "tc":   {"spacy": 47.11, "stanza": 70.97, "sparknlp": 71.11},
    },
    "news": {
        "news": {"spacy": 52.36, "stanza": 76.34, "sparknlp": 87.37},
        "bc":   {"spacy": 68.62, "stanza": 79.76, "sparknlp": 60.66},
        "wb":   {"spacy": 52.48, "stanza": 79.56, "sparknlp": 76.18},
        "tc":   {"spacy": 46.24, "stanza": 71.35, "sparknlp": 73.12},
    },
}

LIBRARIES = ["spacy", "stanza", "sparknlp"]
LIB_LABEL = {"spacy": "spaCy", "stanza": "Stanza", "sparknlp": "Spark NLP"}
