"""Helper package for the notebook-based SOTANER reproduction.

Import surface used by ``SOTANER_Reproduction.ipynb``. Every module is adapted
(minimally) from the proven scripts in ``../SOTANER Windows/``; only
``conll_to_bio`` (local CoNLL-2012 v4 reader) and ``paper_values`` are new.
"""
from . import (  # noqa: F401
    bio_utils,
    conll_to_bio,
    crossgenre,
    eval_blackbox,
    hf_to_bio,
    paper_values,
    perturb,
    report_utils,
    splits,
)

from .bio_utils import (  # noqa: F401
    ALL6,
    CROSS_GENRES,
    ENTITY_TYPES,
    GENRES,
    NEWS,
    TABLE3_TYPES,
    count_entities,
    entity_type_counts,
    read_bio_file,
    read_sentence_blocks,
    write_bio,
)
from .conll_to_bio import build_all  # noqa: F401
from .hf_to_bio import build_all as build_all_hf  # noqa: F401
from .crossgenre import make_sets  # noqa: F401
from .eval_blackbox import evaluate_bio, load_spacy, load_stanza, run_blackbox  # noqa: F401
from .perturb import make_all_perturbations, perturb_file  # noqa: F401
from .report_utils import (  # noqa: F401
    comparison_table,
    export_all,
    f1_of,
    micro_f1_from_text,
    summarise_folds,
)
from .splits import make_kfold_splits  # noqa: F401

__all__ = [
    "bio_utils", "conll_to_bio", "hf_to_bio", "crossgenre", "eval_blackbox",
    "paper_values", "perturb", "report_utils", "splits",
    "GENRES", "NEWS", "ALL6", "CROSS_GENRES", "ENTITY_TYPES", "TABLE3_TYPES",
    "read_bio_file", "read_sentence_blocks", "write_bio", "count_entities",
    "entity_type_counts", "build_all", "build_all_hf", "make_sets", "evaluate_bio",
    "load_spacy", "load_stanza", "run_blackbox", "make_all_perturbations", "perturb_file",
    "comparison_table", "export_all", "f1_of", "summarise_folds", "make_kfold_splits",
]
