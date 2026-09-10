"""Shared BIO helpers for the notebook-based SOTANER reproduction.

The 4-column tab-separated BIO format every step of this pipeline uses:

    token \\t POS \\t constituency \\t NER-tag

one token per line, a blank line between sentences. Files may carry a leading
``-DOCSTART- -X- -X- O`` line + blank line (some downstream Spark/split code
slices off the first line assuming a header); the readers here skip it.
"""
from __future__ import annotations

import os
from collections import Counter

# OntoNotes genres. ``pt`` (Old/New Testament) carries no named-entity layer and
# is excluded from every NER experiment in the paper -- kept in the list only so
# the converter can report it.
GENRES = ["bc", "bn", "mz", "nw", "tc", "wb"]

# The paper regroups the six NER sources into four genres.
NEWS = ["bn", "mz", "nw"]                     # paper's "News" genre
ALL6 = ["bn", "mz", "nw", "bc", "tc", "wb"]   # standard NER test split (no pt)
CROSS_GENRES = ["news", "bc", "tc", "wb"]     # the paper's 4 cross-genre buckets

# 18 OntoNotes entity types (order = the supplementary table in Results-Details.xlsx).
ENTITY_TYPES = [
    "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY",
    "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME",
    "WORK_OF_ART",
]

# The 8 types the paper prints in Table 3 (4 most frequent + 4 least frequent).
TABLE3_TYPES = ["DATE", "GPE", "ORG", "PERSON", "LANGUAGE", "LAW", "EVENT", "PRODUCT"]

DOCSTART = "-DOCSTART- -X- -X- O"


def read_bio_file(path, numcols=4):
    """Read a BIO file -> (sentences, netags): two parallel lists of token lists.

    ``numcols`` switches 3- vs 4-column input (token is always column 0, the NER
    tag is always the last column).
    """
    sentences, netags = [], []
    cur_toks, cur_tags = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("-DOCSTART-"):
                continue
            if line.strip() == "":
                if cur_toks:
                    sentences.append(cur_toks)
                    netags.append(cur_tags)
                    cur_toks, cur_tags = [], []
                continue
            parts = line.rstrip("\n").split("\t")
            cur_toks.append(parts[0])
            cur_tags.append(parts[numcols - 1])
    if cur_toks:
        sentences.append(cur_toks)
        netags.append(cur_tags)
    return sentences, netags


def read_sentence_blocks(path):
    """Return the file as a list of sentence blocks (each a ``\\n``-joined str),
    skipping the ``-DOCSTART-`` header. Used for split / cross-genre assembly."""
    blocks, cur = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("-DOCSTART-"):
                continue
            if line.strip() == "":
                if cur:
                    blocks.append("\n".join(cur))
                    cur = []
                continue
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def write_bio(path, blocks, header=True):
    """Write sentence blocks (list[str]) to ``path`` with a blank line between
    sentences and, by default, a ``-DOCSTART-`` header."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if header:
            fh.write(DOCSTART + "\n\n")
        for b in blocks:
            fh.write(b.rstrip("\n"))
            fh.write("\n\n")


def count_entities(blocks_or_tags):
    """Count entity spans (``B-`` prefixes). Accepts a list of sentence-block
    strings or a list of tag lists."""
    n = 0
    if blocks_or_tags and isinstance(blocks_or_tags[0], str):
        for b in blocks_or_tags:
            for line in b.split("\n"):
                if line.strip():
                    tag = line.rsplit("\t", 1)[-1]
                    if tag.startswith("B-"):
                        n += 1
    else:
        for tags in blocks_or_tags:
            n += sum(1 for t in tags if t.startswith("B-"))
    return n


def entity_type_counts(netags):
    """Counter of entity-type -> span count over a list of tag lists."""
    c = Counter()
    for tags in netags:
        for t in tags:
            if t.startswith("B-"):
                c[t[2:]] += 1
    return c


def bioes_to_bio(tag):
    """Normalise a BIOES / IOB tag to IOB2 (B-/I-/O). Stanza emits BIOES."""
    if not tag or tag == "O":
        return "O"
    prefix, _, etype = tag.partition("-")
    if prefix in ("B", "S"):
        return "B-" + etype
    return "I-" + etype  # I / E
