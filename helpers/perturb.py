"""Adversarial test-set generation (paper Section 3.2 / Table 6).

Rewrite the *surface tokens* of a chosen entity type with Faker-generated values,
leaving the sentence context and the BIO tags untouched. Adapted from
``SOTANER Windows/input_perturbances_faker.py`` + ``make_perturbations.py`` --
turned into plain functions, seeded for reproducibility.

The 6 perturbations in the paper:
  P1  PERSON -> the literal word "Dodo"            (no Faker; memorisation probe)
  P2  PERSON -> Faker en_US names
  P3  PERSON -> Faker en_IN names
  P4  PERSON -> Faker en_TH female names
  P5  PERSON -> Faker en_IN female names
  P6  GPE    -> Faker en_IE place names
"""
from __future__ import annotations

import os
import random

from faker import Faker

# (perturb id, entity category, mode, Faker locale)
JOBS = [
    ("perturb1", "PERSON", "literal:Dodo", None),
    ("perturb2", "PERSON", "name", "en_US"),
    ("perturb3", "PERSON", "name", "en_IN"),
    ("perturb4", "PERSON", "name_female", "en_TH"),
    ("perturb5", "PERSON", "name_female", "en_IN"),
    ("perturb6", "GPE", "gpe", "en_IE"),
]


def perturb_file(inpath, outpath, category, mode, locale=None, seed=0):
    """Rewrite tokens tagged ``B-<category>`` / ``I-<category>`` in a 4-col BIO file.

    mode:
      ``literal:<word>`` -> every entity token becomes ``<word>``
      ``name``           -> B- = first_name(),         I- = last_name()
      ``name_female``     -> B- = first_name_female(),  I- = last_name_female()
      ``gpe``            -> B- = first token of city()/country(), I- = street_suffix()
    """
    fake = Faker(locale) if locale else Faker()
    Faker.seed(seed)
    random.seed(seed)

    b_tag, i_tag = "B-" + category, "I-" + category
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    n_changed = 0

    with open(inpath, encoding="utf-8") as fh, open(outpath, "w", encoding="utf-8") as fw:
        for line in fh:
            raw = line.rstrip("\n")
            if raw.startswith("-DOCSTART-") or raw.strip() == "":
                fw.write(raw + "\n")
                continue
            parts = raw.split("\t")
            if len(parts) == 4 and parts[3] in (b_tag, i_tag):
                is_b = parts[3] == b_tag
                if mode.startswith("literal:"):
                    parts[0] = mode.split(":", 1)[1]
                elif mode == "name":
                    parts[0] = (fake.first_name() if is_b else fake.last_name()).split()[0]
                elif mode == "name_female":
                    parts[0] = (fake.first_name_female() if is_b else fake.last_name_female()).split()[0]
                elif mode == "gpe":
                    if is_b:
                        parts[0] = random.choice([fake.city(), fake.country()]).split()[0]
                    else:
                        parts[0] = fake.street_suffix().split()[0]
                n_changed += 1
            fw.write("\t".join(parts) + "\n")
    return n_changed


def make_all_perturbations(src_bio, out_dir, seed=0, verbose=True):
    """Build all 6 perturbed test sets from ``src_bio`` (usually
    ``data/bio/test/onto.all6.ner``). Returns {perturb_id: output_path}."""
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for pid, cat, mode, loc in JOBS:
        dst = os.path.join(out_dir, f"onto.all.test.{pid}.ner")
        n = perturb_file(src_bio, dst, category=cat, mode=mode, locale=loc, seed=seed)
        out[pid] = dst
        if verbose:
            print(f"{pid}: {cat:6s} {mode:14s} locale={loc or '-':6s} -> {n} tokens rewritten  ({dst})")
    return out
