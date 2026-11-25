#!/usr/bin/env python
"""02 - Prepare frozen, reproducible datasets from ADE-Corpus V2, BC5CDR and the Chilean Waiting List (CWLC).

Stage 1 (data), step 2.  Script 01 audited the raw corpora; this script turns them into the
"processed" datasets that every later stage (exploration, modelling, validation) will read.
It never modifies anything under data/raw.

What it does, and why
---------------------
ADE-Corpus V2
    * One unit per (PMID, sentence).  Using one row per *relation* (23,516 rows) instead would
      duplicate sentences and leak them across splits.
    * Units are connected into groups (same PMID *or* same normalised text) and whole groups are
      assigned to train/val/test, so neither a PMID nor a sentence text can cross splits.
    * The split is seeded, balances size and positive prevalence, and is verified (leakage = 0).
    * Output: BIO tags per entity type (DRUG, EFFECT) plus a merged layer, and the sentence label
      (1 = contains at least one drug-adverse-effect relation).

BC5CDR
    * Official train/dev/test splits are kept (they are PMID-disjoint, verified in script 01).
    * Output: one record per abstract (title + " " + abstract) with character spans (the source of
      truth) and BIO tags on a simple regex tokenisation.  Splitting into <=512 WordPiece windows
      is deliberately left to the modelling stage because it depends on the tokenizer.

CWLC (Chilean Waiting List)
    * Offsets: brat offsets are verified per document against the raw text.  Most documents match
      when line breaks are kept as in the file ("raw"); a few match only when CRLF is treated as
      one character ("universal_newlines").  The reading that matches is chosen per document.
    * The seven official CoNLL layers (one flat BIO model per entity type) are merged into one
      multi-layer file, keeping the official splits.  Tokens are checked to be identical across
      layers.  I-X tags without a preceding B-X are repaired to B-X (and counted).
    * Each CoNLL sentence is located inside the standoff documents.  This allows measuring
      document-level leakage across the official splits (a sentence-level random split can put
      sentences of one referral in train and test) and explaining why the CoNLL layers contain
      fewer entities than the standoff files.
    * Exploratory only (NOT labels): dental-vocabulary prevalence and the most frequent entities,
      to judge whether a medical/dental "area" target is feasible.

Usage (from any folder)
-----------------------
    python 02_preparar_datos.py
    python 02_preparar_datos.py --cwlc-root "D:\\other\\path\\Chilean_Waiting_List"
    python 02_preparar_datos.py --skip cwlc
    python 02_preparar_datos.py --split-fractions 0.7 0.1 0.2 --seed 42

Run script 01 first (or let this script download ADE and BC5CDR into data/raw).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sys
import traceback
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# =============================================================================
# Configuration (kept separate from the logic below)
# =============================================================================
DEFAULT_CWLC_ROOT = Path("data") / "raw" / "cwlc"  # not used by the paper; override with --cwlc-root
ADE_RAW_URL = "https://raw.githubusercontent.com/trunghlt/AdverseDrugReaction/master/ADE-Corpus-V2/{name}"
ADE_FILES = ("DRUG-AE.rel", "ADE-NEG.txt")
BC5CDR_ZIP_URL = "https://raw.githubusercontent.com/JHnlp/BioCreative-V-CDR-Corpus/master/CDR_Data.zip"
BC5CDR_FILES = {
    "train": "CDR_TrainingSet.PubTator.txt",
    "val": "CDR_DevelopmentSet.PubTator.txt",  # the official "dev" set is our "val"
    "test": "CDR_TestSet.PubTator.txt",
}

# Split names are standardised across corpora: train / val / test.
SPLIT_ORDER = ("train", "val", "test")
CONLL_SPLIT_ALIASES = {"train": "train", "dev": "val", "val": "val", "test": "test"}
# Default ADE proportions = Table II of the proposal (15,873 / 1,764 / 5,879 of 23,516).
DEFAULT_SPLIT_FRACTIONS = (0.675, 0.075, 0.25)

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")  # words and single punctuation marks

# HYPOTHESIS (confirmed only by counts so far): the 7 CoNLL layers merge some fine standoff types.
CWLC_CONLL_TO_STANDOFF = {
    "Finding": ["Sign_or_Symptom", "Clinical_Finding", "Laboratory_or_Test_Result"],
    "Procedure": ["Procedure", "Laboratory_Procedure", "Diagnostic_Procedure", "Therapeutic_Procedure"],
}

# Sentence-to-document matching limits (see DocumentIndex).
MIN_KEY_LEN = 4  # sentences with fewer non-space characters are too ambiguous to locate
MAX_OCCURRENCES = 50  # stop counting occurrences of very common sentences

# Conservative dental vocabulary (accent-stripped, lower case, prefix match).  EXPLORATORY ONLY.
DENTAL_PATTERN = re.compile(
    r"\b(?:diente|dentici|dental|dentari|molar|premolar|incisiv|caries|ortodon|mordida|apinamiento|"
    r"overjet|deverjet|resalte|gingiv|periodon|endodon|odontolog|exodoncia|cordal)"
)
TOP_ENTITY_TYPES = ("Body_Part", "Disease", "Medication", "Sign_or_Symptom", "Abbreviation")
TOP_ENTITY_N = 25


# =============================================================================
# Generic helpers
# =============================================================================
def project_root() -> Path:
    """Project root = parent of scripts/ when this file lives there, otherwise its own folder."""
    here = Path(__file__).resolve().parent
    return here.parent if here.name.lower() == "scripts" else here


def log(message: str) -> None:
    """Progress message (flushed so it shows immediately on Windows consoles)."""
    print(message, flush=True)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def directory_digest(root: Path, suffixes: tuple[str, ...]) -> dict[str, Any]:
    """Fingerprint of all files with the given suffixes under ``root`` (names + contents)."""
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    combined = hashlib.sha256()
    for path in files:
        combined.update(path.relative_to(root).as_posix().encode("utf-8"))
        combined.update(sha256_file(path).encode("ascii"))
    return {"n_files": len(files), "sha256_of_listing": combined.hexdigest()}


def normalize_text(text: str) -> str:
    """NFKC + lower case + collapsed whitespace; used to detect duplicate texts."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def strip_accents_lower(text: str) -> str:
    """Lower-case text without diacritics (for the exploratory vocabulary counts)."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def quantiles(values: Sequence[float]) -> dict[str, float]:
    """Mean / median / p95 / max of a numeric sequence."""
    if len(values) == 0:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 2),
        "median": float(np.median(arr)),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "max": float(arr.max()),
    }


def guard_not_raw(path: Path) -> None:
    """Refuse to write inside a 'raw' folder: original data must stay untouched."""
    if "raw" in {part.lower() for part in path.parts}:
        raise ValueError(f"Refusing to write inside a raw folder: {path}")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write one JSON object per line (UTF-8, '\\n' endings even on Windows). Returns the count."""
    guard_not_raw(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def flatten(data: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten a nested dict into (dotted.key, value) rows for CSV export."""
    rows: list[tuple[str, Any]] = []
    for key, value in data.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            rows.extend(flatten(value, f"{full}."))
        else:
            rows.append((full, value))
    return rows


def save_report(report: dict[str, Any], name: str, interim_dir: Path, tables_dir: Path) -> None:
    """Write prep_<name>.json (full detail) and prep_<name>.csv (flat) outside raw."""
    for directory in (interim_dir, tables_dir):
        guard_not_raw(directory)
        directory.mkdir(parents=True, exist_ok=True)
    (interim_dir / f"prep_{name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    pd.DataFrame(flatten(report), columns=["metric", "value"]).to_csv(
        tables_dir / f"prep_{name}.csv", index=False, encoding="utf-8-sig"
    )


def download(url: str, dest: Path, timeout: int = 180) -> None:
    """Download ``url`` to ``dest`` (parent folders are created)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "thesis-prepare-data/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        dest.write_bytes(response.read())


def find_file(root: Path, name: str) -> Path | None:
    """Find ``name`` directly under ``root`` or in any sub-folder."""
    direct = root / name
    if direct.exists():
        return direct
    found = sorted(root.rglob(name)) if root.exists() else []
    return found[0] if found else None


def output_entry(path: Path, n_records: int | None = None) -> dict[str, Any]:
    """Manifest entry for one produced file."""
    entry: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
    if n_records is not None:
        entry["n_records"] = n_records
    return entry


# =============================================================================
# Tokenisation and BIO helpers (shared by ADE and BC5CDR)
# =============================================================================
def tokenize_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """Split text into words / punctuation marks, keeping character offsets (end exclusive)."""
    return [(m.start(), m.end(), m.group()) for m in TOKEN_PATTERN.finditer(text)]


def spans_to_bio(
    tokens: Sequence[tuple[int, int, str]], spans: Sequence[tuple[int, int, str]]
) -> dict[str, Any]:
    """Convert character spans (start, end, type) to flat BIO tags over ``tokens``.

    A flat tagging cannot hold overlapping spans, so when two spans overlap the longest one is
    kept and the other is counted in ``dropped``.  ``partial_boundary`` counts spans whose
    boundary falls inside a token (the tag is still assigned to the whole token).
    """
    tags = ["O"] * len(tokens)
    claimed = [False] * len(tokens)
    kept: list[tuple[int, int, str]] = []
    dropped = partial = 0
    for start, end, etype in sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0])):
        idx = [i for i, (a, b, _) in enumerate(tokens) if a < end and b > start]
        if not idx or any(claimed[i] for i in idx):
            dropped += 1
            continue
        if tokens[idx[0]][0] != start or tokens[idx[-1]][1] != end:
            partial += 1
        for rank, i in enumerate(idx):
            tags[i] = ("B-" if rank == 0 else "I-") + etype
            claimed[i] = True
        kept.append((start, end, etype))
    return {"tags": tags, "kept": sorted(kept), "dropped": dropped, "partial_boundary": partial}


def bio_to_spans(tokens: Sequence[tuple[int, int, str]], tags: Sequence[str]) -> list[tuple[int, int, str]]:
    """Inverse of :func:`spans_to_bio`: character spans (start, end, type) from BIO tags."""
    spans: list[tuple[int, int, str]] = []
    current: list[Any] | None = None
    for (start, end, _), tag in zip(tokens, tags):
        starts_new = tag.startswith("B-") or (tag.startswith("I-") and (current is None or current[2] != tag[2:]))
        if starts_new:
            if current is not None:
                spans.append((current[0], current[1], current[2]))
            current = [start, end, tag[2:]]
        elif tag.startswith("I-") and current is not None:
            current[1] = end
        else:
            if current is not None:
                spans.append((current[0], current[1], current[2]))
            current = None
    if current is not None:
        spans.append((current[0], current[1], current[2]))
    return spans


def repair_bio(tags: Sequence[str]) -> tuple[list[str], int]:
    """Turn every I-X that does not continue an X entity into B-X. Returns (tags, n_repairs)."""
    fixed: list[str] = []
    repairs, previous = 0, "O"
    for tag in tags:
        if tag.startswith("I-") and (previous == "O" or previous[2:] != tag[2:]):
            tag = "B-" + tag[2:]
            repairs += 1
        fixed.append(tag)
        previous = tag
    return fixed, repairs


def count_entities(tags: Sequence[str]) -> int:
    """Number of entities in a BIO sequence (an orphan I-X also starts one)."""
    count, previous = 0, "O"
    for tag in tags:
        if tag.startswith("B-") or (tag.startswith("I-") and (previous == "O" or previous[2:] != tag[2:])):
            count += 1
        previous = tag
    return count


# =============================================================================
# PART A - ADE-Corpus V2
# =============================================================================
def ensure_ade(ade_dir: Path, allow_download: bool) -> dict[str, Path]:
    """Return paths of the ADE files, downloading them if missing."""
    paths: dict[str, Path] = {}
    for name in ADE_FILES:
        path = find_file(ade_dir, name)
        if path is None:
            if not allow_download:
                raise FileNotFoundError(f"{name} not found in {ade_dir} (run script 01 or allow downloads)")
            download(ADE_RAW_URL.format(name=name), ade_dir / name)
            path = ade_dir / name
        paths[name] = path
    return paths


def load_ade_relations(path: Path) -> pd.DataFrame:
    """Read DRUG-AE.rel.  Offsets in this file are DOCUMENT-level, not sentence-level."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        pmid, tail = parts[0], parts[-6:]
        sentence = "|".join(parts[1:-6])  # tolerate '|' inside the sentence
        records.append(
            {
                "pmid": pmid,
                "sentence": sentence,
                "effect": tail[0],
                "effect_begin": int(tail[1]),
                "effect_end": int(tail[2]),
                "drug": tail[3],
                "drug_begin": int(tail[4]),
                "drug_end": int(tail[5]),
            }
        )
    return pd.DataFrame(records)


def load_ade_negatives(path: Path) -> pd.DataFrame:
    """Read ADE-NEG.txt (format: 'PMID NEG sentence')."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(\d+) NEG (.*)$", line)
        if match:
            records.append({"pmid": match.group(1), "sentence": match.group(2)})
    return pd.DataFrame(records)


def find_all(text: str, needle: str) -> list[int]:
    """Positions of every (possibly overlapping) occurrence of ``needle`` in ``text``."""
    positions, start = [], text.find(needle)
    while start != -1:
        positions.append(start)
        start = text.find(needle, start + 1)
    return positions


def locate_mentions(
    sentence: str, mentions: Sequence[tuple[str, int, str]]
) -> tuple[list[tuple[int, int, str]], dict[str, int]]:
    """Place document-level mentions inside their sentence.

    ``mentions`` = (type, document_offset, text).  The sentence start inside the document
    (``base``) is the value supported by most mentions: base = document_offset - position_in_sentence.
    Returns sentence-relative spans (start, end, type) and ambiguity counters.
    """
    support: Counter[int] = Counter()
    for _, begin, text in mentions:
        for base in {begin - pos for pos in find_all(sentence, text)}:
            support[base] += 1
    info = {"tie": 0, "fallback_mentions": 0}
    base: int | None = None
    if support:
        best = max(support.values())
        candidates = sorted(b for b, n in support.items() if n == best)
        info["tie"] = int(len(candidates) > 1)
        base = candidates[0]
    spans: set[tuple[int, int, str]] = set()
    for etype, begin, text in mentions:
        pos = -1 if base is None else begin - base
        if pos < 0 or sentence[pos : pos + len(text)] != text:
            occurrences = find_all(sentence, text)
            pos = occurrences[0] if occurrences else -1
            info["fallback_mentions"] += 1
        if pos >= 0:
            spans.add((pos, pos + len(text), etype))
    return sorted(spans), info


def make_ade_unit(
    sentence_id: str, pmid: str, text: str, label: int, n_relations: int, spans: list[tuple[int, int, str]]
) -> dict[str, Any]:
    """Build one ADE record: sentence, label and BIO layers.

    Three layers are stored: ``tags_drug`` and ``tags_effect`` (one flat model per type, keeps every
    span) and ``tags`` (merged; 120 DRUG/EFFECT overlaps force dropping the shorter span).
    """
    tokens = tokenize_with_offsets(text)
    merged = spans_to_bio(tokens, spans)
    per_type = {t: spans_to_bio(tokens, [s for s in spans if s[2] == t]) for t in ("DRUG", "EFFECT")}
    return {
        "sentence_id": sentence_id,
        "pmid": pmid,
        "text": text,
        "label": label,
        "n_relations": n_relations,
        "tokens": [t[2] for t in tokens],
        "token_offsets": [[t[0], t[1]] for t in tokens],
        "tags": merged["tags"],
        "tags_drug": per_type["DRUG"]["tags"],
        "tags_effect": per_type["EFFECT"]["tags"],
        "char_spans": [list(s) for s in spans],
        "_norm": normalize_text(text),  # internal; removed before writing
        "_dropped_overlap": merged["dropped"],
        "_partial": merged["partial_boundary"],
    }


def build_ade_units(rel: pd.DataFrame, neg: pd.DataFrame) -> tuple[list[dict[str, Any]], Counter]:
    """One unit per unique (PMID, normalised sentence); positives first, then negatives.

    Sentence ids match script 01 (ade_pos_#####, ade_neg_#####) so both outputs can be joined.
    """
    units: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for k, ((pmid, sentence), group) in enumerate(rel.groupby(["pmid", "sentence"], sort=False)):
        mentions: list[tuple[str, int, str]] = []
        for row in group.itertuples(index=False):
            mentions.append(("EFFECT", int(row.effect_begin), row.effect))
            mentions.append(("DRUG", int(row.drug_begin), row.drug))
        spans, info = locate_mentions(sentence, mentions)
        stats["ambiguous_base"] += info["tie"]
        stats["fallback_mentions"] += info["fallback_mentions"]
        key = (pmid, normalize_text(sentence))
        if key in seen:
            stats["dropped_duplicate_positive_units"] += 1
            continue
        seen.add(key)
        units.append(make_ade_unit(f"ade_pos_{k:05d}", pmid, sentence, 1, len(group), spans))
    for k, row in enumerate(neg.itertuples(index=False)):
        key = (row.pmid, normalize_text(row.sentence))
        if key in seen:
            stats["dropped_duplicate_negative_units"] += 1
            continue
        seen.add(key)
        units.append(make_ade_unit(f"ade_neg_{k:05d}", row.pmid, row.sentence, 0, 0, []))
    return units, stats


class UnionFind:
    """Minimal disjoint-set structure with path compression."""

    def __init__(self) -> None:
        self.parent: dict[Any, Any] = {}

    def find(self, item: Any) -> Any:
        """Representative of the set containing ``item`` (created on first use)."""
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: Any, b: Any) -> None:
        """Merge the sets of ``a`` and ``b``."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def assign_groups_to_splits(
    groups: dict[Any, dict[str, int]], fractions: Sequence[float], seed: int
) -> dict[Any, str]:
    """Greedy, seeded assignment of whole groups to train/val/test.

    Goal: each split gets its target share of units AND of positive units (stratification by
    class prevalence at group level).  Groups are processed largest first (hardest to place);
    each goes to the split that minimises the squared deviation from the targets.
    """
    if len(fractions) != len(SPLIT_ORDER) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must have {len(SPLIT_ORDER)} values summing to 1, got {fractions}")
    rng = np.random.RandomState(seed)
    tie_break = {g: rng.rand() for g in groups}  # drawn in insertion order -> deterministic
    order = sorted(groups, key=lambda g: (-groups[g]["n"], tie_break[g]))
    total_n = sum(g["n"] for g in groups.values())
    total_pos = max(sum(g["pos"] for g in groups.values()), 1)
    target = {s: (f * total_n, f * total_pos) for s, f in zip(SPLIT_ORDER, fractions)}
    load = {s: [0, 0] for s in SPLIT_ORDER}
    assignment: dict[Any, str] = {}
    for gid in order:
        n, pos = groups[gid]["n"], groups[gid]["pos"]
        best_split, best_cost = SPLIT_ORDER[0], float("inf")
        for candidate in SPLIT_ORDER:
            cost = 0.0
            for s in SPLIT_ORDER:
                add_n, add_pos = (n, pos) if s == candidate else (0, 0)
                cost += ((load[s][0] + add_n - target[s][0]) / total_n) ** 2
                cost += ((load[s][1] + add_pos - target[s][1]) / total_pos) ** 2
            if cost < best_cost - 1e-15:
                best_split, best_cost = candidate, cost
        assignment[gid] = best_split
        load[best_split][0] += n
        load[best_split][1] += pos
    return assignment


def prepare_ade(
    ade_dir: Path, processed_dir: Path, fractions: Sequence[float], seed: int, allow_download: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the frozen ADE split and write data/processed/ade/{train,val,test}.jsonl."""
    paths = ensure_ade(ade_dir, allow_download)
    rel, neg = load_ade_relations(paths["DRUG-AE.rel"]), load_ade_negatives(paths["ADE-NEG.txt"])
    units, stats = build_ade_units(rel, neg)
    log(f"  ADE: {len(units)} unique (PMID, sentence) units")

    # Group units that share a PMID OR a normalised text: whole groups go to one split, so neither
    # a PMID nor a sentence text can appear in two splits (no leakage by construction).
    uf = UnionFind()
    for unit in units:
        uf.union(("pmid", unit["pmid"]), ("text", unit["_norm"]))
    groups: dict[Any, dict[str, int]] = defaultdict(lambda: {"n": 0, "pos": 0})
    for unit in units:
        unit["_group"] = uf.find(("pmid", unit["pmid"]))
        groups[unit["_group"]]["n"] += 1
        groups[unit["_group"]]["pos"] += unit["label"]
    assignment = assign_groups_to_splits(dict(groups), fractions, seed)
    for unit in units:
        unit["split"] = assignment[unit["_group"]]

    # ---- verification: nothing crosses splits ------------------------------------------------
    by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLIT_ORDER}
    for unit in units:
        by_split[unit["split"]].append(unit)
    leakage: dict[str, Any] = {"shared_pmids": {}, "shared_normalized_texts": {}}
    for i, a in enumerate(SPLIT_ORDER):
        for b in SPLIT_ORDER[i + 1 :]:
            pair = f"{a}|{b}"
            leakage["shared_pmids"][pair] = len({u["pmid"] for u in by_split[a]} & {u["pmid"] for u in by_split[b]})
            leakage["shared_normalized_texts"][pair] = len(
                {u["_norm"] for u in by_split[a]} & {u["_norm"] for u in by_split[b]}
            )
    if any(v for part in leakage.values() for v in part.values()):
        raise RuntimeError(f"ADE split leaks across splits: {leakage}")

    # ---- write ---------------------------------------------------------------------------------
    outputs: list[dict[str, Any]] = []
    split_report: dict[str, Any] = {}
    total_n = len(units)
    total_pos = sum(u["label"] for u in units)
    for split, fraction in zip(SPLIT_ORDER, fractions):
        items = sorted(by_split[split], key=lambda u: u["sentence_id"])
        public = [{k: v for k, v in u.items() if not k.startswith("_") and k != "split"} | {"split": split} for u in items]
        out_path = processed_dir / "ade" / f"{split}.jsonl"
        n_written = write_jsonl(out_path, public)
        outputs.append(output_entry(out_path, n_written))
        pos = sum(u["label"] for u in items)
        tag_counts: Counter[str] = Counter(t for u in items for t in u["tags"])
        split_report[split] = {
            "units": len(items),
            "target_units": round(fraction * total_n, 1),
            "positives": pos,
            "target_positives": round(fraction * total_pos, 1),
            "negatives": len(items) - pos,
            "positive_prevalence": round(pos / len(items), 4) if items else 0.0,
            "pmids": len({u["pmid"] for u in items}),
            "tag_counts_merged": dict(tag_counts),
        }
    report = {
        "parameters": {"seed": seed, "fractions_train_val_test": list(fractions)},
        "input_manifest": {name: sha256_file(p) for name, p in paths.items()},
        "units": {
            "total": total_n,
            "positive": total_pos,
            "negative": total_n - total_pos,
            "prevalence_overall": round(total_pos / total_n, 4),
            **{k: v for k, v in stats.items()},
        },
        "groups": {
            "n_groups": len(groups),
            "largest_group_units": max(g["n"] for g in groups.values()),
            "groups_with_more_than_one_pmid_or_text_link": sum(1 for g in groups.values() if g["n"] > 1),
        },
        "splits": split_report,
        "leakage_check_after_split": leakage,
        "bio": {
            "spans_dropped_in_merged_layer_due_to_overlap": sum(u["_dropped_overlap"] for u in units),
            "spans_with_boundary_inside_a_token": sum(u["_partial"] for u in units),
            "note": "use tags_drug / tags_effect to keep every span; 'tags' is the merged, flat layer",
        },
    }
    return report, outputs


# =============================================================================
# PART B - BC5CDR
# =============================================================================
_PUBTATOR_TEXT = re.compile(r"^(\d+)\|([ta])\|(.*)$")


def ensure_bc5cdr(bc5_dir: Path, allow_download: bool) -> dict[str, Path]:
    """Return the three PubTator files; download and extract the official zip if missing."""
    found = {split: find_file(bc5_dir, name) for split, name in BC5CDR_FILES.items()}
    if all(found.values()):
        return {k: v for k, v in found.items() if v is not None}
    if not allow_download:
        raise FileNotFoundError(f"BC5CDR PubTator files not found in {bc5_dir} (run script 01 or allow downloads)")
    archive = bc5_dir / "CDR_Data.zip"
    download(BC5CDR_ZIP_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(bc5_dir)
    found = {split: find_file(bc5_dir, name) for split, name in BC5CDR_FILES.items()}
    if not all(found.values()):
        raise FileNotFoundError("The BC5CDR zip does not contain the expected PubTator files")
    return {k: v for k, v in found.items() if v is not None}


def parse_pubtator(path: Path) -> list[dict[str, Any]]:
    """Parse a PubTator file into articles with mentions and CID relations."""
    articles: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = _PUBTATOR_TEXT.match(line)
        if match:
            pmid, kind, text = match.groups()
            article = articles.setdefault(
                pmid, {"pmid": pmid, "title": "", "abstract": "", "mentions": [], "relations": []}
            )
            article["title" if kind == "t" else "abstract"] = text
            continue
        parts = line.split("\t")
        article = articles.get(parts[0])
        if article is None:
            continue
        if len(parts) >= 4 and parts[1] == "CID":
            article["relations"].append((parts[2], parts[3]))
        elif len(parts) >= 5:
            article["mentions"].append(
                {
                    "start": int(parts[1]),
                    "end": int(parts[2]),
                    "text": parts[3],
                    "type": parts[4],
                    "id": parts[5] if len(parts) > 5 else "",
                }
            )
    return list(articles.values())


def build_bc5cdr_record(article: dict[str, Any], split: str) -> dict[str, Any]:
    """One abstract -> one record.  Character spans are the source of truth for any tokenizer."""
    text = f"{article['title']} {article['abstract']}"  # offsets in PubTator assume exactly this join
    spans = sorted({(m["start"], m["end"], m["type"]) for m in article["mentions"]})
    tokens = tokenize_with_offsets(text)
    merged = spans_to_bio(tokens, spans)
    layers = {t: spans_to_bio(tokens, [s for s in spans if s[2] == t]) for t in ("Chemical", "Disease")}
    roundtrip_bad = 0
    for etype, layer in layers.items():
        expected = [s for s in spans if s[2] == etype]
        roundtrip_bad += int(sorted(bio_to_spans(tokens, layer["tags"])) != expected)
    mention_by_span = {(m["start"], m["end"], m["type"]): m for m in article["mentions"]}
    return {
        "pmid": article["pmid"],
        "split": split,
        "text": text,
        "title_end": len(article["title"]),
        "tokens": [t[2] for t in tokens],
        "token_offsets": [[t[0], t[1]] for t in tokens],
        "tags": merged["tags"],
        "tags_chemical": layers["Chemical"]["tags"],
        "tags_disease": layers["Disease"]["tags"],
        "mentions": [
            {"start": s, "end": e, "type": t, "text": mention_by_span[(s, e, t)]["text"], "concept_id": mention_by_span[(s, e, t)]["id"]}
            for s, e, t in spans
        ],
        "cid_pairs": sorted(set(article["relations"])),
        "_roundtrip_bad": roundtrip_bad,
        "_dropped": merged["dropped"],
    }


def prepare_bc5cdr(bc5_dir: Path, processed_dir: Path, allow_download: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert the official BC5CDR splits (kept as they are) and write data/processed/bc5cdr/*.jsonl."""
    paths = ensure_bc5cdr(bc5_dir, allow_download)
    outputs: list[dict[str, Any]] = []
    report: dict[str, Any] = {"splits": {}, "input_manifest": {}, "note": "official splits kept; 'dev' is stored as 'val'"}
    pmids: dict[str, set[str]] = {}
    for split in SPLIT_ORDER:
        articles = parse_pubtator(paths[split])
        records = [build_bc5cdr_record(a, split) for a in articles]
        pmids[split] = {r["pmid"] for r in records}
        n_tokens = [len(r["tokens"]) for r in records]
        mention_types = Counter(m["type"] for r in records for m in r["mentions"])
        public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
        out_path = processed_dir / "bc5cdr" / f"{split}.jsonl"
        n_written = write_jsonl(out_path, public)
        outputs.append(output_entry(out_path, n_written))
        report["input_manifest"][paths[split].name] = sha256_file(paths[split])
        report["splits"][split] = {
            "n_abstracts": len(records),
            "mentions_per_type": dict(mention_types),
            "regex_tokens_per_abstract": quantiles(n_tokens),
            # WordPiece usually yields noticeably more pieces than regex tokens, so these are lower bounds:
            "abstracts_over_256_regex_tokens": int(sum(n > 256 for n in n_tokens)),
            "abstracts_over_400_regex_tokens": int(sum(n > 400 for n in n_tokens)),
            "layers_where_bio_does_not_reproduce_char_spans": int(sum(r["_roundtrip_bad"] for r in records)),
            "spans_dropped_in_merged_layer": int(sum(r["_dropped"] for r in records)),
        }
    for i, a in enumerate(SPLIT_ORDER):
        for b in SPLIT_ORDER[i + 1 :]:
            if pmids[a] & pmids[b]:
                raise RuntimeError(f"BC5CDR splits {a}/{b} share PMIDs")
    report["leakage_check"] = "no PMID shared between splits"
    return report, outputs


# =============================================================================
# PART C - Chilean Waiting List (CWLC)
# =============================================================================
_BRAT_ENTITY = re.compile(r"^(T\d+)\t(\S+) ([\d; ]+)\t(.*)$")


def parse_brat(ann_text: str) -> dict[str, Any]:
    """Parse a brat .ann file: entities, attributes (A), relations (R) and a count of notes (#)."""
    entities: list[dict[str, Any]] = []
    attributes: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    n_notes = 0
    for line in ann_text.splitlines():
        match = _BRAT_ENTITY.match(line)
        if match:
            eid, etype, offsets, mention = match.groups()
            chunks = [chunk.split() for chunk in offsets.split(";")]  # discontinuous: '0 5;10 15'
            entities.append(
                {
                    "id": eid,
                    "type": etype,
                    "start": int(chunks[0][0]),
                    "end": int(chunks[-1][1]),
                    "text": mention,
                    "discontinuous": len(chunks) > 1,
                }
            )
            continue
        if line.startswith("#"):
            n_notes += 1
            continue
        fields = line.split("\t")
        parts = fields[1].split() if len(fields) > 1 else []
        if line.startswith("A") and len(parts) >= 2:  # e.g. 'A1<TAB>Negated T3'
            attributes.append(
                {"id": fields[0], "name": parts[0], "target": parts[1], "value": parts[2] if len(parts) > 2 else None}
            )
        elif line.startswith("R") and len(parts) >= 3:  # e.g. 'R1<TAB>Has Arg1:T1 Arg2:T2'
            args = dict(p.split(":", 1) for p in parts[1:] if ":" in p)
            relations.append({"id": fields[0], "type": parts[0], "arg1": args.get("Arg1"), "arg2": args.get("Arg2")})
    return {"entities": entities, "attributes": attributes, "relations": relations, "n_notes": n_notes}


def universal_newlines(text: str) -> str:
    """Treat CRLF and lone CR as a single '\\n' (what text-mode reading does on Windows)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def count_offset_mismatches(text: str, entities: Sequence[dict[str, Any]]) -> int:
    """Entities whose annotated text differs from text[start:end] (discontinuous ones are skipped)."""
    bad = 0
    for entity in entities:
        if entity["discontinuous"]:
            continue
        if normalize_text(text[entity["start"] : entity["end"]]) != normalize_text(entity["text"]):
            bad += 1
    return bad


def resolve_reading(raw: str, entities: Sequence[dict[str, Any]]) -> tuple[str, str, int]:
    """Choose, per document, the text reading for which the brat offsets are valid.

    Script 01 showed that most documents match when line breaks are kept as in the file ("raw"),
    while a few (~46) match only when CRLF counts as one character ("universal_newlines").
    Returns (text, reading_name, remaining_mismatches).
    """
    bad_raw = count_offset_mismatches(raw, entities)
    if bad_raw == 0 or "\r" not in raw:
        return raw, "raw", bad_raw
    alternative = universal_newlines(raw)
    bad_alt = count_offset_mismatches(alternative, entities)
    if bad_alt < bad_raw:
        return alternative, "universal_newlines", bad_alt
    return raw, "raw", bad_raw


def load_cwlc_documents(standoff_dir: Path) -> list[dict[str, Any]]:
    """Read every .ann/.txt pair and resolve its text reading."""
    docs: list[dict[str, Any]] = []
    for ann in sorted(standoff_dir.rglob("*.ann")):
        txt = ann.with_suffix(".txt")
        if not txt.exists():
            continue
        # Bytes are decoded manually so that '\r\n' is NOT translated to '\n' (offsets depend on it).
        raw = txt.read_bytes().decode("utf-8", errors="replace").lstrip("\ufeff")
        parsed = parse_brat(ann.read_text(encoding="utf-8"))
        text, reading, bad = resolve_reading(raw, parsed["entities"])
        docs.append(
            {
                "doc_id": ann.stem,
                "text": text,
                "reading": reading,
                "offset_mismatches": bad,
                "has_cr": "\r" in raw,
                **parsed,
            }
        )
    return docs


def nesting_definitions(docs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Nested-span rate under several definitions (script 01 used the first one).

    strict_pair            span strictly contains or is strictly contained in another span
    strict_or_identical    the above, plus spans with identical boundaries as another span
    inner_strict           span strictly contained in another span
    inner_or_identical     the above, plus spans with identical boundaries as another span
    The Zenodo page reports '>48 % embedded in, or contain, other entities'.
    """
    totals: Counter[str] = Counter()
    n_spans = 0
    for doc in docs:
        ents = doc["entities"]
        n = len(ents)
        n_spans += n
        strict, identical, inner = [False] * n, [False] * n, [False] * n
        for i in range(n):
            for j in range(i + 1, n):
                si, ei, sj, ej = ents[i]["start"], ents[i]["end"], ents[j]["start"], ents[j]["end"]
                if (si, ei) == (sj, ej):
                    identical[i] = identical[j] = True
                elif si <= sj and ej <= ei:
                    strict[i] = strict[j] = True
                    inner[j] = True
                elif sj <= si and ei <= ej:
                    strict[i] = strict[j] = True
                    inner[i] = True
        totals["strict_pair"] += sum(strict)
        totals["strict_or_identical"] += sum(a or b for a, b in zip(strict, identical))
        totals["inner_strict"] += sum(inner)
        totals["inner_or_identical"] += sum(a or b for a, b in zip(inner, identical))
    return {
        "n_spans": n_spans,
        **{f"{name}_spans": total for name, total in totals.items()},
        **{f"{name}_rate": round(total / n_spans, 4) if n_spans else 0.0 for name, total in totals.items()},
    }


# ---- CoNLL reading ------------------------------------------------------------------------------
def read_conll(path: Path) -> list[list[list[str]]]:
    """Read a CoNLL file into sentences; each row is [token, tag, ...extra columns]."""
    sentences: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if current:
                sentences.append(current)
                current = []
            continue
        if line.startswith("-DOCSTART-"):
            continue
        current.append(line.split())
    if current:
        sentences.append(current)
    return sentences


def conll_split_name(path: Path) -> str:
    """'train' / 'val' / 'test' from Disease/Disease_dev.conll or Disease/dev.conll."""
    etype, stem = path.parent.name, path.stem
    prefix = f"{etype.lower()}_"
    raw_name = stem[len(prefix):] if stem.lower().startswith(prefix) else stem
    return CONLL_SPLIT_ALIASES.get(raw_name.lower(), raw_name.lower())


def find_cwlc_dirs(root: Path) -> tuple[Path | None, Path | None]:
    """Locate the standoff folder (cwlc) and the CoNLL folder (cwlc_conll-format)."""
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    conll = next((p for p in subdirs if "conll" in p.name.lower()), None)
    standoff = next((p for p in subdirs if p != conll and p.name.lower().startswith("cwlc")), None)
    return standoff, conll


# ---- Locating CoNLL sentences inside standoff documents -----------------------------------------
def norm_char(ch: str) -> str:
    """One base, lower-case character per input character (accents stripped, 1:1 mapping kept)."""
    return unicodedata.normalize("NFD", ch)[0].lower()[0]


def build_key(text: str) -> tuple[str, list[int]]:
    """Whitespace-free, accent-free, lower-case key of ``text`` and key-position -> char-offset map."""
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        chars.append(norm_char(ch))
        index.append(i)
    return "".join(chars), index


def sentence_key(tokens: Sequence[str]) -> str:
    """Key of a tokenised sentence built with the same normalisation as :func:`build_key`."""
    return "".join(norm_char(ch) for tok in tokens for ch in tok if not ch.isspace())


class DocumentIndex:
    """Search index to find which standoff document(s) contain a given CoNLL sentence.

    All document keys are concatenated into one big string separated by NUL; a sentence key is
    searched with str.find, and each hit is mapped back to its document by bisecting the start
    offsets.  Because keys ignore whitespace and accents, the match is robust to tokenisation.
    """

    def __init__(self, docs: Sequence[dict[str, Any]]) -> None:
        self.maps: list[list[int]] = []
        self.starts: list[int] = []
        parts: list[str] = []
        position = 0
        for doc in docs:
            key, index = build_key(doc["text"])
            self.maps.append(index)
            self.starts.append(position)
            parts.append(key)
            position += len(key) + 1  # +1 for the NUL separator
        self.big = "\x00".join(parts)
        self._cache: dict[str, dict[str, Any]] = {}

    def locate(self, skey: str) -> dict[str, Any]:
        """Candidate documents for a sentence key.

        status: 'too_short' | 'none' | 'unique' (one hit overall) | 'single_doc_repeated'
        (several hits, all in one document) | 'multi_doc' | 'capped' (>= MAX_OCCURRENCES hits).
        """
        if skey in self._cache:
            return self._cache[skey]
        if len(skey) < MIN_KEY_LEN:
            result = {"status": "too_short", "n_docs": None, "doc_index": None, "key_start": None}
            self._cache[skey] = result
            return result
        hits: list[tuple[int, int]] = []
        pos = self.big.find(skey)
        while pos != -1 and len(hits) < MAX_OCCURRENCES:
            doc_idx = bisect.bisect_right(self.starts, pos) - 1
            hits.append((doc_idx, pos - self.starts[doc_idx]))
            pos = self.big.find(skey, pos + 1)
        capped = pos != -1
        docs = {d for d, _ in hits}
        if capped:
            status = "capped"
        elif not hits:
            status = "none"
        elif len(hits) == 1:
            status = "unique"
        elif len(docs) == 1:
            status = "single_doc_repeated"
        else:
            status = "multi_doc"
        result = {
            "status": status,
            "n_docs": len(docs),
            # the document is only trusted when exactly one document contains the sentence
            "doc_index": next(iter(docs)) if len(docs) == 1 and not capped else None,
            "key_start": hits[0][1] if status == "unique" else None,
        }
        self._cache[skey] = result
        return result

    def char_range(self, doc_index: int, key_start: int, key_len: int) -> tuple[int, int]:
        """Character range [start, end) in the document text covered by a located sentence."""
        index = self.maps[doc_index]
        return index[key_start], index[key_start + key_len - 1] + 1


def flatten_loss(spans: Sequence[tuple[int, int]]) -> int:
    """Spans lost if overlapping spans must be flattened (longest first wins)."""
    kept: list[tuple[int, int]] = []
    lost = 0
    for start, end in sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0])):
        if any(start < k_end and end > k_start for k_start, k_end in kept):
            lost += 1
        else:
            kept.append((start, end))
    return lost


def entity_coverage(
    docs: Sequence[dict[str, Any]],
    ranges_by_doc: dict[int, list[tuple[int, int]]],
    conll_entities: dict[str, int],
) -> dict[str, Any]:
    """Explain the gap between standoff and CoNLL entity counts, layer by layer.

    For every continuous standoff entity of a CoNLL layer (documents whose offsets could not be
    resolved are excluded), classify it as: inside a located sentence / crossing a sentence
    boundary / in a region with no located sentence / in a document with no located sentence.
    Also estimate how many entities a flat tagging per layer must lose because of overlaps.
    """
    reverse = {member: layer for layer, members in CWLC_CONLL_TO_STANDOFF.items() for member in members}
    rows: dict[str, Counter[str]] = defaultdict(Counter)
    for d, doc in enumerate(docs):
        if doc["offset_mismatches"] > 0:
            rows["_all"]["documents_excluded_offset_unresolved"] += 1
            continue
        per_layer: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for entity in doc["entities"]:
            if entity["discontinuous"]:
                continue
            layer = reverse.get(entity["type"], entity["type"])
            if layer not in conll_entities:
                continue
            per_layer[layer].append((entity["start"], entity["end"]))
            row = rows[layer]
            row["standoff_entities_checked"] += 1
            ranges = ranges_by_doc.get(d, [])
            if not ranges:
                row["doc_without_located_sentences"] += 1
            elif any(r0 <= entity["start"] and entity["end"] <= r1 for r0, r1 in ranges):
                row["inside_located_sentence"] += 1
            elif any(entity["start"] < r1 and entity["end"] > r0 for r0, r1 in ranges):
                row["crosses_sentence_boundary"] += 1
            else:
                row["in_region_without_located_sentence"] += 1
        for layer, spans in per_layer.items():
            rows[layer]["estimated_flattening_loss_within_layer"] += flatten_loss(spans)
    result: dict[str, Any] = {
        "note": "categories 'crosses_sentence_boundary' and 'estimated_flattening_loss_within_layer' can overlap "
        "(an entity may be both), so do not simply add them up to the gap"
    }
    for layer, n_conll in conll_entities.items():
        row = rows.get(layer, Counter())
        result[layer] = {"conll_entities": n_conll, **dict(row)}
    result["_documents_excluded_offset_unresolved"] = rows["_all"]["documents_excluded_offset_unresolved"]
    return result


# ---- Main CWLC routine ----------------------------------------------------------------------------
def prepare_cwlc(root: Path, processed_dir: Path, tables_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge the CoNLL layers, resolve standoff offsets, align both views and write the outputs."""
    if not root.exists():
        raise FileNotFoundError(f"CWLC folder does not exist: {root}")
    standoff_dir, conll_dir = find_cwlc_dirs(root)
    if conll_dir is None:
        raise FileNotFoundError(f"No CoNLL sub-folder (cwlc_conll-format) under {root}")
    report: dict[str, Any] = {"root": str(root)}
    outputs: list[dict[str, Any]] = []

    # ---- 1. standoff documents: offsets, nesting, attributes ---------------------------------------
    docs: list[dict[str, Any]] = []
    index: DocumentIndex | None = None
    if standoff_dir is not None:
        log("  CWLC: reading standoff documents ...")
        docs = load_cwlc_documents(standoff_dir)
        readings = Counter(d["reading"] for d in docs)
        report["standoff"] = {
            "n_documents": len(docs),
            "reading_used": dict(readings),
            "documents_with_unresolved_offsets": sum(d["offset_mismatches"] > 0 for d in docs),
            "entities_with_unresolved_offsets": sum(d["offset_mismatches"] for d in docs),
            "entities_per_type": dict(Counter(e["type"] for d in docs for e in d["entities"])),
            "attributes": dict(Counter(a["name"] for d in docs for a in d["attributes"])),
            "relations": dict(Counter(r["type"] for d in docs for r in d["relations"])),
            "attribute_targets_by_entity_type": {},
            "nesting": nesting_definitions(docs),
            "manifest": directory_digest(standoff_dir, (".ann", ".txt")),
        }
        # Which entity types carry each attribute (e.g. is 'Negated' only on findings?).
        targets: dict[str, Counter[str]] = defaultdict(Counter)
        for d in docs:
            type_of = {e["id"]: e["type"] for e in d["entities"]}
            for a in d["attributes"]:
                targets[a["name"]][type_of.get(a["target"], "?")] += 1
        report["standoff"]["attribute_targets_by_entity_type"] = {k: dict(v) for k, v in targets.items()}
        index = DocumentIndex(docs)

    # ---- 2. CoNLL layers: strict alignment across entity types -------------------------------------
    files = sorted(conll_dir.rglob("*.conll"))
    if not files:
        raise FileNotFoundError(f"No .conll files under {conll_dir}")
    by_split: dict[str, dict[str, list[list[list[str]]]]] = defaultdict(dict)
    manifest: dict[str, str] = {}
    for path in files:
        by_split[conll_split_name(path)][path.parent.name] = read_conll(path)
        manifest[f"{path.parent.name}/{path.name}"] = sha256_file(path)
    layer_names = sorted({t for per_type in by_split.values() for t in per_type})
    for split, per_type in by_split.items():
        if set(per_type) != set(layer_names):
            raise ValueError(f"Split '{split}' is missing layers: {sorted(set(layer_names) - set(per_type))}")
    unknown = [s for s in by_split if s not in SPLIT_ORDER]
    if unknown:
        raise ValueError(f"Unexpected split names in the CoNLL files: {unknown}")

    train_norms: set[str] = set()
    records: dict[str, list[dict[str, Any]]] = {}
    conll_entities: Counter[str] = Counter()
    repairs: Counter[str] = Counter()
    extra_column_lines: Counter[str] = Counter()
    for split in SPLIT_ORDER:
        per_type = by_split[split]
        reference = per_type[layer_names[0]]
        tokens_list = [[row[0] for row in sent] for sent in reference]
        for etype in layer_names:
            if [[row[0] for row in sent] for sent in per_type[etype]] != tokens_list:
                raise ValueError(f"Tokens differ between layers '{layer_names[0]}' and '{etype}' in split '{split}'")
        sentence_tags: list[dict[str, list[str]]] = [dict() for _ in tokens_list]
        for etype in layer_names:
            for i, sent in enumerate(per_type[etype]):
                extra_column_lines[etype] += sum(len(row) > 2 for row in sent)
                tags, n_fixed = repair_bio([row[1] for row in sent])  # first tag column only
                repairs[etype] += n_fixed
                conll_entities[etype] += count_entities(tags)
                sentence_tags[i][etype] = tags
        norms = [normalize_text(" ".join(toks)) for toks in tokens_list]
        counts = Counter(norms)
        if split == "train":
            train_norms = set(norms)
        records[split] = [
            {
                "sentence_id": f"cwlc_{split}_{i:05d}",
                "split": split,
                "tokens": tokens_list[i],
                "tags": sentence_tags[i],
                "seen_in_train": split != "train" and norms[i] in train_norms,
                "duplicate_within_split": counts[norms[i]] > 1,
                "_key": sentence_key(tokens_list[i]),
            }
            for i in range(len(tokens_list))
        ]
    report["conll"] = {
        "layers": layer_names,
        "sentences_per_split": {s: len(r) for s, r in records.items()},
        "tokens_identical_across_layers": True,
        "entities_per_layer_after_bio_repair": dict(conll_entities),
        "bio_repairs_per_layer": dict(repairs),
        "lines_with_extra_tag_columns_ignored": dict(extra_column_lines),
        "sentences_seen_in_train": {s: sum(r["seen_in_train"] for r in records[s]) for s in ("val", "test")},
        "sentences_duplicated_within_split": {s: sum(r["duplicate_within_split"] for r in records[s]) for s in SPLIT_ORDER},
        "manifest": manifest,
    }

    # ---- 3. locate every sentence inside the standoff documents ------------------------------------
    ranges_by_doc: dict[int, list[tuple[int, int]]] = defaultdict(list)
    if index is not None:
        log("  CWLC: locating CoNLL sentences inside documents ...")
        status_counts: Counter[str] = Counter()
        for split in SPLIT_ORDER:
            for rec in records[split]:
                hit = index.locate(rec["_key"])
                rec["doc_match"] = hit["status"]
                rec["n_candidate_docs"] = hit["n_docs"]
                rec["doc_id"] = docs[hit["doc_index"]]["doc_id"] if hit["doc_index"] is not None else None
                rec["_doc_index"] = hit["doc_index"]
                status_counts[hit["status"]] += 1
                if hit["key_start"] is not None:
                    ranges_by_doc[hit["doc_index"]].append(
                        index.char_range(hit["doc_index"], hit["key_start"], len(rec["_key"]))
                    )
        total_sentences = sum(status_counts.values())
        report["alignment"] = {
            "match_status": dict(status_counts),
            "share_assigned_to_a_single_document": round(
                sum(1 for s in SPLIT_ORDER for r in records[s] if r["doc_id"] is not None) / total_sentences, 4
            ),
            "note": "'multi_doc' / 'capped' sentences are boilerplate present in several referrals; they stay unassigned",
        }
        # Document-level leakage: do sentences of the same referral sit in different splits?
        splits_of_doc: dict[int, set[str]] = defaultdict(set)
        for split in SPLIT_ORDER:
            for rec in records[split]:
                if rec["_doc_index"] is not None:
                    splits_of_doc[rec["_doc_index"]].add(split)
        straddling = sum(1 for s in splits_of_doc.values() if len(s) > 1)
        leak: dict[str, Any] = {
            "documents_with_located_sentences": len(splits_of_doc),
            "documents_with_sentences_in_more_than_one_split": straddling,
            "share_of_documents_straddling": round(straddling / len(splits_of_doc), 4) if splits_of_doc else None,
        }
        for split in ("val", "test"):
            assigned = [r for r in records[split] if r["_doc_index"] is not None]
            shared = [r for r in assigned if "train" in splits_of_doc[r["_doc_index"]]]
            leak[f"{split}_sentences_assigned_to_a_document"] = len(assigned)
            leak[f"{split}_sentences_whose_document_also_has_sentences_in_train"] = len(shared)
            leak[f"{split}_share_sharing_document_with_train"] = round(len(shared) / len(assigned), 4) if assigned else None
        # Are sentences stored in document order inside each file?
        for split in SPLIT_ORDER:
            pairs = [
                (a["_doc_index"], b["_doc_index"])
                for a, b in zip(records[split], records[split][1:])
                if a["_doc_index"] is not None and b["_doc_index"] is not None
            ]
            leak[f"{split}_consecutive_sentences_from_same_document"] = (
                round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None
            )
        report["document_level_leakage"] = leak
        report["entity_coverage"] = entity_coverage(docs, ranges_by_doc, dict(conll_entities))

    # ---- 4. exploratory area vocabulary (NOT labels) ---------------------------------------------------
    if docs:
        hits = [len(DENTAL_PATTERN.findall(strip_accents_lower(d["text"]))) for d in docs]
        for d, h in zip(docs, hits):
            d["dental_term_hits"] = h
        report["area_exploration"] = {
            "warning": "EXPLORATORY vocabulary counts, not area labels; to be validated by experts",
            "documents_with_at_least_1_dental_term": round(sum(h >= 1 for h in hits) / len(hits), 4),
            "documents_with_at_least_2_dental_terms": round(sum(h >= 2 for h in hits) / len(hits), 4),
            "documents_with_at_least_3_dental_terms": round(sum(h >= 3 for h in hits) / len(hits), 4),
            "share_of_documents_in_all_caps_like_text": round(
                sum(1 for d in docs if sum(c.isupper() for c in d["text"]) > 0.6 * max(sum(c.isalpha() for c in d["text"]), 1))
                / len(docs),
                4,
            ),
        }
        top_rows = []
        for etype in TOP_ENTITY_TYPES:
            counter = Counter(normalize_text(e["text"]) for d in docs for e in d["entities"] if e["type"] == etype)
            top_rows += [{"entity_type": etype, "text": text, "count": n} for text, n in counter.most_common(TOP_ENTITY_N)]
        guard_not_raw(tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(top_rows).to_csv(tables_dir / "cwlc_top_entities.csv", index=False, encoding="utf-8-sig")

    # ---- 5. write outputs --------------------------------------------------------------------------------
    for split in SPLIT_ORDER:
        public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records[split]]
        out_path = processed_dir / "cwlc" / f"{split}.jsonl"
        outputs.append(output_entry(out_path, write_jsonl(out_path, public)))
    if docs:
        # documents.jsonl = standoff view: resolved text, fine-grained (nested) entities, attributes, relations
        out_path = processed_dir / "cwlc" / "documents.jsonl"
        outputs.append(output_entry(out_path, write_jsonl(out_path, docs)))
    return report, outputs


# =============================================================================
# Command line and orchestration
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments."""
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cwlc-root", type=Path, default=Path(os.environ.get("CWLC_ROOT", DEFAULT_CWLC_ROOT)))
    parser.add_argument("--ade-dir", type=Path, default=root / "data" / "raw" / "ade_corpus_v2")
    parser.add_argument("--bc5cdr-dir", type=Path, default=root / "data" / "raw" / "bc5cdr")
    parser.add_argument("--processed-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--interim-dir", type=Path, default=root / "data" / "interim")
    parser.add_argument("--tables-dir", type=Path, default=root / "outputs" / "tables")
    parser.add_argument("--skip", nargs="*", default=[], choices=["ade", "bc5cdr", "cwlc"])
    parser.add_argument("--no-download", action="store_true", help="do not download ADE/BC5CDR if missing")
    parser.add_argument("--seed", type=int, default=42, help="seed of the ADE group split")
    parser.add_argument(
        "--split-fractions", type=float, nargs=3, default=list(DEFAULT_SPLIT_FRACTIONS),
        metavar=("TRAIN", "VAL", "TEST"), help="ADE split proportions (must sum to 1)",
    )
    return parser


def main() -> None:
    """Run the three preparation steps; a failure in one corpus does not stop the others."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]  # avoid cp1252 errors on Windows
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args()
    reports: dict[str, dict[str, Any]] = {}
    produced: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    steps = {
        "ade": lambda: prepare_ade(args.ade_dir, args.processed_dir, args.split_fractions, args.seed, not args.no_download),
        "bc5cdr": lambda: prepare_bc5cdr(args.bc5cdr_dir, args.processed_dir, not args.no_download),
        "cwlc": lambda: prepare_cwlc(args.cwlc_root, args.processed_dir, args.tables_dir),
    }
    for number, (name, step) in enumerate(steps.items(), start=1):
        if name in args.skip:
            continue
        log(f"[{number}/3] {name.upper()}")
        try:
            reports[name], produced[name] = step()
            save_report(reports[name], name, args.interim_dir, args.tables_dir)
        except Exception as exc:  # noqa: BLE001 - keep going with the other corpora
            errors[name] = f"{type(exc).__name__}: {exc}"
            log(traceback.format_exc())

    # Manifest of everything produced, so later stages can verify they read the frozen files.
    guard_not_raw(args.processed_dir)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.processed_dir / "manifest_processed.json"
    manifest: dict[str, Any] = {"parameters": {}, "files": {}}
    if manifest_path.exists():  # keep entries of corpora that were not (re)built in this run
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, entries in produced.items():
        for entry in entries:  # store paths relative to the processed folder so the manifest is portable
            entry["path"] = Path(os.path.relpath(entry["path"], args.processed_dir)).as_posix()
        manifest["files"][name] = entries
    if "ade" in produced:
        manifest["parameters"]["ade"] = {"seed": args.seed, "split_fractions": list(args.split_fractions)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if errors:
        guard_not_raw(args.interim_dir)
        args.interim_dir.mkdir(parents=True, exist_ok=True)
        (args.interim_dir / "errores_02.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- console summary (Spanish, for the thesis author) ---------------------------------------------
    log("\n=== RESUMEN ===")
    if "ade" in reports:
        for split, info in reports["ade"]["splits"].items():
            log(f"ADE {split:5s}: {info['units']:6d} oraciones | {info['positives']:5d} positivas "
                f"({info['positive_prevalence']:.1%}) | {info['pmids']} PMID")
        log("ADE: sin PMID ni texto compartidos entre splits (verificado)")
    if "bc5cdr" in reports:
        sizes = {s: i["n_abstracts"] for s, i in reports["bc5cdr"]["splits"].items()}
        log(f"BC5CDR: resumenes por split {sizes}")
    if "cwlc" in reports:
        cw = reports["cwlc"]
        if "standoff" in cw:
            log(f"CWLC: lectura de texto usada {cw['standoff']['reading_used']}; "
                f"documentos con offsets sin resolver: {cw['standoff']['documents_with_unresolved_offsets']}")
        if "document_level_leakage" in cw:
            leak = cw["document_level_leakage"]
            share = leak["test_share_sharing_document_with_train"]
            if share is not None:
                log(f"CWLC: {share:.1%} de las oraciones de test (asignables a un documento) comparten documento con train")
    for name, message in errors.items():
        log(f"ERROR en {name}: {message}")
    log(f"\nDatos procesados: {args.processed_dir}\nReportes: {args.interim_dir} y {args.tables_dir}")


if __name__ == "__main__":
    main()
