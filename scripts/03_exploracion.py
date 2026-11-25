#!/usr/bin/env python3
"""Script 03 - Descriptive statistics, thesis figures and simple lexical baselines.

Stage 2 of the thesis plan (exploration). This script reads ONLY the frozen files
written by script 02 (data/processed/**) and never touches data/raw.

What it does
------------
0. Integrity check: SHA-256 of every processed file against manifest_processed.json
   (and, optionally, against a second manifest passed with --expected-manifest, e.g.
   the one produced on another machine). A mismatch aborts the run, because every
   number below would otherwise describe a different dataset.
1. Descriptive tables per corpus and split (sizes, lengths, entity counts/density,
   multi-token share, duplicates). CSV + LaTeX (booktabs) for the thesis.
2. Unseen-mention rate: share of val/test mention occurrences whose exact surface
   string never appears as a gold mention in train. This is the single most useful
   number to read any NER result: it bounds what memorisation can achieve.
3. Baseline A (ADE-Corpus V2, sentence classification "has an adverse drug event"):
   TF-IDF + logistic regression, hyper-parameter and threshold chosen on val only,
   95% CI on test by cluster bootstrap over PMIDs, top-weighted n-grams, and a
   leakage demonstration (frozen group split vs random splits, including the
   "one row per relation" construction that inflates a naive split).
4. Baseline B (NER, every corpus/layer): exact-match dictionary built from train
   mentions (longest match, left to right). Entity-level exact-match P/R/F1 with
   cluster-bootstrap CI, plus recall split by seen/unseen mentions. For CWLC a
   second test variant keeps only sentences not present in train.
5. Figures (PNG 300 dpi + vector PDF) designed to survive black-and-white printing:
   gray levels + hatches + line styles, never color alone.

Why these baselines
-------------------
They are NOT competitors for the fine-tuned models; they are floors and diagnostics.
If a PEFT model barely beats a dictionary, the benchmark is mostly memorisation; if the
dictionary reaches its own upper bound, the corpus has little lexical novelty.

Assumptions and limitations (also written to data/interim/eda_summary.json)
---------------------------------------------------------------------------
* Mention identity = lower-cased token tuple. Case-insensitive matching is deliberate
  (49.8% of CWLC documents are all caps) but merges genuinely different strings.
* The dictionary ignores context, so it cannot separate e.g. "Abbreviation" senses.
* Bootstrap clusters: PMID (ADE, BC5CDR) and doc_id (CWLC, falls back to sentence).
  If doc_id is missing for many CWLC sentences the CI is optimistic.
* ADE-sentence label = at least one annotated drug-effect relation. It is not a
  causality judgement about the drug; it is the annotators' relation.
* Everything is single-seed (--seed) except the random-split leakage demo, which uses
  --n-repeats seeds so that its spread is visible.

Usage
-----
    python scripts/03_exploracion.py                       # all corpora found
    python scripts/03_exploracion.py --skip cwlc
    python scripts/03_exploracion.py --n-boot 200 --no-baselines   # quick look
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

SPLITS: tuple[str, ...] = ("train", "val", "test")
CORPORA: tuple[str, ...] = ("ade", "bc5cdr", "cwlc")
CORPUS_LABEL = {"ade": "ADE-Corpus V2", "bc5cdr": "BC5CDR", "cwlc": "CWLC"}
UNIT_LABEL = {"ade": "sentence", "bc5cdr": "abstract", "cwlc": "sentence"}

# Tag layers per corpus. ADE and BC5CDR store one flat BIO sequence per entity type
# (the joint "tags" column can drop overlapping mentions, so it is not used here).
# CWLC layers are discovered from the "tags" dict of each record.
FIXED_LAYERS: dict[str, dict[str, str]] = {
    "ade": {"drug": "tags_drug", "effect": "tags_effect"},
    "bc5cdr": {"chemical": "tags_chemical", "disease": "tags_disease"},
}

MAX_MENTION_LEN = 8  # dictionary entries longer than this many tokens are ignored
RATE_GRID = (0.10, 0.25, 0.50, 0.75)  # min "entity rate" for a dictionary entry (tuned on val)
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)  # inverse regularisation strengths for the ADE classifier

# Black-and-white friendly encodings, one per split (fill gray level, hatch, line style)
SPLIT_FILL = {"train": "0.25", "val": "0.62", "test": "0.92"}
SPLIT_HATCH = {"train": "", "val": "////", "test": ".."}
SPLIT_LINE = {"train": "-", "val": "--", "test": ":"}


def project_root() -> Path:
    """Project root = parent of scripts/ when this file lives there, otherwise its own folder.

    Same rule as script 02, so both find data/processed whether the scripts sit in
    <root>/scripts/ or directly in <root>/ (e.g. the user's Tesis/Codigo folder).
    """
    here = Path(__file__).resolve().parent
    return here.parent if here.name.lower() == "scripts" else here


def log(message: str) -> None:
    """Progress message, flushed so it shows immediately on Windows consoles."""
    print(message, flush=True)


# =============================================================================
# I/O and integrity
# =============================================================================
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSON-lines file."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_manifest(processed_dir: Path, manifest_path: Path | None = None,
                    expected_path: Path | None = None) -> pd.DataFrame:
    """Check every file listed in the manifest against its recorded SHA-256.

    Paths in the manifest are relative to ``processed_dir`` (script 02 convention).
    If ``expected_path`` is given, its hashes must also agree with the manifest for
    the files both contain (useful to compare against a run on another machine).

    Returns a table (path, recorded, computed, ok, matches_expected). Raises
    ``RuntimeError`` if any file is missing or differs, so downstream numbers can
    never silently describe a different dataset.
    """
    manifest_path = manifest_path or processed_dir / "manifest_processed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: dict[str, str] = {}
    if expected_path is not None:
        other = json.loads(expected_path.read_text(encoding="utf-8"))
        expected = {e["path"]: e["sha256"] for entries in other["files"].values() for e in entries}

    rows = []
    for entries in manifest["files"].values():
        for entry in entries:
            path = processed_dir / entry["path"]
            computed = sha256_file(path) if path.exists() else "MISSING"
            rows.append({
                "path": entry["path"],
                "recorded": entry["sha256"],
                "computed": computed,
                "ok": computed == entry["sha256"],
                "matches_expected": (expected[entry["path"]] == entry["sha256"]) if entry["path"] in expected else None,
            })
    table = pd.DataFrame(rows)
    bad = table[(~table["ok"]) | (table["matches_expected"] == False)]  # noqa: E712 (None must not count)
    if not bad.empty:
        raise RuntimeError("Processed files do not match the manifest:\n" + bad[["path", "ok", "matches_expected"]].to_string())
    return table


def load_corpus(processed_dir: Path, corpus: str) -> dict[str, list[dict[str, Any]]] | None:
    """Load train/val/test records of one corpus, or None if the folder is absent."""
    folder = processed_dir / corpus
    if not all((folder / f"{s}.jsonl").exists() for s in SPLITS):
        return None
    return {s: read_jsonl(folder / f"{s}.jsonl") for s in SPLITS}


def layer_tags(corpus: str, records: Sequence[dict[str, Any]]) -> dict[str, list[list[str]]]:
    """Return {layer: [BIO tag sequence per record]} for a corpus (see FIXED_LAYERS)."""
    if corpus in FIXED_LAYERS:
        return {layer: [r[col] for r in records] for layer, col in FIXED_LAYERS[corpus].items()}
    names = sorted({name for r in records for name in r["tags"]})
    return {
        name: [r["tags"].get(name, ["O"] * len(r["tokens"])) for r in records]
        for name in names
    }


def cluster_ids(corpus: str, records: Sequence[dict[str, Any]]) -> list[str]:
    """Independence unit used for bootstrap resampling (PMID or document)."""
    if corpus in ("ade", "bc5cdr"):
        return [str(r["pmid"]) for r in records]
    return [str(r.get("doc_id") or r["sentence_id"]) for r in records]


# =============================================================================
# BIO helpers
# =============================================================================
def extract_spans(tags: Sequence[str]) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive)] of the mentions in one single-type BIO sequence.

    Lenient decoding: an ``I-`` that follows ``O`` (or starts the sequence) opens a new
    mention instead of being dropped, and ``B-`` always closes the previous mention.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, tag in enumerate(tags):
        if tag == "O":
            if start is not None:
                spans.append((start, i))
                start = None
        elif tag.startswith("B-") or start is None:
            if start is not None:
                spans.append((start, i))
            start = i
    if start is not None:
        spans.append((start, len(tags)))
    return spans


def mention_key(tokens: Sequence[str], span: tuple[int, int]) -> tuple[str, ...]:
    """Case-insensitive identity of a mention: lower-cased token tuple."""
    return tuple(t.lower() for t in tokens[span[0]:span[1]])


# =============================================================================
# Output helpers (CSV + LaTeX)
# =============================================================================
_LATEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                  "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def latex_escape(text: Any) -> str:
    """Escape LaTeX special characters in a cell."""
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in str(text))


def to_latex(df: pd.DataFrame, caption: str, label: str, float_digits: int = 3) -> str:
    """Small booktabs table writer (requires \\usepackage{booktabs} in the thesis).

    Floats are formatted with ``float_digits`` decimals, integers with thousands
    separators in braces ({,}) so LaTeX does not insert a space after the comma.
    """
    def cell(value: Any) -> str:
        if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
            return "--"
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}".replace(",", "{,}")
        if isinstance(value, (float, np.floating)):
            return f"{value:.{float_digits}f}"
        return latex_escape(value)

    # Keep the original column order: left-align non-numeric columns, right-align numeric ones
    align ="".join("r" if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) else "l" for c in df.columns)
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        rf"\caption{{{latex_escape(caption)}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}", r"\toprule",
        " & ".join(latex_escape(c) for c in df.columns) + r" \\", r"\midrule",
    ]
    lines += [" & ".join(cell(v) for v in row) + r" \\" for row in df.itertuples(index=False, name=None)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def save_table(df: pd.DataFrame, name: str, tables_dir: Path, caption: str, float_digits: int = 3) -> str:
    """Write ``name.csv`` and ``name.tex``; return the CSV path (relative name)."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / f"{name}.csv", index=False)
    (tables_dir / f"{name}.tex").write_text(to_latex(df, caption, f"tab:{name}", float_digits), encoding="utf-8")
    return f"{name}.csv"


# =============================================================================
# Bootstrap
# =============================================================================
def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of a bootstrap distribution (NaNs ignored)."""
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    alpha = (1 - level) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def cluster_bootstrap_indices(clusters: Sequence[str], n_boot: int, seed: int) -> Iterable[np.ndarray]:
    """Yield ``n_boot`` index arrays that resample whole clusters with replacement.

    Resampling clusters (PMIDs / documents) instead of rows keeps the within-cluster
    dependence in the interval; row-level resampling would make the CI too narrow.
    """
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    members = [[] for _ in range(inverse.max() + 1)]
    for row, cluster in enumerate(inverse):
        members[cluster].append(row)
    members = [np.asarray(m, dtype=np.int64) for m in members]
    for _ in range(n_boot):
        picked = rng.integers(0, len(members), size=len(members))
        yield np.concatenate([members[i] for i in picked])


def bootstrap_micro_prf(clusters: Sequence[str], tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
                        n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    """95% CI of micro precision/recall/F1 from per-unit counts, resampling clusters.

    Counts are pre-aggregated per cluster, so each replicate is one fancy-indexed sum.
    """
    uniq, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    agg = np.zeros((len(uniq), 3), dtype=np.int64)
    np.add.at(agg, inverse, np.stack([tp, fp, fn], axis=1))
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, 3))
    for b in range(n_boot):
        s_tp, s_fp, s_fn = agg[rng.integers(0, len(uniq), size=len(uniq))].sum(axis=0)
        p = s_tp / (s_tp + s_fp) if s_tp + s_fp else 0.0
        r = s_tp / (s_tp + s_fn) if s_tp + s_fn else 0.0
        out[b] = (p, r, 2 * p * r / (p + r) if p + r else 0.0)
    return {name: percentile_ci(out[:, i]) for i, name in enumerate(("precision", "recall", "f1"))}


# =============================================================================
# 1. Descriptive statistics
# =============================================================================
def describe_sizes(corpus: str, data: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """One row per split: units, tokens, length distribution and corpus-specific extras."""
    rows = []
    for split in SPLITS:
        records = data[split]
        lengths = np.array([len(r["tokens"]) for r in records])
        row: dict[str, Any] = {
            "corpus": corpus, "split": split, "unit": UNIT_LABEL[corpus], "n_units": len(records),
            "n_tokens": int(lengths.sum()), "len_mean": float(lengths.mean()),
            "len_median": float(np.median(lengths)), "len_p95": float(np.quantile(lengths, 0.95)),
            "len_max": int(lengths.max()),
        }
        if corpus == "ade":
            row["n_pmids"] = len({r["pmid"] for r in records})
            row["n_positive"] = int(sum(r["label"] for r in records))
            row["prevalence"] = row["n_positive"] / len(records)
            row["n_relations"] = int(sum(r["n_relations"] for r in records))
        elif corpus == "bc5cdr":
            row["n_cid_pairs"] = int(sum(len(r["cid_pairs"]) for r in records))
        else:
            row["pct_duplicate_within_split"] = 100 * float(np.mean([bool(r["duplicate_within_split"]) for r in records]))
            row["pct_seen_in_train"] = 100 * float(np.mean([bool(r["seen_in_train"]) for r in records])) if split != "train" else float("nan")
            row["n_docs_linked"] = len({r["doc_id"] for r in records if r.get("doc_id")})
        rows.append(row)
    return pd.DataFrame(rows)


INT_COLUMNS = ("n_units", "n_tokens", "len_max", "n_pmids", "n_positive", "n_relations", "n_cid_pairs", "n_docs_linked")


def per_corpus_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a stacked table by corpus, drop columns that are empty for that corpus and
    restore integer dtype (stacking corpora with different columns turns counts into floats)."""
    out = {}
    for corpus, part in df.groupby("corpus", sort=False):
        part = part.drop(columns="corpus").dropna(axis=1, how="all").reset_index(drop=True)
        for col in INT_COLUMNS:
            if col in part:
                part[col] = part[col].astype("Int64")
        out[corpus] = part
    return out


def describe_entities(corpus: str, data: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """One row per (split, layer): mention counts, density and span-length profile."""
    rows = []
    for split in SPLITS:
        records = data[split]
        n_tokens = sum(len(r["tokens"]) for r in records)
        for layer, seqs in layer_tags(corpus, records).items():
            spans = [extract_spans(tags) for tags in seqs]
            lengths = np.array([e - s for sp in spans for s, e in sp])
            keys = Counter(mention_key(r["tokens"], span) for r, sp in zip(records, spans) for span in sp)
            rows.append({
                "corpus": corpus, "split": split, "layer": layer,
                "n_mentions": int(lengths.size),
                "n_unique_mentions": len(keys),
                "mentions_per_1k_tokens": 1000 * lengths.size / n_tokens if n_tokens else float("nan"),
                "mean_len_tokens": float(lengths.mean()) if lengths.size else float("nan"),
                "pct_multi_token": 100 * float((lengths > 1).mean()) if lengths.size else float("nan"),
                "pct_units_with_mention": 100 * float(np.mean([bool(sp) for sp in spans])),
            })
    return pd.DataFrame(rows)


def unseen_mention_rates(corpus: str, data: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """Share of val/test mentions whose lower-cased string is not a gold mention in train.

    Reported at occurrence level (every mention counts once per appearance, as in
    micro-averaged F1) and at type level (distinct strings).
    """
    rows = []
    train_tags = layer_tags(corpus, data["train"])
    for layer in train_tags:
        train_keys = {
            mention_key(r["tokens"], span)
            for r, tags in zip(data["train"], train_tags[layer]) for span in extract_spans(tags)
        }
        for split in ("val", "test"):
            seqs = layer_tags(corpus, data[split])[layer]
            occ = [mention_key(r["tokens"], span) for r, tags in zip(data[split], seqs) for span in extract_spans(tags)]
            distinct = set(occ)
            rows.append({
                "corpus": corpus, "layer": layer, "split": split,
                "n_mentions": len(occ),
                "unseen_pct_occurrences": 100 * sum(k not in train_keys for k in occ) / len(occ) if occ else float("nan"),
                "n_distinct": len(distinct),
                "unseen_pct_distinct": 100 * sum(k not in train_keys for k in distinct) / len(distinct) if distinct else float("nan"),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 2. Baseline A - ADE sentence classifier (TF-IDF + logistic regression)
# =============================================================================
def normalise_text(text: str) -> str:
    """Whitespace-collapsed, lower-cased text used to detect train/test duplicates."""
    return re.sub(r"\s+", " ", text).strip().lower()


def make_text_model(c_value: float, seed: int):
    """TF-IDF (1-2 grams, sublinear tf) + class-balanced logistic regression.

    ``class_weight='balanced'`` because prevalence is ~20%; thresholds are still tuned
    on val, so the weighting only helps the optimiser, it does not fix the cut-off.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(C=c_value, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed)),
    ])


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Score threshold that maximises F1 on the given (validation) data."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """AUROC, average precision and P/R/F1 of the positive class at a fixed threshold."""
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    pred = (scores >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "avg_precision": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def fit_select_evaluate(train_texts: Sequence[str], train_y: np.ndarray, val_texts: Sequence[str], val_y: np.ndarray,
                        test_texts: Sequence[str], test_y: np.ndarray, seed: int,
                        c_grid: Sequence[float] = C_GRID, fixed_c: float | None = None):
    """Fit on train, choose C (by val average precision) and threshold (val F1), score test.

    Returns (model, chosen C, threshold, val scores, test scores). If ``fixed_c`` is given
    no C search is done (used by the leakage demo so scenarios differ only in the split).
    """
    from sklearn.metrics import average_precision_score

    best: tuple[float, float, Any, np.ndarray] | None = None
    for c_value in ([fixed_c] if fixed_c is not None else c_grid):
        model = make_text_model(c_value, seed).fit(train_texts, train_y)
        val_scores = model.decision_function(val_texts)
        ap = average_precision_score(val_y, val_scores)
        if best is None or ap > best[0]:
            best = (ap, c_value, model, val_scores)
    _, c_value, model, val_scores = best
    threshold = best_f1_threshold(val_y, val_scores)
    return model, c_value, threshold, val_scores, model.decision_function(test_texts)


def top_features(model, k: int = 20) -> pd.DataFrame:
    """Highest-weight n-grams for the positive and negative class (sanity check of the baseline)."""
    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    pos, neg = order[::-1][:k], order[:k]
    return pd.DataFrame({
        "rank": np.arange(1, k + 1),
        "ngram_positive": vocab[pos], "weight_positive": coef[pos],
        "ngram_negative": vocab[neg], "weight_negative": coef[neg],
    })


def ade_rows_like_proof_of_concept(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the naive 'one row per relation' table, used only to demonstrate leakage.

    Every positive sentence appears ``n_relations`` times (negatives once). A random
    split of this table puts copies of the same sentence on both sides of the split,
    which is exactly the leakage that script 02 removed with a group split.
    """
    rows: list[dict[str, Any]] = []
    for r in records:
        rows.extend([{"text": r["text"], "label": r["label"], "pmid": r["pmid"]}] * max(1, int(r["n_relations"])))
    return rows


def random_split_indices(labels: np.ndarray, fractions: Sequence[float], seed: int) -> dict[str, np.ndarray]:
    """Label-stratified random split with no grouping at all (the naive strategy)."""
    rng = np.random.default_rng(seed)
    parts: dict[str, list[int]] = {s: [] for s in SPLITS}
    for value in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == value))
        n_train, n_val = int(round(fractions[0] * idx.size)), int(round(fractions[1] * idx.size))
        parts["train"] += idx[:n_train].tolist()
        parts["val"] += idx[n_train:n_train + n_val].tolist()
        parts["test"] += idx[n_train + n_val:].tolist()
    return {s: np.array(sorted(v)) for s, v in parts.items()}


def duplicate_share(train_texts: Sequence[str], test_texts: Sequence[str], test_y: np.ndarray) -> dict[str, float]:
    """Percentage of test rows (all / positive) whose exact normalised text is also in train."""
    seen = {normalise_text(t) for t in train_texts}
    flags = np.array([normalise_text(t) in seen for t in test_texts])
    return {
        "pct_test_text_in_train": 100 * float(flags.mean()),
        "pct_test_positive_text_in_train": 100 * float(flags[test_y == 1].mean()) if (test_y == 1).any() else float("nan"),
    }


def run_ade_classifier(data: dict[str, list[dict[str, Any]]], seed: int, n_boot: int, n_repeats: int,
                       fractions: Sequence[float]) -> dict[str, pd.DataFrame]:
    """Baseline A + leakage demonstration. Returns the tables to save."""
    texts = {s: [r["text"] for r in data[s]] for s in SPLITS}
    y = {s: np.array([r["label"] for r in data[s]]) for s in SPLITS}

    model, c_value, threshold, val_scores, test_scores = fit_select_evaluate(
        texts["train"], y["train"], texts["val"], y["val"], texts["test"], y["test"], seed)
    point = classification_metrics(y["test"], test_scores, threshold)
    val_point = classification_metrics(y["val"], val_scores, threshold)

    # Cluster bootstrap over PMIDs on the test split (fixed model and threshold)
    clusters = cluster_ids("ade", data["test"])
    draws: dict[str, list[float]] = defaultdict(list)
    for idx in cluster_bootstrap_indices(clusters, n_boot, seed):
        if y["test"][idx].min() == y["test"][idx].max():
            continue  # degenerate resample (cannot happen at 20% prevalence; kept as a guard)
        for name, value in classification_metrics(y["test"][idx], test_scores[idx], threshold).items():
            draws[name].append(value)

    rows = []
    for name, value in point.items():
        lo, hi = percentile_ci(np.array(draws[name]))
        rows.append({"metric": name, "test": value, "ci95_low": lo, "ci95_high": hi, "val": val_point[name]})
    metrics = pd.DataFrame(rows)
    metrics.insert(0, "n_test", len(y["test"]))
    metrics["chosen_C"] = c_value
    metrics["threshold_from_val"] = threshold
    metrics["test_prevalence"] = float(y["test"].mean())

    # Leakage demonstration: same model family, same sizes, three ways of splitting.
    scenarios = []
    leak = duplicate_share(texts["train"], texts["test"], y["test"])
    scenarios.append({"scenario": "group_split_frozen", "repeat": 0, **classification_metrics(y["test"], test_scores, threshold), **leak,
                      "n_train": len(y["train"]), "n_test": len(y["test"])})

    unique_records = [r for s in SPLITS for r in data[s]]
    poc_rows = ade_rows_like_proof_of_concept(unique_records)
    tables = {
        "random_unique_sentences": ([r["text"] for r in unique_records], np.array([r["label"] for r in unique_records])),
        "random_relation_rows": ([r["text"] for r in poc_rows], np.array([r["label"] for r in poc_rows])),
    }
    for scenario, (all_texts, all_y) in tables.items():
        for repeat in range(n_repeats):
            idx = random_split_indices(all_y, fractions, seed + repeat)
            pick = lambda s, arr: [arr[i] for i in idx[s]]  # noqa: E731
            tr_t, va_t, te_t = (pick(s, all_texts) for s in SPLITS)
            tr_y, va_y, te_y = (all_y[idx[s]] for s in SPLITS)
            _, _, thr, _, te_scores = fit_select_evaluate(tr_t, tr_y, va_t, va_y, te_t, te_y, seed + repeat, fixed_c=c_value)
            scenarios.append({"scenario": scenario, "repeat": repeat, **classification_metrics(te_y, te_scores, thr),
                              **duplicate_share(tr_t, te_t, te_y), "n_train": len(tr_y), "n_test": len(te_y)})
    leakage = pd.DataFrame(scenarios)
    summary = (leakage.groupby("scenario", sort=False)
               .agg(n_runs=("repeat", "size"), n_train=("n_train", "mean"), n_test=("n_test", "mean"),
                    f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
                    avg_precision_mean=("avg_precision", "mean"), avg_precision_sd=("avg_precision", "std"),
                    pct_test_text_in_train=("pct_test_text_in_train", "mean"),
                    pct_test_positive_text_in_train=("pct_test_positive_text_in_train", "mean"))
               .reset_index())
    return {"metrics": metrics, "features": top_features(model), "leakage_runs": leakage, "leakage_summary": summary}


# =============================================================================
# 3. Baseline B - exact-match dictionary NER
# =============================================================================
def lowered(records: Sequence[dict[str, Any]]) -> list[list[str]]:
    """Lower-cased tokens per record (computed once and shared by all layers)."""
    return [[t.lower() for t in r["tokens"]] for r in records]


def first_token_index(keys: Iterable[tuple[str, ...]]) -> dict[str, list[int]]:
    """Map first token -> candidate lengths (descending) for fast longest-match lookups."""
    lengths: dict[str, set[int]] = defaultdict(set)
    for key in keys:
        lengths[key[0]].add(len(key))
    return {tok: sorted(ls, reverse=True) for tok, ls in lengths.items()}


def fit_dictionary(tokens_lc: Sequence[Sequence[str]], tag_seqs: Sequence[Sequence[str]],
                   max_len: int = MAX_MENTION_LEN) -> dict[tuple[str, ...], float]:
    """Learn {mention: entity rate} from train.

    entity rate = (times the string is a gold mention) / (times the string occurs in
    train text at all). It penalises ambiguous strings ("cold", "mg", "will") that are
    mostly NOT entities, so a min-rate threshold can drop them.
    """
    as_entity: Counter = Counter()
    for toks, tags in zip(tokens_lc, tag_seqs):
        for s, e in extract_spans(tags):
            if e - s <= max_len:
                as_entity[tuple(toks[s:e])] += 1
    index = first_token_index(as_entity)
    anywhere: Counter = Counter()
    for toks in tokens_lc:
        for i, tok in enumerate(toks):
            for length in index.get(tok, ()):
                if i + length > len(toks):
                    continue  # a window cut by the end of the unit is not an occurrence of a longer entry
                key = tuple(toks[i:i + length])
                if key in as_entity:
                    anywhere[key] += 1
    return {key: as_entity[key] / max(anywhere[key], as_entity[key]) for key in as_entity}


def predict_spans(toks: Sequence[str], keep: set[tuple[str, ...]], index: dict[str, list[int]]) -> list[tuple[int, int]]:
    """Greedy left-to-right longest-match tagging with the dictionary (no overlaps)."""
    spans, i = [], 0
    while i < len(toks):
        for length in index.get(toks[i], ()):
            # The bound matters: toks[i:i+length] is silently shortened at the end of the unit, and the
            # shortened tuple can equal a shorter entry, which would yield a span running past the end.
            if i + length <= len(toks) and tuple(toks[i:i + length]) in keep:
                spans.append((i, i + length))
                i += length
                break
        else:
            i += 1
    return spans


def score_dictionary(tokens_lc: Sequence[Sequence[str]], tag_seqs: Sequence[Sequence[str]],
                     rates: dict[tuple[str, ...], float], min_rate: float,
                     mask: np.ndarray | None = None) -> dict[str, Any]:
    """Entity-level exact-match counts per unit, with seen/unseen recall bookkeeping.

    ``mask`` (bool per unit) restricts the evaluation to a subset (e.g. novel sentences).
    A mention is *seen* if its string is a gold mention anywhere in train (regardless of
    the rate threshold), so the seen share is the recall ceiling of any dictionary.
    """
    keep = {k for k, v in rates.items() if v >= min_rate}
    index = first_token_index(keep)
    n = len(tokens_lc)
    tp, fp, fn = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    seen_total = seen_hit = unseen_total = unseen_hit = 0
    for u, (toks, tags) in enumerate(zip(tokens_lc, tag_seqs)):
        if mask is not None and not mask[u]:
            continue
        gold = set(extract_spans(tags))
        pred = set(predict_spans(toks, keep, index))
        tp[u], fp[u], fn[u] = len(gold & pred), len(pred - gold), len(gold - pred)
        for span in gold:
            hit = span in pred
            if tuple(toks[span[0]:span[1]]) in rates:
                seen_total, seen_hit = seen_total + 1, seen_hit + hit
            else:
                unseen_total, unseen_hit = unseen_total + 1, unseen_hit + hit
    return {"tp": tp, "fp": fp, "fn": fn, "seen_total": seen_total, "seen_hit": seen_hit,
            "unseen_total": unseen_total, "unseen_hit": unseen_hit}


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Micro precision, recall, F1 from counts."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def run_dictionary_ner(corpus: str, data: dict[str, list[dict[str, Any]]], n_boot: int, seed: int) -> pd.DataFrame:
    """Dictionary baseline for every layer of one corpus; threshold tuned on val only.

    ADE caveat: only the sentences that contain a drug-effect relation carry entity
    annotations; the 'negative' sentences have NO gold mentions even though they are full
    of drugs and effects. Scoring or fitting on them would count every correct dictionary
    hit as a false positive and dilute the entity rates. Therefore, for ADE the dictionary
    is fitted and tuned on positive sentences only, and the headline test variant
    ('positive_sentences') is restricted to them. The 'all_sentences' variant is kept as a
    diagnostic of how much precision collapses when unannotated sentences are included.

    Each row has ``primary=True`` for the variant that should be quoted in the thesis.
    """
    positive_only = corpus == "ade"
    fit_data = {s: ([r for r in data[s] if r["label"] == 1] if positive_only else data[s]) for s in SPLITS}
    lc_fit = {s: lowered(fit_data[s]) for s in SPLITS}
    tags_fit = {s: layer_tags(corpus, fit_data[s]) for s in SPLITS}
    # Evaluation views of the test split: (name -> units, mask or None)
    lc_test_all, tags_test_all = lowered(data["test"]), layer_tags(corpus, data["test"])
    clusters_test_all = np.asarray(cluster_ids(corpus, data["test"]))
    if corpus == "ade":
        variants: dict[str, np.ndarray | None] = {
            "positive_sentences": np.array([r["label"] == 1 for r in data["test"]]), "all_sentences": None}
    elif corpus == "cwlc":
        variants = {"all": None, "novel_sentences": np.array([not r["seen_in_train"] for r in data["test"]])}
    else:
        variants = {"all": None}
    primary_variant = next(iter(variants))

    rows = []
    for layer in tags_fit["train"]:
        rates = fit_dictionary(lc_fit["train"], tags_fit["train"][layer])
        # Tune the entity-rate threshold on val (micro-F1); ties go to the stricter threshold
        best_rate, best_f1 = RATE_GRID[0], -1.0
        for min_rate in RATE_GRID:
            sc = score_dictionary(lc_fit["val"], tags_fit["val"][layer], rates, min_rate)
            f1 = prf(sc["tp"].sum(), sc["fp"].sum(), sc["fn"].sum())[2]
            if f1 >= best_f1:
                best_rate, best_f1 = min_rate, f1
        evaluations = [("val", "all" if not positive_only else "positive_sentences", lc_fit["val"], tags_fit["val"][layer], None, np.asarray(cluster_ids(corpus, fit_data["val"])))]
        evaluations += [("test", name, lc_test_all, tags_test_all[layer], mask, clusters_test_all) for name, mask in variants.items()]
        for split, variant, toks, tag_seqs, mask, clusters in evaluations:
            sc = score_dictionary(toks, tag_seqs, rates, best_rate, mask)
            p, r, f1 = prf(sc["tp"].sum(), sc["fp"].sum(), sc["fn"].sum())
            n_gold = sc["seen_total"] + sc["unseen_total"]
            ceiling_recall = sc["seen_total"] / n_gold if n_gold else float("nan")
            row = {
                "corpus": corpus, "layer": layer, "split": split, "variant": variant,
                "primary": split == "test" and variant == primary_variant,
                "min_entity_rate": best_rate, "n_dictionary_entries": sum(v >= best_rate for v in rates.values()),
                "n_gold": n_gold, "precision": p, "recall": r, "f1": f1,
                "recall_seen": sc["seen_hit"] / sc["seen_total"] if sc["seen_total"] else float("nan"),
                "recall_unseen": sc["unseen_hit"] / sc["unseen_total"] if sc["unseen_total"] else float("nan"),
                "recall_ceiling": ceiling_recall,
                # best possible F1 for a dictionary: perfect precision, recall = seen share
                "f1_ceiling": 2 * ceiling_recall / (1 + ceiling_recall) if n_gold else float("nan"),
                "ci95_f1_low": float("nan"), "ci95_f1_high": float("nan"),
            }
            if split == "test" and n_boot > 0:
                sel = np.ones(len(sc["tp"]), bool) if mask is None else mask
                ci = bootstrap_micro_prf(clusters[sel], sc["tp"][sel], sc["fp"][sel], sc["fn"][sel], n_boot, seed)
                row["ci95_f1_low"], row["ci95_f1_high"] = ci["f1"]
            rows.append(row)
        log(f"    {corpus}/{layer}: min_rate={best_rate}, val F1={best_f1:.3f}")
    return pd.DataFrame(rows)


# =============================================================================
# 4. Figures (black-and-white safe)
# =============================================================================
def setup_matplotlib():
    """Return pyplot with thesis-friendly defaults (headless backend, 9 pt text, light grid)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global FuncFormatter, NullFormatter
    from matplotlib.ticker import FuncFormatter, NullFormatter

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.5, "axes.axisbelow": True,
        "hatch.linewidth": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300,
    })
    return plt


def save_figure(fig, name: str, figures_dir: Path) -> list[str]:
    """Save PNG (300 dpi) and PDF (vector); return the file names."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
        out.append(f"{name}.{ext}")
    return out


def grouped_bars(ax, categories: Sequence[str], series: dict[str, Sequence[float]], width_total: float = 0.8):
    """Draw grouped bars, one series per split, with gray fill + hatch + black edge."""
    n = len(series)
    width = width_total / n
    x = np.arange(len(categories))
    for i, (split, values) in enumerate(series.items()):
        ax.bar(x - width_total / 2 + width * (i + 0.5), values, width=width * 0.92, color=SPLIT_FILL[split],
               hatch=SPLIT_HATCH[split], edgecolor="black", linewidth=0.6, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30 if len(categories) > 3 else 0, ha="right" if len(categories) > 3 else "center")


def fig_length_ecdf(data_by_corpus: dict[str, dict[str, list[dict[str, Any]]]], figures_dir: Path) -> list[str]:
    """ECDF of unit length (tokens) per corpus; one line style per split."""
    plt = setup_matplotlib()
    corpora = list(data_by_corpus)
    fig, axes = plt.subplots(1, len(corpora), figsize=(2.3 * len(corpora) + 0.8, 2.6), squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        for split in SPLITS:
            lengths = np.sort([len(r["tokens"]) for r in data_by_corpus[corpus][split]])
            ax.step(lengths, np.arange(1, lengths.size + 1) / lengths.size, where="post", color="black", lw=1.3,
                    linestyle=SPLIT_LINE[split], label=split)
        ax.set_xscale("log")
        # Plain integer tick labels; minor ticks stay as unlabeled marks (scientific notation is unreadable here)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{CORPUS_LABEL[corpus]} ({UNIT_LABEL[corpus]}s)")
        ax.set_xlabel("Tokens per " + UNIT_LABEL[corpus])
    axes[0][0].set_ylabel("Cumulative proportion")
    axes[0][0].legend(title="Split", frameon=False, loc="lower right")
    fig.tight_layout()
    names = save_figure(fig, "fig_length_ecdf", figures_dir)
    plt.close(fig)
    return names


def _panel_widths(frames: dict[str, pd.DataFrame]) -> list[float]:
    """Relative panel widths proportional to the number of categories."""
    return [max(1.2, 0.55 * df["layer"].nunique() + 0.6) for df in frames.values()]


def fig_entity_density(entities: pd.DataFrame, figures_dir: Path) -> list[str]:
    """Mentions per 1,000 tokens per layer and split (shows whether splits are comparable)."""
    plt = setup_matplotlib()
    frames = {c: g for c, g in entities.groupby("corpus", sort=False)}
    widths = _panel_widths(frames)
    fig, axes = plt.subplots(1, len(frames), figsize=(sum(widths) + 0.6, 2.9), gridspec_kw={"width_ratios": widths}, squeeze=False)
    for ax, (corpus, df) in zip(axes[0], frames.items()):
        layers = list(dict.fromkeys(df["layer"]))
        grouped_bars(ax, layers, {s: [df[(df.split == s) & (df.layer == l)]["mentions_per_1k_tokens"].iloc[0] for l in layers] for s in SPLITS})
        ax.set_title(CORPUS_LABEL[corpus])
    axes[0][0].set_ylabel("Mentions per 1,000 tokens")
    axes[0][-1].legend(title="Split", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    names = save_figure(fig, "fig_entity_density", figures_dir)
    plt.close(fig)
    return names


def fig_unseen(unseen: pd.DataFrame, figures_dir: Path) -> list[str]:
    """Share of val/test mention occurrences not seen as gold mentions in train."""
    plt = setup_matplotlib()
    frames = {c: g for c, g in unseen.groupby("corpus", sort=False)}
    widths = _panel_widths(frames)
    fig, axes = plt.subplots(1, len(frames), figsize=(sum(widths) + 0.6, 2.9), gridspec_kw={"width_ratios": widths}, squeeze=False)
    for ax, (corpus, df) in zip(axes[0], frames.items()):
        layers = list(dict.fromkeys(df["layer"]))
        grouped_bars(ax, layers, {s: [df[(df.split == s) & (df.layer == l)]["unseen_pct_occurrences"].iloc[0] for l in layers] for s in ("val", "test")})
        ax.set_ylim(0, 100)
        ax.set_title(CORPUS_LABEL[corpus])
    axes[0][0].set_ylabel("Mention occurrences unseen in train (%)")
    # Legend above the plot area so it can never cover a bar (bars may reach 100%)
    axes[0][-1].legend(title="Split", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    names = save_figure(fig, "fig_unseen_mentions", figures_dir)
    plt.close(fig)
    return names


def fig_dictionary_baseline(ner: pd.DataFrame, figures_dir: Path) -> list[str]:
    """Test F1 of the dictionary (bars + CI) against its theoretical ceiling (diamonds)."""
    plt = setup_matplotlib()
    test = ner[ner.primary].reset_index(drop=True)
    labels = [f"{CORPUS_LABEL[c]}: {l}" for c, l in zip(test.corpus, test.layer)]
    x = np.arange(len(test))
    fig, ax = plt.subplots(figsize=(max(5.0, 0.55 * len(test) + 1.5), 3.4))
    err = np.vstack([test.f1 - test.ci95_f1_low, test.ci95_f1_high - test.f1]).clip(min=0)
    ax.bar(x, test.f1, width=0.6, color="0.7", edgecolor="black", linewidth=0.6, yerr=err, ecolor="black",
           error_kw={"elinewidth": 0.8, "capsize": 2}, label="Dictionary F1 (test, 95% CI)")
    ax.plot(x, test.f1_ceiling, linestyle="none", marker="D", markersize=5, markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=1.0, label="Ceiling of any exact-match dictionary")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Entity-level F1 (exact match)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    names = save_figure(fig, "fig_dictionary_baseline", figures_dir)
    plt.close(fig)
    return names


def fig_ade_leakage(summary: pd.DataFrame, metrics: pd.DataFrame, figures_dir: Path) -> list[str]:
    """(a) Positive-class F1 by splitting strategy; (b) share of test texts also present in train."""
    plt = setup_matplotlib()
    names_x = {"group_split_frozen": "Group split\n(frozen)", "random_unique_sentences": "Random\nsplit,\nunique\nsentences",
               "random_relation_rows": "Random\nsplit, one\nrow per\nrelation"}
    summary = summary.set_index("scenario").loc[[s for s in names_x if s in set(summary.scenario)]]
    fills, hatches = ["0.25", "0.62", "0.92"], ["", "////", ".."]
    f1_ci = metrics.set_index("metric").loc["f1"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.3, 3.2))
    x = np.arange(len(summary))
    for i, (scenario, row) in enumerate(summary.iterrows()):
        if scenario == "group_split_frozen":  # bootstrap 95% CI of the frozen split
            height, yerr = f1_ci["test"], [[f1_ci["test"] - f1_ci["ci95_low"]], [f1_ci["ci95_high"] - f1_ci["test"]]]
        else:  # +/- 1 SD across the random-split repeats
            sd = row.f1_sd if row.n_runs > 1 else 0.0
            height, yerr = row.f1_mean, [[sd], [sd]]
        ax_a.bar(i, height, width=0.65, color=fills[i], hatch=hatches[i], edgecolor="black", linewidth=0.6, yerr=yerr,
                 ecolor="black", error_kw={"elinewidth": 0.8, "capsize": 3})
        ax_a.text(i, height + 0.04, f"{height:.2f}", ha="center", va="bottom", fontsize=8)
        ax_b.bar(i, row.pct_test_positive_text_in_train, width=0.65, color=fills[i], hatch=hatches[i], edgecolor="black", linewidth=0.6)
        ax_b.text(i, row.pct_test_positive_text_in_train + 2, f"{row.pct_test_positive_text_in_train:.1f}", ha="center", va="bottom", fontsize=8)
    for ax in (ax_a, ax_b):
        ax.set_xticks(x)
        ax.set_xticklabels([names_x[s] for s in summary.index], fontsize=7.5)
    ax_a.set_ylim(0, 1.0)
    ax_a.set_ylabel("F1, positive class (test)")
    ax_a.set_title("(a) Reported performance")
    ax_b.set_ylim(0, 100)
    ax_b.set_ylabel("Positive test texts also in train (%)")
    ax_b.set_title("(b) Train/test duplication")
    fig.tight_layout()
    names = save_figure(fig, "fig_ade_leakage", figures_dir)
    plt.close(fig)
    return names


# =============================================================================
# Orchestration
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments (defaults follow the folder layout of scripts 01 and 02)."""
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--interim-dir", type=Path, default=root / "data" / "interim")
    parser.add_argument("--tables-dir", type=Path, default=root / "outputs" / "tables")
    parser.add_argument("--figures-dir", type=Path, default=root / "outputs" / "figures")
    parser.add_argument("--expected-manifest", type=Path, default=None,
                        help="second manifest_processed.json (e.g. from another machine) whose hashes must agree")
    parser.add_argument("--skip", nargs="*", default=[], choices=list(CORPORA))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap replicates (0 disables CIs)")
    parser.add_argument("--n-repeats", type=int, default=5, help="seeds for the random-split leakage demo")
    parser.add_argument("--no-baselines", action="store_true", help="descriptive statistics and figures only")
    parser.add_argument("--no-verify", action="store_true", help="skip the manifest hash check (not recommended)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run stage 2: verify -> describe -> baselines -> figures -> summary."""
    args = build_parser().parse_args(argv)
    summary: dict[str, Any] = {
        "script": "03_exploracion.py", "seed": args.seed, "n_boot": args.n_boot,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "outputs": {"tables": [], "figures": []}, "notes": [],
    }

    if not args.no_verify:
        log("[0] Verifying processed files against the manifest ...")
        check = verify_manifest(args.processed_dir, expected_path=args.expected_manifest)
        summary["manifest_check"] = check.drop(columns=["recorded"]).to_dict(orient="records")
        log(f"    {len(check)} files verified OK")
    else:
        summary["notes"].append("Manifest verification skipped (--no-verify).")

    data_by_corpus = {}
    for corpus in CORPORA:
        if corpus in args.skip:
            continue
        loaded = load_corpus(args.processed_dir, corpus)
        if loaded is None:
            log(f"    {corpus}: processed files not found, skipping")
            summary["notes"].append(f"{corpus}: not found in {args.processed_dir}")
            continue
        data_by_corpus[corpus] = loaded
    if not data_by_corpus:
        log("No processed corpus found. Run script 02 first.")
        return 1

    # --- 1. descriptive statistics --------------------------------------------------------
    log("[1] Descriptive statistics ...")
    sizes = pd.concat([describe_sizes(c, d) for c, d in data_by_corpus.items()], ignore_index=True)
    entities = pd.concat([describe_entities(c, d) for c, d in data_by_corpus.items()], ignore_index=True)
    unseen = pd.concat([unseen_mention_rates(c, d) for c, d in data_by_corpus.items()], ignore_index=True)
    for name, df, caption, digits in (
        ("eda_sizes", sizes, "Size and length of {c} by split.", 2),
        ("eda_entities", entities, "Gold mentions per layer and split in {c}.", 2),
        ("eda_unseen_mentions", unseen, "Share of validation/test mentions of {c} not seen as gold mentions in train.", 1),
    ):
        # Stacked CSV for analysis plus one compact table per corpus for the thesis text
        summary["outputs"]["tables"].append(save_table(df, name, args.tables_dir, caption.format(c="all corpora"), digits))
        for corpus, part in per_corpus_tables(df).items():
            summary["outputs"]["tables"].append(save_table(part, f"{name}_{corpus}", args.tables_dir, caption.format(c=CORPUS_LABEL[corpus]), digits))

    # --- 2/3. baselines -------------------------------------------------------------------
    ade_result: dict[str, pd.DataFrame] | None = None
    ner_frames: list[pd.DataFrame] = []
    if not args.no_baselines:
        if "ade" in data_by_corpus:
            log("[2] ADE classifier (TF-IDF + logistic regression) and leakage demo ...")
            fractions = _ade_fractions(args.processed_dir)
            ade_result = run_ade_classifier(data_by_corpus["ade"], args.seed, args.n_boot, args.n_repeats, fractions)
            for key, caption in (("metrics", "ADE sentence classifier, TF-IDF + logistic regression (test, 95\\% PMID-cluster bootstrap)."),
                                 ("features", "Highest-weight n-grams of the ADE classifier."),
                                 ("leakage_runs", "ADE leakage demonstration, every run."),
                                 ("leakage_summary", "ADE leakage demonstration: effect of the splitting strategy.")):
                summary["outputs"]["tables"].append(save_table(ade_result[key], f"baseline_ade_{key}", args.tables_dir, caption))
        log("[3] Dictionary NER baseline ...")
        for corpus, data in data_by_corpus.items():
            ner_frames.append(run_dictionary_ner(corpus, data, args.n_boot, args.seed))
        ner = pd.concat(ner_frames, ignore_index=True)
        summary["outputs"]["tables"].append(save_table(ner, "baseline_dictionary_ner", args.tables_dir,
                                                        "Exact-match dictionary baseline, entity-level micro scores.", 3))
    else:
        ner = None

    # --- 4. figures -----------------------------------------------------------------------
    log("[4] Figures ...")
    summary["outputs"]["figures"] += fig_length_ecdf(data_by_corpus, args.figures_dir)
    summary["outputs"]["figures"] += fig_entity_density(entities, args.figures_dir)
    summary["outputs"]["figures"] += fig_unseen(unseen, args.figures_dir)
    if ner is not None:
        summary["outputs"]["figures"] += fig_dictionary_baseline(ner, args.figures_dir)
    if ade_result is not None:
        summary["outputs"]["figures"] += fig_ade_leakage(ade_result["leakage_summary"], ade_result["metrics"], args.figures_dir)

    # --- key numbers for quick reading ----------------------------------------------------
    if ade_result is not None:
        summary["ade_classifier_test"] = ade_result["metrics"].set_index("metric")[["test", "ci95_low", "ci95_high"]].round(4).to_dict(orient="index")
        summary["ade_leakage"] = ade_result["leakage_summary"].round(4).to_dict(orient="records")
    if ner is not None:
        best = ner[ner.primary][["corpus", "layer", "variant", "f1", "recall_ceiling", "f1_ceiling"]]
        summary["dictionary_test"] = best.round(4).to_dict(orient="records")
    summary["unseen_test"] = unseen[unseen.split == "test"][["corpus", "layer", "unseen_pct_occurrences"]].round(2).to_dict(orient="records")
    args.interim_dir.mkdir(parents=True, exist_ok=True)
    (args.interim_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    log("\nDone. Tables: " + str(args.tables_dir) + " | Figures: " + str(args.figures_dir))
    if ade_result is not None:
        m = ade_result["metrics"].set_index("metric")
        log(f"ADE classifier test F1 = {m.loc['f1', 'test']:.3f} [{m.loc['f1', 'ci95_low']:.3f}, {m.loc['f1', 'ci95_high']:.3f}]")
    return 0


def _ade_fractions(processed_dir: Path) -> tuple[float, float, float]:
    """Split fractions used by script 02 (read from the manifest; default 67.5/7.5/25)."""
    try:
        manifest = json.loads((processed_dir / "manifest_processed.json").read_text(encoding="utf-8"))
        return tuple(manifest["parameters"]["ade"]["split_fractions"])  # type: ignore[return-value]
    except (OSError, KeyError, ValueError):
        return (0.675, 0.075, 0.25)


if __name__ == "__main__":
    sys.exit(main())
