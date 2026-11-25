#!/usr/bin/env python3
"""Script 05 - Paired comparisons and error analysis (closes the supervised-baseline step of stage 3).

Script 04 produced one test F1 (with its own CI) per model, and script 03 one per lexical baseline.
Putting two intervals side by side is NOT a test: when they overlap (e.g. CWLC Medication, n = 150)
the question stays open, and when they do not, the comparison ignores that both systems were scored
on the very same sentences. This script answers the two follow-up questions with data that already exist:

1. Is the difference real?  (paired comparisons)
   * supervised model  vs  its lexical baseline (dictionary for NER, TF-IDF + LR for ADE classification)
   * model vs model on the same task (e.g. BETO vs bsc-bio-ehr-es on CWLC), whenever two models were run
   Paired cluster bootstrap: whole PMIDs / documents are resampled, and in every replicate BOTH systems
   are scored on the SAME resampled units, so the interval of the difference is much tighter than the
   difference of two independent intervals. Several comparisons are made, so a Holm correction is
   reported alongside the raw bootstrap p-value (multiplicity risk of the thesis plan).

2. Where does the model fail?  (error analysis)
   * Exact-match vs lenient (overlap) P/R/F1: if the lenient F1 is much higher, the errors are span
     BOUNDARIES; if it barely moves, they are missed/spurious mentions (a CLASS or vocabulary problem).
   * Every gold mention is classified exact / boundary-error / missed, and every predicted mention
     exact / boundary-error / spurious, for the dictionary and for the supervised model.
   * The most frequent missed and spurious strings, each with its "entity rate" in train (how often that
     string is annotated as this type when it occurs). Mid-range rates mean the annotation depends on
     context, which no word-level model can fix.

Inputs (nothing is trained here, no GPU, no torch)
--------------------------------------------------
* data/processed/**                       frozen files of script 02 (verified against the manifest)
* data/interim/sup_runs/*.json and data/interim/predictions/*_test.jsonl   written by script 04
* The dictionary and TF-IDF baselines are REBUILT here with the functions of script 03 (same protocol,
  same seed) because script 03 only saved their scores, not their predictions per unit. A consistency
  check against outputs/tables/baseline_*.csv warns if the rebuilt numbers differ.

Outputs
-------
* outputs/tables/cmp_vs_baseline.csv/.tex   supervised - baseline, per task
* outputs/tables/cmp_models.csv/.tex        model - model (only when >= 2 models were run on a task)
* outputs/tables/err_summary.csv/.tex       exact vs lenient scores and error-type shares
* outputs/tables/err_top_strings.csv        most frequent missed / spurious / boundary strings
* outputs/figures/fig_paired_delta, fig_model_delta (if applicable), fig_error_types  (PNG + PDF)

Statistical notes and limitations
---------------------------------
* The bootstrap p-value is two-sided, p = 2 * min(P(delta* <= 0), P(delta* >= 0)) with add-one
  smoothing, so its resolution is 1 / (n_boot + 1). It is an approximation; read it together with the
  interval. 'significant_holm' means p_holm < 0.05 within each table (the table is the family).
* The bootstrap captures TEST-SAMPLING variability only. With several seeds per model the compared
  quantity is the mean over seeds, which also averages part of the training noise, but the seeds are
  not resampled. For model-vs-model comparisons with >= 3 common seeds a paired exact Wilcoxon test over
  seeds is added, as in the thesis proposal. With n paired seeds the smallest possible two-sided p is
  2 / 2^n (n = 5 -> 0.0625, never below 0.05; n = 6 -> 0.03125); the column 'min_p_wilcoxon' shows it.
* Error analysis uses ONE seed (--error-seed, default: the lowest available) so that every count refers
  to one concrete set of predictions.
* ADE NER is conditional on positive sentences (same as scripts 03 and 04), so its comparisons say
  nothing about the end-to-end pipeline.

Usage
-----
    python 05_comparaciones_errores.py
    python 05_comparaciones_errores.py --n-boot 5000 --top-k 25
    python 05_comparaciones_errores.py --skip errors          # only the paired comparisons
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


def load_sibling(filename: str):
    """Import a script that lives next to this one (file names start with a digit, so no `import`)."""
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} must be in the same folder as this script ({path.parent})")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


# Single source of truth for BIO decoding, the dictionary and TF-IDF baselines, the bootstrap and table writers.
eda = load_sibling("03_exploracion.py")
log = eda.log

# Models treated as the reference in model-vs-model comparisons (delta = challenger - reference).
# Keep in sync with DEFAULT_ES_MODEL / DEFAULT_EN_MODEL of script 04 (not imported: it pulls in torch).
REFERENCE_MODELS = ("dccuchile/bert-base-spanish-wwm-uncased", "dmis-lab/biobert-base-cased-v1.1")
ALPHA = 0.05

# Black-and-white-safe fills for the error-type bars: (face colour, hatch)
OUTCOME_STYLE = {
    "exact": ("0.25", ""),
    "boundary": ("0.72", "////"),
    "missed": ("white", ""),
    "spurious": ("white", ""),
}


# =============================================================================
# Small statistics helpers
# =============================================================================
def f1_from_sums(sums: np.ndarray) -> np.ndarray:
    """F1 = 2tp / (2tp + fp + fn) from an array whose last axis holds (tp, fp, fn)."""
    tp, fp, fn = sums[..., 0], sums[..., 1], sums[..., 2]
    denom = 2.0 * tp + fp + fn
    return np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1.0), 0.0)


def bootstrap_p_value(delta_draws: np.ndarray) -> float:
    """Two-sided bootstrap p-value of 'no difference' with add-one smoothing (resolution 1/(B+1))."""
    draws = delta_draws[~np.isnan(delta_draws)]
    if draws.size == 0:
        return float("nan")
    le, ge = int((draws <= 0).sum()), int((draws >= 0).sum())
    return float(min(1.0, 2.0 * (1 + min(le, ge)) / (draws.size + 1)))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjustment (controls the family-wise error rate); NaNs are kept as NaN."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(order) - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def seed_wilcoxon(f1_a: dict[int, float], f1_b: dict[int, float]) -> tuple[int, float, float]:
    """Exact two-sided Wilcoxon signed-rank test over the seeds both models share.

    Returns (n common seeds, p-value, smallest p attainable with that many pairs). The p-value is NaN
    with fewer than 3 pairs or when every difference is zero.
    """
    from scipy.stats import wilcoxon

    common = sorted(set(f1_a) & set(f1_b))
    n = len(common)
    min_p = 2.0 / 2 ** n if n else float("nan")
    if n < 3:
        return n, float("nan"), min_p
    a, b = np.array([f1_a[s] for s in common]), np.array([f1_b[s] for s in common])
    if np.allclose(a, b):
        return n, float("nan"), min_p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return n, float(wilcoxon(a, b, method="exact").pvalue), min_p


def paired_bootstrap_counts(clusters: Sequence[str], counts_a: np.ndarray, counts_b: np.ndarray,
                            n_boot: int, seed: int) -> dict[str, Any]:
    """Paired cluster bootstrap of the micro-F1 difference A - B from per-unit (tp, fp, fn) counts.

    ``counts_a`` / ``counts_b`` have shape (n_seeds, n_units, 3); with several seeds the compared
    quantity is the mean F1 over seeds. Counts are pre-summed per cluster; every replicate draws the
    clusters once and applies the SAME draw to both systems (that is what makes it paired).
    """
    uniq, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = len(uniq)

    def per_cluster(counts: np.ndarray) -> np.ndarray:
        agg = np.zeros((n_clusters, counts.shape[0], 3))
        np.add.at(agg, inverse, counts.transpose(1, 0, 2))
        return agg

    agg_a, agg_b = per_cluster(counts_a), per_cluster(counts_b)
    rng = np.random.default_rng(seed)
    f_a, f_b = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        weights = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters)
        f_a[b] = f1_from_sums(np.tensordot(weights, agg_a, axes=(0, 0))).mean()
        f_b[b] = f1_from_sums(np.tensordot(weights, agg_b, axes=(0, 0))).mean()
    delta = f_a - f_b
    seed_a = f1_from_sums(counts_a.sum(axis=1))  # full-sample F1 of every seed
    seed_b = f1_from_sums(counts_b.sum(axis=1))
    return {
        "value_a": float(seed_a.mean()), "value_b": float(seed_b.mean()), "delta": float(seed_a.mean() - seed_b.mean()),
        "ci95_low": eda.percentile_ci(delta)[0], "ci95_high": eda.percentile_ci(delta)[1],
        "p_boot": bootstrap_p_value(delta), "n_clusters": n_clusters,
        "seed_delta_min": float((seed_a - seed_b.mean()).min()), "seed_delta_max": float((seed_a - seed_b.mean()).max()),
    }


def paired_bootstrap_scores(clusters: Sequence[str], y: np.ndarray, scores_a: np.ndarray, thr_a: np.ndarray,
                            scores_b: np.ndarray, thr_b: np.ndarray, n_boot: int, seed: int) -> dict[str, dict[str, Any]]:
    """Paired cluster bootstrap of F1 (at each system's own validation threshold) and AUROC for A - B.

    Scores have shape (n_seeds, n_units). Returns {'f1': stats, 'auroc': stats} in the same format
    as :func:`paired_bootstrap_counts`.
    """
    from sklearn.metrics import roc_auc_score

    def f1_of(scores: np.ndarray, thr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        pred, truth = scores[:, idx] >= thr[:, None], y[idx].astype(bool)
        tp = (pred & truth).sum(axis=1)
        fp = (pred & ~truth).sum(axis=1)
        fn = (~pred & truth).sum(axis=1)
        return f1_from_sums(np.stack([tp, fp, fn], axis=-1))

    def auroc_of(scores: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return np.array([roc_auc_score(y[idx], s[idx]) for s in scores])

    everything = np.arange(len(y))
    point = {"f1": (f1_of(scores_a, thr_a, everything), f1_of(scores_b, thr_b, everything)),
             "auroc": (auroc_of(scores_a, everything), auroc_of(scores_b, everything))}
    draws = {m: ([], []) for m in point}
    for idx in eda.cluster_bootstrap_indices(clusters, n_boot, seed):
        if y[idx].min() == y[idx].max():
            continue  # degenerate resample (cannot happen at ~20% prevalence; kept as a guard)
        draws["f1"][0].append(f1_of(scores_a, thr_a, idx).mean())
        draws["f1"][1].append(f1_of(scores_b, thr_b, idx).mean())
        draws["auroc"][0].append(auroc_of(scores_a, idx).mean())
        draws["auroc"][1].append(auroc_of(scores_b, idx).mean())
    n_clusters = len(np.unique(np.asarray(clusters)))
    out = {}
    for metric, (pa, pb) in point.items():
        delta = np.array(draws[metric][0]) - np.array(draws[metric][1])
        out[metric] = {
            "value_a": float(pa.mean()), "value_b": float(pb.mean()), "delta": float(pa.mean() - pb.mean()),
            "ci95_low": eda.percentile_ci(delta)[0], "ci95_high": eda.percentile_ci(delta)[1],
            "p_boot": bootstrap_p_value(delta), "n_clusters": n_clusters,
            "seed_delta_min": float((pa - pb.mean()).min()), "seed_delta_max": float((pa - pb.mean()).max()),
        }
    return out


# =============================================================================
# Span helpers: exact counts, lenient counts and error types
# =============================================================================
Span = tuple[int, int]


def overlaps(a: Span, b: Span) -> bool:
    """True when two half-open token spans share at least one token."""
    return a[0] < b[1] and b[0] < a[1]


def exact_counts(gold: Sequence[set[Span]], pred: Sequence[set[Span]]) -> np.ndarray:
    """Per-unit entity-level exact-match (tp, fp, fn), shape (n_units, 3)."""
    return np.array([[len(g & p), len(p - g), len(g - p)] for g, p in zip(gold, pred)], dtype=np.int64).reshape(-1, 3)


def classify_unit(gold: set[Span], pred: set[Span]) -> dict[str, int]:
    """Outcome of every gold and every predicted mention of one unit.

    gold:  exact (predicted identically) / boundary (overlaps a prediction but is not identical) / missed
    pred:  exact / boundary (overlaps a gold mention without matching it) / spurious (overlaps nothing)
    """
    exact = len(gold & pred)
    gold_boundary = sum(1 for g in gold - pred if any(overlaps(g, p) for p in pred))
    pred_boundary = sum(1 for p in pred - gold if any(overlaps(p, g) for g in gold))
    return {"gold_exact": exact, "gold_boundary": gold_boundary, "gold_missed": len(gold) - exact - gold_boundary,
            "pred_exact": exact, "pred_boundary": pred_boundary, "pred_spurious": len(pred) - exact - pred_boundary}


def prf_from(tp: float, n_pred: float, n_gold: float) -> tuple[float, float, float]:
    """Precision, recall and F1 from a number of hits, predictions and gold mentions."""
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# =============================================================================
# Data views aligned with script 04
# =============================================================================
@dataclass
class NerView:
    """One split of a single-type NER task, built exactly like ``build_ner_splits`` of script 04."""
    unit_ids: list[str]
    tokens_lc: list[list[str]]
    tags: list[list[str]]
    clusters: list[str]

    @property
    def gold(self) -> list[set[Span]]:
        return [set(eda.extract_spans(t)) for t in self.tags]


def ner_view(corpus: str, layer: str, records: Sequence[dict[str, Any]]) -> NerView:
    """Single-layer view; ADE keeps only positive sentences (negatives carry no entity annotation)."""
    if corpus == "ade":
        records = [r for r in records if r["label"] == 1]
    tags = eda.layer_tags(corpus, records)[layer]
    return NerView(
        unit_ids=[str(r.get("sentence_id") or r["pmid"]) for r in records],
        tokens_lc=eda.lowered(records), tags=[list(t) for t in tags], clusters=eda.cluster_ids(corpus, records))


@dataclass(frozen=True)
class Run:
    """One finished run of script 04."""
    run_id: str
    corpus: str
    task: str
    layer: str | None
    model: str
    seed: int
    threshold: float | None


def load_runs(interim_dir: Path) -> list[Run]:
    """Finished (status ok, not smoke) runs found in data/interim/sup_runs."""
    runs = []
    for path in sorted((interim_dir / "sup_runs").glob("*.json")):
        if path.stem.endswith("-smoke"):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))["rows"][0]
        if row.get("status") != "ok":
            continue
        threshold = row.get("threshold_from_val")
        runs.append(Run(row["run_id"], row["corpus"], row["task"], row.get("layer"), row["model"], int(row["seed"]),
                        None if threshold is None or (isinstance(threshold, float) and math.isnan(threshold)) else float(threshold)))
    return runs


def read_predictions(interim_dir: Path, run: Run) -> list[dict[str, Any]]:
    path = interim_dir / "predictions" / f"{run.run_id}_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Predictions of {run.run_id} not found ({path}); re-run script 04 with --overwrite.")
    return eda.read_jsonl(path)


def ner_predictions(interim_dir: Path, run: Run, view: NerView) -> list[set[Span]]:
    """Predicted spans of a run, after checking that the units and the gold spans match the frozen test split."""
    records = read_predictions(interim_dir, run)
    if [str(r["unit_id"]) for r in records] != view.unit_ids:
        raise ValueError(f"{run.run_id}: the units of the predictions differ from the frozen test split (was script 02 re-run?)")
    gold = [{tuple(s) for s in r["gold"]} for r in records]
    if gold != view.gold:
        raise ValueError(f"{run.run_id}: gold spans in the predictions differ from the frozen test split")
    return [{tuple(s) for s in r["pred"]} for r in records]


def short_name(model: str) -> str:
    return model.split("/")[-1]


# =============================================================================
# Lexical baselines rebuilt from script 03
# =============================================================================
@dataclass
class DictionarySystem:
    rates: dict[tuple[str, ...], float]
    min_rate: float
    val_f1: float
    test_pred: list[set[Span]]


def dictionary_system(corpus: str, layer: str, data: dict[str, list[dict[str, Any]]]) -> DictionarySystem:
    """Dictionary NER with the exact protocol of script 03: fit on train, min entity rate tuned on val."""
    views = {split: ner_view(corpus, layer, data[split]) for split in eda.SPLITS}
    rates = eda.fit_dictionary(views["train"].tokens_lc, views["train"].tags)
    best_rate, best_f1 = eda.RATE_GRID[0], -1.0
    for min_rate in eda.RATE_GRID:  # ties go to the stricter threshold, as in script 03
        sc = eda.score_dictionary(views["val"].tokens_lc, views["val"].tags, rates, min_rate)
        f1 = eda.prf(int(sc["tp"].sum()), int(sc["fp"].sum()), int(sc["fn"].sum()))[2]
        if f1 >= best_f1:
            best_rate, best_f1 = min_rate, f1
    keep = {k for k, v in rates.items() if v >= best_rate}
    index = eda.first_token_index(keep)
    test_pred = [set(eda.predict_spans(toks, keep, index)) for toks in views["test"].tokens_lc]
    return DictionarySystem(rates, best_rate, best_f1, test_pred)


def tfidf_system(data: dict[str, list[dict[str, Any]]], seed: int) -> tuple[np.ndarray, float]:
    """TF-IDF + LR scores on the ADE test split and the threshold tuned on val (script 03 protocol)."""
    texts = {s: [r["text"] for r in data[s]] for s in eda.SPLITS}
    y = {s: np.array([r["label"] for r in data[s]]) for s in eda.SPLITS}
    _, _, threshold, _, test_scores = eda.fit_select_evaluate(
        texts["train"], y["train"], texts["val"], y["val"], texts["test"], y["test"], seed)
    return np.asarray(test_scores), float(threshold)


def check_against_script_03(tables_dir: Path, corpus: str, layer: str, f1: float, kind: str) -> None:
    """Warn when a rebuilt baseline does not reproduce the F1 that script 03 saved."""
    path = tables_dir / ("baseline_dictionary_ner.csv" if kind == "dictionary" else "baseline_ade_metrics.csv")
    if not path.exists():
        return
    saved = pd.read_csv(path)
    if kind == "dictionary":
        row = saved[(saved.corpus == corpus) & (saved.layer == layer) & (saved.split == "test") & (saved.primary == True)]  # noqa: E712
        reference = float(row["f1"].iloc[0]) if len(row) else float("nan")
    else:
        row = saved[saved.metric == "f1"]
        reference = float(row["test"].iloc[0]) if len(row) else float("nan")
    if math.isnan(reference):
        return
    status = "OK" if abs(reference - f1) < 1e-6 else "WARNING: differs"
    log(f"    [check] {kind} {corpus}/{layer or 'clf'}: rebuilt F1={f1:.4f}, script 03 F1={reference:.4f} -> {status}")


# =============================================================================
# 1. Paired comparisons
# =============================================================================
def group_runs(runs: Sequence[Run]) -> dict[tuple[str, str, str | None, str], list[Run]]:
    """{(corpus, task, layer, model): runs sorted by seed}."""
    groups: dict[tuple[str, str, str | None, str], list[Run]] = {}
    for run in sorted(runs, key=lambda r: (r.corpus, r.task, r.layer or "", r.model, r.seed)):
        groups.setdefault((run.corpus, run.task, run.layer, run.model), []).append(run)
    return groups


def ner_counts_stack(interim_dir: Path, runs: Sequence[Run], view: NerView) -> np.ndarray:
    """(n_seeds, n_units, 3) exact-match counts of the runs of one model."""
    gold = view.gold
    return np.stack([exact_counts(gold, ner_predictions(interim_dir, r, view)) for r in runs])


def clf_scores_stack(interim_dir: Path, runs: Sequence[Run], unit_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(scores (S, n), thresholds (S,), labels (n,)) of the classification runs of one model."""
    scores, thresholds, labels = [], [], None
    for run in runs:
        records = read_predictions(interim_dir, run)
        if [str(r["unit_id"]) for r in records] != unit_ids:
            raise ValueError(f"{run.run_id}: units differ from the frozen ADE test split")
        scores.append([r["score"] for r in records])
        thresholds.append(run.threshold)
        labels = np.array([r["label"] for r in records])
    return np.array(scores, dtype=float), np.array(thresholds, dtype=float), labels


def comparison_row(corpus: str, task: str, layer: str | None, name_a: str, name_b: str, metric: str, primary: bool,
                   stats: dict[str, Any], n_units: int, n_seeds_a: int, n_seeds_b: int) -> dict[str, Any]:
    return {"corpus": corpus, "task": task, "layer": layer or "-", "system_a": name_a, "system_b": name_b,
            "metric": metric, "primary": primary, "n_units": n_units, "n_clusters": stats["n_clusters"],
            "n_seeds_a": n_seeds_a, "n_seeds_b": n_seeds_b, "value_a": stats["value_a"], "value_b": stats["value_b"],
            "delta": stats["delta"], "ci95_low": stats["ci95_low"], "ci95_high": stats["ci95_high"],
            "p_boot": stats["p_boot"], "seed_delta_min": stats["seed_delta_min"], "seed_delta_max": stats["seed_delta_max"]}


def add_holm(table: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjust the primary metric of every comparison in the table (the table is the family of tests)."""
    if table.empty:
        return table
    table = table.copy()
    table["p_holm"] = np.nan
    mask = table["primary"].to_numpy(bool)
    table.loc[mask, "p_holm"] = holm_adjust(table.loc[mask, "p_boot"].to_numpy())
    table["significant_holm"] = (table["p_holm"] < ALPHA).astype("boolean")  # nullable: secondary metrics stay empty
    table.loc[~mask, "significant_holm"] = pd.NA
    return table


def compare_vs_baselines(runs: Sequence[Run], data: dict[str, dict[str, list[dict[str, Any]]]], interim_dir: Path,
                         tables_dir: Path, n_boot: int, seed: int) -> pd.DataFrame:
    """Every supervised model against the lexical baseline of its task."""
    rows: list[dict[str, Any]] = []
    tfidf_cache: tuple[np.ndarray, float] | None = None
    dictionaries: dict[tuple[str, str], DictionarySystem] = {}
    for (corpus, task, layer, model), group in group_runs(runs).items():
        if task == "clf":
            if tfidf_cache is None:
                tfidf_cache = tfidf_system(data[corpus], seed)
            scores_b, thr_b = tfidf_cache
            test = data[corpus]["test"]
            unit_ids = [str(r["sentence_id"]) for r in test]
            scores_a, thr_a, labels = clf_scores_stack(interim_dir, group, unit_ids)
            if not np.array_equal(labels, np.array([r["label"] for r in test])):
                raise ValueError(f"{corpus}/clf: labels in the predictions differ from the frozen test split")
            y = labels
            pred_b = scores_b >= thr_b
            f1_b = f1_from_sums(np.array([(pred_b & (y == 1)).sum(), (pred_b & (y == 0)).sum(), (~pred_b & (y == 1)).sum()]))
            check_against_script_03(tables_dir, corpus, "-", float(f1_b), "tfidf")
            stats = paired_bootstrap_scores(eda.cluster_ids(corpus, test), y, scores_a, thr_a,
                                            scores_b[None, :], np.array([thr_b]), n_boot, seed)
            for metric, primary in (("f1", True), ("auroc", False)):
                rows.append(comparison_row(corpus, task, None, short_name(model), "tfidf_lr", metric, primary,
                                           stats[metric], len(y), len(group), 1))
            continue
        view = ner_view(corpus, layer, data[corpus]["test"])
        if (corpus, layer) not in dictionaries:
            dictionaries[(corpus, layer)] = dictionary_system(corpus, layer, data[corpus])
            sys_d = dictionaries[(corpus, layer)]
            check_against_script_03(tables_dir, corpus, layer, float(f1_from_sums(exact_counts(view.gold, sys_d.test_pred).sum(axis=0))), "dictionary")
        counts_b = exact_counts(view.gold, dictionaries[(corpus, layer)].test_pred)[None]
        counts_a = ner_counts_stack(interim_dir, group, view)
        stats = paired_bootstrap_counts(view.clusters, counts_a, counts_b, n_boot, seed)
        rows.append(comparison_row(corpus, task, layer, short_name(model), "dictionary", "f1", True, stats,
                                   len(view.unit_ids), len(group), 1))
    return add_holm(pd.DataFrame(rows))


def order_pair(model_x: str, model_y: str, references: Sequence[str]) -> tuple[str, str]:
    """(challenger, reference): the reference model is B so that delta = challenger - reference."""
    if model_x in references and model_y not in references:
        return model_y, model_x
    if model_y in references and model_x not in references:
        return model_x, model_y
    return tuple(sorted((model_x, model_y)))  # type: ignore[return-value]


def compare_two(corpus: str, task: str, layer: str | None, a_name: str, b_name: str, a_runs: Sequence[Run],
                b_runs: Sequence[Run], data: dict[str, dict[str, list[dict[str, Any]]]], interim_dir: Path,
                n_boot: int, seed: int) -> list[dict[str, Any]]:
    """Paired comparison A - B of two systems on one task (rows of cmp_models, before the Holm adjustment)."""
    if task == "clf":
        test = data[corpus]["test"]
        unit_ids = [str(r["sentence_id"]) for r in test]
        sa, ta, y = clf_scores_stack(interim_dir, a_runs, unit_ids)
        sb, tb, _ = clf_scores_stack(interim_dir, b_runs, unit_ids)
        stats = paired_bootstrap_scores(eda.cluster_ids(corpus, test), y, sa, ta, sb, tb, n_boot, seed)
        per_seed = lambda scores, thr: f1_from_sums(np.stack([  # noqa: E731
            ((scores >= thr[:, None]) & (y == 1)).sum(1), ((scores >= thr[:, None]) & (y == 0)).sum(1),
            ((scores < thr[:, None]) & (y == 1)).sum(1)], axis=-1))
        f1_a, f1_b = per_seed(sa, ta), per_seed(sb, tb)
        n_units, metrics = len(y), (("f1", True), ("auroc", False))
    else:
        view = ner_view(corpus, layer, data[corpus]["test"])
        ca, cb = ner_counts_stack(interim_dir, a_runs, view), ner_counts_stack(interim_dir, b_runs, view)
        stats = {"f1": paired_bootstrap_counts(view.clusters, ca, cb, n_boot, seed)}
        f1_a, f1_b = f1_from_sums(ca.sum(axis=1)), f1_from_sums(cb.sum(axis=1))
        n_units, metrics = len(view.unit_ids), (("f1", True),)
    n_common, p_w, min_p = seed_wilcoxon({r.seed: float(f) for r, f in zip(a_runs, f1_a)},
                                         {r.seed: float(f) for r, f in zip(b_runs, f1_b)})
    rows = []
    for metric, primary in metrics:
        row = comparison_row(corpus, task, layer, short_name(a_name), short_name(b_name), metric, primary,
                             stats[metric], n_units, len(a_runs), len(b_runs))
        row.update({"n_common_seeds": n_common, "wilcoxon_p": p_w if metric == "f1" else np.nan, "min_p_wilcoxon": min_p})
        rows.append(row)
    return rows


def compare_models(runs: Sequence[Run], data: dict[str, dict[str, list[dict[str, Any]]]], interim_dir: Path,
                   references: Sequence[str], n_boot: int, seed: int) -> pd.DataFrame:
    """Model vs model on the same task (only when at least two models have finished runs)."""
    by_task: dict[tuple[str, str, str | None], dict[str, list[Run]]] = {}
    for (corpus, task, layer, model), group in group_runs(runs).items():
        by_task.setdefault((corpus, task, layer), {})[model] = group
    rows: list[dict[str, Any]] = []
    for (corpus, task, layer), models in by_task.items():
        names = sorted(models)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_name, b_name = order_pair(names[i], names[j], references)
                rows += compare_two(corpus, task, layer, a_name, b_name, models[a_name], models[b_name],
                                    data, interim_dir, n_boot, seed)
    return add_holm(pd.DataFrame(rows))


# =============================================================================
# 2. Error analysis (NER)
# =============================================================================
def error_profile(view: NerView, pred: Sequence[set[Span]], rates: dict[tuple[str, ...], float],
                  top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Exact and lenient scores, outcome shares and the most frequent error strings of one system."""
    gold = view.gold
    totals: Counter = Counter()
    missed: Counter = Counter()
    spurious: Counter = Counter()
    boundary: Counter = Counter()
    for toks, g, p in zip(view.tokens_lc, gold, pred):
        totals.update(classify_unit(g, p))
        for span in g - p:
            if not any(overlaps(span, q) for q in p):
                missed[" ".join(toks[span[0]:span[1]])] += 1
        for span in p - g:
            overlapped = [s for s in g if overlaps(span, s)]
            if overlapped:
                nearest = max(overlapped, key=lambda s: min(span[1], s[1]) - max(span[0], s[0]))
                boundary[(" ".join(toks[nearest[0]:nearest[1]]), " ".join(toks[span[0]:span[1]]))] += 1
            else:
                spurious[" ".join(toks[span[0]:span[1]])] += 1
    n_gold, n_pred = sum(len(g) for g in gold), sum(len(p) for p in pred)
    p_ex, r_ex, f_ex = prf_from(totals["gold_exact"], n_pred, n_gold)
    p_le, r_le, f_le = prf_from(totals["pred_exact"] + totals["pred_boundary"], n_pred, n_gold)
    r_le = (totals["gold_exact"] + totals["gold_boundary"]) / n_gold if n_gold else 0.0
    f_le = 2 * p_le * r_le / (p_le + r_le) if p_le + r_le else 0.0

    def known_share(counter: Counter) -> float:  # share of the error occurrences whose string is gold somewhere in train
        total = sum(counter.values())
        return sum(c for s, c in counter.items() if tuple(s.split()) in rates) / total if total else float("nan")

    summary = {
        "n_gold": n_gold, "n_pred": n_pred, "precision_exact": p_ex, "recall_exact": r_ex, "f1_exact": f_ex,
        "precision_lenient": p_le, "recall_lenient": r_le, "f1_lenient": f_le, "f1_gap_boundary": f_le - f_ex,
        "gold_exact": totals["gold_exact"] / n_gold if n_gold else float("nan"),
        "gold_boundary": totals["gold_boundary"] / n_gold if n_gold else float("nan"),
        "gold_missed": totals["gold_missed"] / n_gold if n_gold else float("nan"),
        "pred_exact": totals["pred_exact"] / n_pred if n_pred else float("nan"),
        "pred_boundary": totals["pred_boundary"] / n_pred if n_pred else float("nan"),
        "pred_spurious": totals["pred_spurious"] / n_pred if n_pred else float("nan"),
        "missed_known_string_share": known_share(missed), "spurious_known_string_share": known_share(spurious),
    }
    strings: list[dict[str, Any]] = []
    for kind, counter in (("missed", missed), ("spurious", spurious)):
        for text, count in counter.most_common(top_k):
            strings.append({"kind": kind, "string": text, "predicted_as": "", "count": count,
                            "gold_in_train": tuple(text.split()) in rates,
                            "entity_rate_train": rates.get(tuple(text.split()), 0.0)})
    for (gold_text, pred_text), count in boundary.most_common(top_k):
        strings.append({"kind": "boundary", "string": gold_text, "predicted_as": pred_text, "count": count,
                        "gold_in_train": tuple(gold_text.split()) in rates,
                        "entity_rate_train": rates.get(tuple(gold_text.split()), 0.0)})
    return summary, strings


def error_analysis(runs: Sequence[Run], data: dict[str, dict[str, list[dict[str, Any]]]], interim_dir: Path,
                   error_seed: int | None, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Error profile of the dictionary and of every supervised model, per NER task."""
    summary_rows: list[dict[str, Any]] = []
    string_rows: list[dict[str, Any]] = []
    for (corpus, layer), group in _ner_groups(runs).items():
        view = ner_view(corpus, layer, data[corpus]["test"])
        dic = dictionary_system(corpus, layer, data[corpus])
        systems: list[tuple[str, Sequence[set[Span]]]] = [("dictionary", dic.test_pred)]
        for model, model_runs in group.items():
            chosen = next((r for r in model_runs if r.seed == error_seed), model_runs[0]) if error_seed is not None else model_runs[0]
            systems.append((f"{short_name(model)} (seed {chosen.seed})", ner_predictions(interim_dir, chosen, view)))
        for name, pred in systems:
            summary, strings = error_profile(view, pred, dic.rates, top_k)
            summary_rows.append({"corpus": corpus, "layer": layer, "system": name, **summary})
            string_rows += [{"corpus": corpus, "layer": layer, "system": name, **s} for s in strings]
    return pd.DataFrame(summary_rows), pd.DataFrame(string_rows)


def _ner_groups(runs: Sequence[Run]) -> dict[tuple[str, str], dict[str, list[Run]]]:
    groups: dict[tuple[str, str], dict[str, list[Run]]] = {}
    for (corpus, task, layer, model), group in group_runs(runs).items():
        if task == "ner":
            groups.setdefault((corpus, layer), {})[model] = group
    return groups


# =============================================================================
# 3. Figures (black-and-white safe)
# =============================================================================
def task_label(row: pd.Series) -> str:
    names = {"ade": "ADE", "bc5cdr": "BC5CDR", "cwlc": "CWLC"}
    return f"{names.get(row['corpus'], row['corpus'])} {'sentence clf' if row['task'] == 'clf' else row['layer']}"


def fig_delta(table: pd.DataFrame, figures_dir: Path, name: str, xlabel: str) -> list[str]:
    """Dot-and-whisker plot of the F1 difference with its paired 95% interval (filled marker = Holm-significant)."""
    plt = eda.setup_matplotlib()
    frame = table[table["primary"]].reset_index(drop=True)
    if "system_b" in frame:
        frame["label"] = frame.apply(lambda r: f"{task_label(r)}\n{r['system_a']} vs {r['system_b']}", axis=1)
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(frame) + 1.4))
    for y, (_, row) in enumerate(frame.iterrows()):
        significant = bool(row["significant_holm"]) if pd.notna(row["significant_holm"]) else False
        ax.plot([row["ci95_low"], row["ci95_high"]], [y, y], color="black", linewidth=1.2, solid_capstyle="butt")
        ax.plot(row["delta"], y, marker="o", markersize=6.5, color="black", markerfacecolor="black" if significant else "white",
                markeredgewidth=1.2)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame["label"])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.plot([], [], "o", color="black", markerfacecolor="black", label="Holm-adjusted p < 0.05")
    ax.plot([], [], "o", color="black", markerfacecolor="white", label="not significant")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    saved = eda.save_figure(fig, name, figures_dir)
    plt.close(fig)
    return saved


def fig_error_types(summary: pd.DataFrame, figures_dir: Path) -> list[str]:
    """Stacked bars: outcome of every gold mention (left) and of every predicted mention (right)."""
    plt = eda.setup_matplotlib()
    frame = summary.reset_index(drop=True)
    labels = [f"{r['corpus'].upper() if r['corpus'] != 'cwlc' else 'CWLC'} {r['layer']} - {r['system'].split(' (')[0]}"
              for _, r in frame.iterrows()]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 0.34 * len(frame) + 1.6), sharey=True)
    panels = (("Gold mentions", ("gold_exact", "gold_boundary", "gold_missed"), ("exact", "boundary", "missed")),
              ("Predicted mentions", ("pred_exact", "pred_boundary", "pred_spurious"), ("exact", "boundary", "spurious")))
    for ax, (title, columns, kinds) in zip(axes, panels):
        left = np.zeros(len(frame))
        for column, kind in zip(columns, kinds):
            face, hatch = OUTCOME_STYLE[kind]
            values = frame[column].fillna(0).to_numpy()
            ax.barh(range(len(frame)), values, left=left, color=face, hatch=hatch, edgecolor="black", linewidth=0.6,
                    height=0.78, label=kind)
            left += values
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of mentions")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    axes[0].set_yticks(range(len(frame)))
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    saved = eda.save_figure(fig, "fig_error_types", figures_dir)
    plt.close(fig)
    return saved


# =============================================================================
# Orchestration
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments."""
    root = eda.project_root()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=root / "data" / "processed")
    p.add_argument("--interim-dir", type=Path, default=root / "data" / "interim")
    p.add_argument("--tables-dir", type=Path, default=root / "outputs" / "tables")
    p.add_argument("--figures-dir", type=Path, default=root / "outputs" / "figures")
    p.add_argument("--corpora", nargs="*", default=["ade", "bc5cdr", "cwlc"], choices=["ade", "bc5cdr", "cwlc"])
    p.add_argument("--n-boot", type=int, default=2000, help="paired bootstrap replicates (p-value resolution = 1/(n+1))")
    p.add_argument("--seed", type=int, default=42, help="bootstrap seed and seed of the rebuilt TF-IDF baseline (as script 03)")
    p.add_argument("--reference-models", nargs="*", default=list(REFERENCE_MODELS),
                   help="models used as the reference (B) in model-vs-model comparisons")
    p.add_argument("--error-seed", type=int, default=None, help="training seed analysed in the error analysis (default: lowest)")
    p.add_argument("--top-k", type=int, default=15, help="strings listed per error kind")
    p.add_argument("--skip", nargs="*", default=[], choices=["paired", "errors", "figures"])
    p.add_argument("--no-verify", action="store_true", help="skip the manifest check of data/processed")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = [r for r in load_runs(args.interim_dir) if r.corpus in args.corpora]
    if not runs:
        log(f"No finished supervised runs found in {args.interim_dir / 'sup_runs'}. Run script 04 first.")
        return 1
    if not args.no_verify:
        log("[0] Verifying processed files against the manifest ...")
        eda.verify_manifest(args.processed_dir)
    data = {c: eda.load_corpus(args.processed_dir, c) for c in sorted({r.corpus for r in runs})}
    missing = [c for c, d in data.items() if d is None]
    if missing:
        log(f"Processed files not found for: {missing}. Run script 02 first.")
        return 1
    log(f"Runs found: {len(runs)} ({len({(r.corpus, r.task, r.layer, r.model) for r in runs})} model/task combinations)")

    vs_baseline = pd.DataFrame()
    models = pd.DataFrame()
    if "paired" not in args.skip:
        log("[1] Paired bootstrap: supervised vs lexical baseline ...")
        vs_baseline = compare_vs_baselines(runs, data, args.interim_dir, args.tables_dir, args.n_boot, args.seed)
        show = vs_baseline[vs_baseline["primary"]]
        for _, r in show.iterrows():
            log(f"    {task_label(r):<22} {r['system_a'][:18]:<18} vs {r['system_b']:<10} dF1={r['delta']:+.3f} "
                f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]  p_holm={r['p_holm']:.4f}")
        eda.save_table(vs_baseline, "cmp_vs_baseline", args.tables_dir,
                       "Supervised model minus lexical baseline on the test split (paired cluster bootstrap, 95\\% CI; "
                       "Holm correction across the table).", 3)
        log("[2] Paired bootstrap: model vs model ...")
        models = compare_models(runs, data, args.interim_dir, args.reference_models, args.n_boot, args.seed)
        if models.empty:
            log("    only one model per task so far: no model-vs-model table (run script 04 with another --model-es/--model-en).")
        else:
            for _, r in models[models["primary"]].iterrows():
                log(f"    {task_label(r):<22} {r['system_a'][:22]} vs {r['system_b'][:22]} dF1={r['delta']:+.3f} "
                    f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]  p_holm={r['p_holm']:.4f}")
            eda.save_table(models, "cmp_models", args.tables_dir,
                           "Model comparison on the test split (paired cluster bootstrap, 95\\% CI; Holm correction across the table).", 3)

    error_summary = pd.DataFrame()
    if "errors" not in args.skip:
        log("[3] Error analysis (exact vs lenient match, error types, frequent strings) ...")
        error_summary, strings = error_analysis(runs, data, args.interim_dir, args.error_seed, args.top_k)
        if not error_summary.empty:
            eda.save_table(error_summary, "err_summary", args.tables_dir,
                           "Exact vs lenient (overlap) scores and outcome of gold and predicted mentions on test.", 3)
            strings.to_csv(args.tables_dir / "err_top_strings.csv", index=False, encoding="utf-8-sig")
            for _, r in error_summary[error_summary["system"] != "dictionary"].iterrows():
                log(f"    {r['corpus']}/{r['layer']:<10} {r['system'][:34]:<34} F1 exact={r['f1_exact']:.3f} lenient={r['f1_lenient']:.3f}  "
                    f"missed={r['gold_missed']:.2f} boundary={r['gold_boundary']:.2f} spurious={r['pred_spurious']:.2f}")

    if "figures" not in args.skip:
        saved: list[str] = []
        if not vs_baseline.empty:
            saved += fig_delta(vs_baseline, args.figures_dir, "fig_paired_delta", r"$\Delta$F1 (supervised $-$ lexical baseline)")
        if not models.empty:
            saved += fig_delta(models, args.figures_dir, "fig_model_delta", r"$\Delta$F1 (challenger $-$ reference)")
        if not error_summary.empty:
            saved += fig_error_types(error_summary, args.figures_dir)
        log(f"Figures in {args.figures_dir}: " + ", ".join(saved))
    log(f"Tables in {args.tables_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
