"""07 - Update Tables III-V and Figs. 3-6 of the paper from finished runs (no training).

INPUTS
    outputs/tables/sup_runs.csv, peft_runs.csv, peft_lr_selection.csv   (scripts 04 and 06)
    data/processed/ade/{train,val,test}.jsonl                           (script 02)
    data/interim/predictions/<run_id>_test.jsonl                        (needed for the NER results;
                                                                         also double-checks classification)

OUTPUTS (same layout as the paper)
    outputs/tables/paper/table3_clf, table4_ner, table5_stats           (.csv and .tex)
    outputs/figures/paper/fig3 ... fig6                                 (.png and .pdf, "Blues" colour map)
    outputs/tables/paper/extra/                                         mean (SD), entity-level metrics,
                                                                        Holm / effect size, efficiency

DEFINITIONS
    * Tables III and IV and Figs. 3-6 describe one representative run (--seed, default 42).
    * Table V uses every seed shared by all available configurations; p values are uncorrected.
    * The two NER models (drug, effect) are combined into the five BIO labels of the paper, see
      merged_matrix(). A word annotated in both layers is counted once for each of its two labels,
      so the row totals of the matrices equal the label counts of Table I.
    * NER training time = drug model + effect model; peak memory = the larger of the two.
    * Memory is bytes / 1024**3 and is labelled "GB", as in the paper.
    * Configurations that were not run yet appear as "pending".

USAGE (from the folder that holds scripts 03 and 05)
    python 07_paper_results.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def load_sibling(filename: str):
    """Import a script that lives next to this one (file names start with a digit)."""
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} must be in the same folder as this script ({path.parent})")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


eda = load_sibling("03_exploracion.py")                # corpus loading, table and figure helpers
cmp = load_sibling("05_comparaciones_errores.py")      # exact Wilcoxon and Holm
log = eda.log

BACKBONES = ("BERT", "BioBERT")
STRATEGIES = ("FT", "LoRA")
LAYERS = ("drug", "effect")
CONFIGS = [(b, s) for s in STRATEGIES for b in BACKBONES]  # column order of Tables III and IV
COLUMN_NAMES = {c: f"{'Full fine-tuning' if c[1] == 'FT' else 'LoRA'} {c[0]}" for c in CONFIGS}
CLF_LABELS = ("0", "1")
NER_LABELS = ("O", "B-DRUG", "I-DRUG", "B-EFFECT", "I-EFFECT")
COMPARISONS = [  # row order of Table V
    (("BERT", "FT"), ("BioBERT", "FT")),
    (("BERT", "LoRA"), ("BioBERT", "LoRA")),
    (("BERT", "FT"), ("BERT", "LoRA")),
    (("BioBERT", "FT"), ("BioBERT", "LoRA")),
    (("BioBERT", "LoRA"), ("BERT", "FT")),
]
ENTITY_ROWS = [(f"{m} ({layer} entities)", f"entity_{k}_{layer}") for layer in LAYERS
               for m, k in (("F1-score", "f1"), ("Precision", "precision"), ("Recall", "recall"))]
BYTES_SAVED_PER_FROZEN_PARAM = 12  # gradient + two AdamW moments, 4 bytes each
NAVY = "#1f3a5f"

Matrices = dict[tuple[str, str, int], np.ndarray]


def config_label(config: tuple[str, str]) -> str:
    return f"{config[0]} {config[1]}"


# =============================================================================
# Loading and integrity checks
# =============================================================================
def backbone_name(model: str) -> str:
    """'dmis-lab/biobert-...+lora-...' -> 'BioBERT'; 'bert-base-cased' -> 'BERT'."""
    base = model.split("+")[0].split("/")[-1].lower()
    if base.startswith("biobert"):
        return "BioBERT"
    if base.startswith("bert-base"):
        return "BERT"
    return base


def read_runs_csv(path: Path) -> pd.DataFrame:
    """Primary test rows of the finished ADE runs of one results file."""
    # round_trip: the default parser can shift a saved float by one unit in the last place
    frame = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    keep = (frame["corpus"] == "ade") & (frame["split"] == "test") & (frame["status"] == "ok")
    frame = frame[keep & frame["primary"].astype(str).str.lower().eq("true")].copy()
    frame["layer"] = frame["layer"].fillna("-")
    return frame


def load_runs(tables_dir: Path) -> pd.DataFrame:
    """Full fine-tuning runs plus the LoRA runs trained with the validation-selected learning rate."""
    ft = read_runs_csv(tables_dir / "sup_runs.csv")
    ft["strategy"], ft["checkpoint"] = "FT", ft["model"]
    ft["n_trainable_m"] = ft["n_params_m"]  # every weight is trained

    lora = read_runs_csv(tables_dir / "peft_runs.csv")
    selection = pd.read_csv(tables_dir / "peft_lr_selection.csv", encoding="utf-8-sig")
    selection = selection[(selection["corpus"] == "ade") & selection["chosen"] & selection["complete"]]
    chosen = {(r.task, r.layer, r.backbone): r.lr for r in selection.itertuples()}
    keep = [np.isclose(r.lr, chosen.get((r.task, r.layer, r.backbone), np.nan)) for r in lora.itertuples()]
    lora = lora[np.array(keep, dtype=bool)].copy()
    lora["strategy"], lora["checkpoint"] = "LoRA", lora["backbone"]

    runs = pd.concat([ft, lora], ignore_index=True)
    runs["backbone_name"] = runs["checkpoint"].map(backbone_name)
    unknown = sorted(set(runs["backbone_name"]) - set(BACKBONES))
    if unknown:
        log(f"  [ignored] runs of other backbones: {unknown}")
        runs = runs[runs["backbone_name"].isin(BACKBONES)].copy()
    for (name, task), group in runs.groupby(["backbone_name", "task"]):
        if group["checkpoint"].nunique() > 1:
            raise ValueError(f"Several checkpoints are labelled {name} in task {task}: "
                             f"{sorted(group['checkpoint'].unique())}. Keep only one of them in the results files.")
    duplicated = runs.duplicated(["backbone_name", "strategy", "task", "layer", "seed"])
    if duplicated.any():
        raise ValueError(f"Duplicated runs: {runs.loc[duplicated, 'run_id'].tolist()}")
    return runs


def split_counts(data: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Sizes of the frozen splits that every run must agree with."""
    test_pos = [r for r in data["test"] if r["label"] == 1]
    return {
        "n_train_clf": len(data["train"]),
        "n_train_ner": sum(r["label"] == 1 for r in data["train"]),
        "n_test_pos": len(test_pos),
        "n_test_neg": len(data["test"]) - len(test_pos),
        "gold_drug": sum(len(eda.extract_spans(r["tags_drug"])) for r in test_pos),
        "gold_effect": sum(len(eda.extract_spans(r["tags_effect"])) for r in test_pos),
    }


def check_runs(runs: pd.DataFrame, counts: dict[str, int]) -> None:
    """Fail loudly if a run was trained or evaluated on other data, or with other settings."""
    for row in runs.itertuples():
        n_train = counts["n_train_clf"] if row.task == "clf" else counts["n_train_ner"]
        n_gold = counts["n_test_pos"] if row.task == "clf" else counts[f"gold_{row.layer}"]
        if int(row.n_train_units) != n_train or int(row.n_gold) != n_gold:
            raise ValueError(f"{row.run_id}: n_train={row.n_train_units} (expected {n_train}), "
                             f"n_gold={row.n_gold} (expected {n_gold}). This run does not match the current splits.")
    for column in ("max_len", "batch_size", "grad_accum", "epochs"):
        values = runs[column].dropna().unique()
        if len(values) > 1:
            raise ValueError(f"Runs differ in {column}: {sorted(values)}; they cannot be compared.")


def restrict_to_common_seeds(runs: pd.DataFrame, seeds: Sequence[int] | None) -> tuple[pd.DataFrame, list[int]]:
    """Use the same seeds in every available (backbone, strategy, task, layer) cell."""
    cells = runs.groupby(["backbone_name", "strategy", "task", "layer"])["seed"].apply(lambda s: set(s.astype(int)))
    common = set.intersection(*cells.tolist())
    if seeds:
        missing = set(seeds) - common
        if missing:
            raise ValueError(f"Seeds {sorted(missing)} are not available in every configuration.")
        common = set(seeds)
    if not common:
        raise ValueError(f"No seed is shared by all configurations:\n{cells}")
    dropped = set.union(*cells.tolist()) - common
    if dropped:
        log(f"  [warning] seeds {sorted(dropped)} are not available everywhere and are left out")
    return runs[runs["seed"].isin(common)].copy(), sorted(common)


# =============================================================================
# Confusion matrices
# =============================================================================
def clf_confusion(precision: float, recall: float, n_pos: int, n_neg: int) -> np.ndarray:
    """Exact [[TN, FP], [FN, TP]] rebuilt from positive-class precision and recall."""
    tp = int(round(recall * n_pos))
    predicted = int(round(tp / precision)) if precision > 0 else 0
    fp = predicted - tp
    if abs(tp / n_pos - recall) > 1e-9 or (predicted and abs(tp / predicted - precision) > 1e-9):
        raise ValueError(f"Cannot rebuild an integer confusion matrix (P={precision}, R={recall}, n_pos={n_pos})")
    return np.array([[n_neg - fp, fp], [n_pos - tp, tp]], dtype=int)


def check_clf_predictions(runs: pd.DataFrame, test: list[dict[str, Any]], pred_dir: Path,
                          counts: dict[str, int]) -> int:
    """Recount the confusion matrix from saved scores wherever they exist; returns the runs verified."""
    labels = {str(r.get("sentence_id") or r["pmid"]): int(r["label"]) for r in test}
    verified = 0
    for run in runs[runs["task"] == "clf"].itertuples():
        path = pred_dir / f"{run.run_id}_test.jsonl"
        if not path.exists():
            continue
        items = eda.read_jsonl(path)
        if sorted(str(i["unit_id"]) for i in items) != sorted(labels) or any(
                int(i["label"]) != labels[str(i["unit_id"])] for i in items):
            raise ValueError(f"{path.name}: ids or labels differ from the processed test split")
        # compare in float32, the precision of the scores when the threshold was applied in training
        scores = np.array([i["score"] for i in items], dtype=np.float32)
        threshold = np.float32(run.threshold_from_val)
        counted = np.zeros((2, 2), dtype=int)
        for item, positive in zip(items, scores >= threshold):
            counted[int(item["label"]), int(positive)] += 1
        rebuilt = clf_confusion(run.precision, run.recall, counts["n_test_pos"], counts["n_test_neg"])
        if not np.array_equal(counted, rebuilt):
            raise ValueError(
                f"{run.run_id}: confusion matrix from predictions differs from the results file. "
                f"[[TN, FP], [FN, TP]] from predictions={counted.tolist()}, from results={rebuilt.tolist()}; "
                f"threshold={float(threshold)!r}, test scores equal to it={int((scores == threshold).sum())}")
        verified += 1
    return verified


def spans_to_tags(spans: Sequence[Sequence[int]], n_tokens: int) -> np.ndarray:
    """0 = O, 1 = B, 2 = I."""
    tags = np.zeros(n_tokens, dtype=int)
    for start, end in spans:
        tags[start] = 1
        tags[start + 1:end] = 2
    return tags


def read_layer(path: Path, records: dict[str, dict[str, Any]],
               layer: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Word-level (gold, predicted) tags of one model, in the order of ``records``, plus entity P/R/F1."""
    items = {str(i["unit_id"]): i for i in eda.read_jsonl(path)}
    if set(items) != set(records):
        raise ValueError(f"{path.name}: sentence ids differ from the ADE-related test sentences")
    gold_all, pred_all = [], []
    tp = fp = fn = 0
    for uid, record in records.items():
        gold = {tuple(s) for s in items[uid]["gold"]}
        pred = {tuple(s) for s in items[uid]["pred"]}
        if gold != set(eda.extract_spans(record[f"tags_{layer}"])):
            raise ValueError(f"{path.name}: gold spans of {uid} differ from the processed test split")
        n = len(record["tokens"])
        gold_all.append(spans_to_tags(sorted(gold), n))
        pred_all.append(spans_to_tags(sorted(pred), n))
        tp, fp, fn = tp + len(gold & pred), fp + len(pred - gold), fn + len(gold - pred)
    return np.concatenate(gold_all), np.concatenate(pred_all), eda.prf(tp, fp, fn)


def merged_matrix(gold_d: np.ndarray, pred_d: np.ndarray, gold_e: np.ndarray,
                  pred_e: np.ndarray) -> tuple[np.ndarray, int]:
    """5x5 (true x predicted) matrix of two BIO layers coded 0 = O, 1 = B, 2 = I.

    * A word with a drug label is counted on that row: the predicted label is the drug model's; if it
      says O and the word is not also annotated as effect, the effect model's label is used.
    * Effect labels are treated the same way, with the roles exchanged.
    * A word outside both layers takes the drug model's label if it is not O, else the effect model's.

    Returns the matrix and the number of entity predictions that have no cell in it (both models
    predicted an entity on a word that is annotated in at most one layer).
    """
    effect_gold = np.where(gold_e > 0, gold_e + 2, 0)
    effect_pred = np.where(pred_e > 0, pred_e + 2, 0)
    matrix = np.zeros((5, 5), dtype=int)

    drug = gold_d > 0
    pred = np.where(pred_d > 0, pred_d, np.where(gold_e == 0, effect_pred, 0))
    np.add.at(matrix, (gold_d[drug], pred[drug]), 1)

    effect = gold_e > 0
    pred = np.where(pred_e > 0, effect_pred, np.where(gold_d == 0, pred_d, 0))
    np.add.at(matrix, (effect_gold[effect], pred[effect]), 1)

    outside = ~drug & ~effect
    pred = np.where(pred_d > 0, pred_d, effect_pred)
    np.add.at(matrix, (np.zeros(outside.sum(), dtype=int), pred[outside]), 1)

    hidden = int(((pred_d > 0) & (pred_e > 0) & ~(drug & effect)).sum())
    return matrix, hidden


def expected_row_totals(records: Sequence[dict[str, Any]]) -> np.ndarray:
    """Words per label as counted in Table I (a word in both layers counts once for each)."""
    totals = np.zeros(5, dtype=int)
    for record in records:
        for drug, effect in zip(record["tags_drug"], record["tags_effect"]):
            for tag in (drug, effect):
                if tag != "O":
                    totals[NER_LABELS.index(tag)] += 1
            totals[0] += drug == "O" and effect == "O"
    return totals


def own_model_precision(tags: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Precision of B-DRUG, I-DRUG, B-EFFECT, I-EFFECT counted on each model alone (all predictions)."""
    values = []
    for layer in LAYERS:
        gold, pred = tags[layer]
        values += [eda.prf(int(((pred == k) & (gold == k)).sum()), int(((pred == k) & (gold != k)).sum()), 0)[0]
                   for k in (1, 2)]
    return np.array(values)


def ner_matrices(runs: pd.DataFrame, test: list[dict[str, Any]], pred_dir: Path) -> tuple[Matrices, dict]:
    """Merged matrix of every NER run whose two prediction files exist.

    The second result holds, per run, the number of predictions without a cell in the matrix and the
    largest difference between the precision of an entity label in the matrix and in its own model.
    """
    records = {str(r.get("sentence_id") or r["pmid"]): r for r in test if r["label"] == 1}
    totals = expected_row_totals(list(records.values()))
    matrices: Matrices = {}
    hidden: dict[tuple[str, str, int], tuple[int, float]] = {}
    missing: list[str] = []
    for key, group in runs[runs["task"] == "ner"].groupby(["backbone_name", "strategy", "seed"]):
        by_layer = group.set_index("layer")
        if set(by_layer.index) != set(LAYERS):
            log(f"  [skip] {key}: NER needs both layers, found {list(by_layer.index)}")
            continue
        paths = {layer: pred_dir / f"{by_layer.loc[layer, 'run_id']}_test.jsonl" for layer in LAYERS}
        absent = [p.name for p in paths.values() if not p.exists()]
        if absent:
            missing += absent
            continue
        tags = {}
        for layer in LAYERS:
            gold, pred, entity = read_layer(paths[layer], records, layer)
            stored = by_layer.loc[layer, ["precision", "recall", "f1"]].to_numpy(dtype=float)
            if not np.allclose(entity, stored, atol=1e-9):
                raise ValueError(f"{paths[layer].name}: entity metrics differ from the results file")
            tags[layer] = (gold, pred)
        key = (key[0], key[1], int(key[2]))
        matrix, n_hidden = merged_matrix(*tags["drug"], *tags["effect"])
        if not np.array_equal(matrix.sum(axis=1), totals):
            raise ValueError(f"{key}: row totals {matrix.sum(axis=1)} differ from the label counts {totals}")
        merged = np.array([class_metrics(matrix, NER_LABELS)[f"precision_{x}"] for x in NER_LABELS[1:]])
        matrices[key], hidden[key] = matrix, (n_hidden, float(np.abs(merged - own_model_precision(tags)).max()))
    if missing:
        log(f"  [pending] {len(missing)} NER prediction files not found in {pred_dir}, e.g. {missing[0]}")
    return matrices, hidden


# =============================================================================
# Per-seed metrics
# =============================================================================
def class_metrics(matrix: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    """Precision, recall and F1 of every class of a (true x predicted) confusion matrix."""
    out: dict[str, float] = {}
    for k, label in enumerate(labels):
        tp = int(matrix[k, k])
        fp, fn = int(matrix[:, k].sum()) - tp, int(matrix[k, :].sum()) - tp
        out[f"precision_{label}"], out[f"recall_{label}"], out[f"f1_{label}"] = eda.prf(tp, fp, fn)
    return out


def seed_metrics(runs: pd.DataFrame, counts: dict[str, int], matrices: Matrices) -> pd.DataFrame:
    """Long table (backbone, strategy, task, seed, metric, value) used by every table."""
    rows: list[dict[str, Any]] = []
    for (backbone, strategy, task, seed), group in runs.groupby(["backbone_name", "strategy", "task", "seed"]):
        values: dict[str, float] = {
            "train_s": group["train_time_s"].sum(),
            "vram": group["peak_vram_gb"].max(),
            "trainable_params": round(group["n_trainable_m"].max() * 1e6),
        }
        if task == "clf":
            row = group.iloc[0]
            matrix = clf_confusion(row["precision"], row["recall"], counts["n_test_pos"], counts["n_test_neg"])
            values.update(class_metrics(matrix, CLF_LABELS))
        else:
            by_layer = group.set_index("layer")
            if set(by_layer.index) != set(LAYERS):
                continue
            for layer in LAYERS:
                for name in ("precision", "recall", "f1"):
                    values[f"entity_{name}_{layer}"] = by_layer.loc[layer, name]
            key = (backbone, strategy, int(seed))
            if key in matrices:
                values.update(class_metrics(matrices[key], NER_LABELS))
                values["f1_macro"] = float(np.mean([values[f"f1_{label}"] for label in NER_LABELS]))
        rows.extend({"backbone": backbone, "strategy": strategy, "task": task, "seed": int(seed),
                     "metric": m, "value": float(v)} for m, v in values.items())
    return pd.DataFrame(rows)


def values_by_seed(long: pd.DataFrame, config: tuple[str, str], task: str, metric: str) -> dict[int, float]:
    sel = long[(long.backbone == config[0]) & (long.strategy == config[1]) & (long.task == task) & (long.metric == metric)]
    return dict(zip(sel["seed"], sel["value"]))


# =============================================================================
# Tables III and IV (representative run) and their mean (SD) versions
# =============================================================================
def format_time(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d} min"


def table_rows(task: str) -> list[tuple[str, str]]:
    """(row label, metric key) in the order used by the paper."""
    labels = CLF_LABELS if task == "clf" else NER_LABELS
    name = (lambda x: f"class {x}") if task == "clf" else (lambda x: x)
    rows = [(f"{m} ({name(x)})", f"{k}_{x}")
            for m, k in (("F1-score", "f1"), ("Precision", "precision"), ("Recall", "recall")) for x in labels]
    return rows


COST_ROWS = [("Training time", "train_s"), ("Peak VRAM (training, allocated)", "vram")]


def paper_table(long: pd.DataFrame, task: str, seed: int) -> pd.DataFrame:
    """Table III (task 'clf') or IV (task 'ner') for the representative run."""
    rows = []
    for label, key in table_rows(task) + COST_ROWS:
        row = {"Metric": label}
        for config in CONFIGS:
            value = values_by_seed(long, config, task, key).get(seed)
            if value is None:
                row[COLUMN_NAMES[config]] = "pending"
            elif key == "train_s":
                row[COLUMN_NAMES[config]] = format_time(value)
            elif key == "vram":
                row[COLUMN_NAMES[config]] = f"{value:.2f} GB"
            else:
                row[COLUMN_NAMES[config]] = f"{value:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def mean_sd_table(long: pd.DataFrame, task: str, spec: Sequence[tuple[str, str]]) -> pd.DataFrame:
    """Same layout, with mean (SD) over all seeds; training time in minutes."""
    rows = []
    for label, key in spec:
        row = {"Metric": f"{label} [min]" if key == "train_s" else label}
        for config in CONFIGS:
            values = np.array(list(values_by_seed(long, config, task, key).values()))
            if key == "train_s":
                values = values / 60.0
            if values.size == 0:
                row[COLUMN_NAMES[config]] = "pending"
            else:
                sd = values.std(ddof=1) if values.size > 1 else float("nan")
                row[COLUMN_NAMES[config]] = f"{values.mean():.3f} ({sd:.3f})"
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Table V and complementary statistics
# =============================================================================
def table_v(long: pd.DataFrame) -> pd.DataFrame:
    """Table V as in the paper: F1 and peak VRAM, mean (SD) over seeds, uncorrected exact p values."""
    blocks = [("Binary classification", "clf", "f1_1"), ("Entity extraction", "ner", "f1_macro")]
    columns = ["Metric", "Model A", "Model B", "Mean A", "Mean B", "P value"]
    rows = []
    for title, task, f1_key in blocks:
        rows.append(dict.fromkeys(columns, "") | {"Metric": title})
        for metric, key in (("F1", f1_key), ("VRAM", "vram")):
            for a, b in COMPARISONS:
                va, vb = values_by_seed(long, a, task, key), values_by_seed(long, b, task, key)
                common = sorted(set(va) & set(vb))
                row = {"Metric": metric, "Model A": config_label(a), "Model B": config_label(b)}
                for column, values in (("Mean A", va), ("Mean B", vb)):
                    x = np.array([values[s] for s in sorted(values)])
                    if x.size == 0:
                        row[column] = "pending"
                    elif key == "vram":
                        row[column] = f"{x.mean():.2f} GB ({x.std(ddof=1):.3f} GB)"
                    else:
                        row[column] = f"{x.mean():.4f} ({x.std(ddof=1):.4f})"
                if not common:
                    row["P value"] = "pending"
                else:
                    p_value = cmp.seed_wilcoxon(va, vb)[1]
                    row["P value"] = "-" if np.isnan(p_value) else f"{p_value:.5f}"
                rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-)."""
    diff = np.round(diff, 12)
    diff = diff[diff != 0]
    if diff.size == 0:
        return float("nan")
    ranks = rankdata(np.abs(diff))
    w_pos, w_neg = ranks[diff > 0].sum(), ranks[diff < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def effect_size_table(long: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjusted p values (one family per metric group) and effect sizes for the Table V rows."""
    rows = []
    for family, task, key in (("performance", "clf", "f1_1"), ("performance", "ner", "f1_macro"),
                              ("memory", "clf", "vram"), ("memory", "ner", "vram")):
        for a, b in COMPARISONS:
            va, vb = values_by_seed(long, a, task, key), values_by_seed(long, b, task, key)
            common = sorted(set(va) & set(vb))
            if common:
                diff = np.array([va[s] - vb[s] for s in common])
                rows.append({"family": family, "task": task, "metric": key, "model_a": config_label(a),
                             "model_b": config_label(b), "n_pairs": len(common), "diff_a_minus_b": diff.mean(),
                             "p_wilcoxon": cmp.seed_wilcoxon(va, vb)[1], "rank_biserial": rank_biserial(diff)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = np.nan
    for family in table["family"].unique():
        mask = table["family"] == family
        table.loc[mask, "p_holm"] = cmp.holm_adjust(table.loc[mask, "p_wilcoxon"].to_numpy())
    return table


def efficiency_table(long: pd.DataFrame, seed: int) -> pd.DataFrame:
    """LoRA versus full fine-tuning cost (representative run and mean over seeds)."""
    rows = []
    for task in ("clf", "ner"):
        for backbone in BACKBONES:
            data = {(s, m): values_by_seed(long, (backbone, s), task, m)
                    for s in STRATEGIES for m in ("train_s", "vram", "trainable_params")}
            if any(not v for v in data.values()):
                continue
            row: dict[str, Any] = {"task": task, "backbone": backbone}
            for tag, pick in (("run", lambda v: v[seed]), ("mean", lambda v: float(np.mean(list(v.values()))))):
                for metric, name in (("train_s", "time"), ("vram", "vram")):
                    full, low = pick(data[("FT", metric)]), pick(data[("LoRA", metric)])
                    row[f"{name}_reduction_pct_{tag}"] = 100 * (1 - low / full)
                    if metric == "vram":
                        row[f"saving_gb_{tag}"] = full - low
            frozen = data[("FT", "trainable_params")][seed] - data[("LoRA", "trainable_params")][seed]
            row["saving_gb_predicted"] = BYTES_SAVED_PER_FROZEN_PARAM * frozen / 1024 ** 3
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Figures (same style as the paper, "Blues" colour map)
# =============================================================================
def setup_figures():
    """Pyplot with Matplotlib's default style (the one of the current figures)."""
    plt = eda.setup_matplotlib()
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "savefig.dpi": 300})
    return plt


def draw_matrix(fig, ax, matrix: np.ndarray, labels: Sequence[str], title: str, limits: tuple[int, int],
                corners: Sequence[Sequence[str]] | None = None, rotate: bool = False) -> None:
    """One annotated matrix with its colour bar; optional tag (TN, FP...) in the corner of each cell."""
    image = ax.imshow(matrix, cmap="Blues", vmin=limits[0], vmax=limits[1])
    light, dark = image.cmap(0.0), image.cmap(1.0)
    threshold = (limits[0] + limits[1]) / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = light if matrix[i, j] > threshold else dark
            ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", color=color, fontsize=12)
            if corners:
                ax.text(j + 0.44, i - 0.43, corners[i][j], ha="right", va="top", color=color, fontsize=10)
    ax.set_xticks(range(len(labels)), labels, rotation=45 if rotate else 0)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=13)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def fig_clf(runs: pd.DataFrame, counts: dict[str, int], seed: int, figures_dir: Path) -> list[str]:
    """Figs. 3 and 4: sentence classification, BERT above BioBERT."""
    plt = setup_figures()
    specs = (("FT", "fig3_cm_clf_full_ft", "Model Performance (Full Fine-Tuning)",
              {"BERT": "BERT base", "BioBERT": "BioBERT"}),
             ("LoRA", "fig4_cm_clf_lora", "Model Performance (LoRA Fine-Tuning)",
              {"BERT": "BERT with LoRA", "BioBERT": "BioBERT with LoRA"}))
    files: list[str] = []
    for strategy, filename, suptitle, titles in specs:
        sel = runs[(runs.task == "clf") & (runs.strategy == strategy) & (runs.seed == seed)].set_index("backbone_name")
        matrices = {b: clf_confusion(sel.loc[b, "precision"], sel.loc[b, "recall"], counts["n_test_pos"],
                                     counts["n_test_neg"]) for b in BACKBONES if b in sel.index}
        if not matrices:
            continue
        limits = (min(m.min() for m in matrices.values()), max(m.max() for m in matrices.values()))
        fig, axes = plt.subplots(len(matrices), 1, figsize=(6.4, 5.0 * len(matrices) + 0.8), squeeze=False,
                                 layout="constrained")
        for ax, (backbone, matrix) in zip(axes[:, 0], matrices.items()):
            draw_matrix(fig, ax, matrix, ("No ADE", "ADE"), f"Confusion Matrix - {titles[backbone]}", limits,
                        corners=(("TN", "FP"), ("FN", "TP")))
        fig.suptitle(suptitle, fontsize=18, fontweight="bold", color=NAVY)
        fig.supxlabel("Metrics: ADE vs No ADE Predictions", fontsize=10, color=NAVY)
        files += eda.save_figure(fig, filename, figures_dir)
        plt.close(fig)
    return files


def fig_ner(matrices: Matrices, seed: int, figures_dir: Path) -> list[str]:
    """Figs. 5 and 6: entity extraction, five BIO labels, BERT above BioBERT."""
    plt = setup_figures()
    specs = (("FT", "fig5_cm_ner_full_ft", None, {"BERT": "BERT Base", "BioBERT": "BioBERT"}),
             ("LoRA", "fig6_cm_ner_lora", "Model Performance (LoRA Fine-Tuning)",
              {"BERT": "LoRA BERT", "BioBERT": "LoRA BioBERT"}))
    files: list[str] = []
    for strategy, filename, suptitle, titles in specs:
        present = {b: matrices[(b, strategy, seed)] for b in BACKBONES if (b, strategy, seed) in matrices}
        if not present:
            continue
        limits = (0, max(m.max() for m in present.values()))
        fig, axes = plt.subplots(len(present), 1, figsize=(7.2, 6.0 * len(present) + (0.6 if suptitle else 0)),
                                 squeeze=False, layout="constrained")
        for ax, (backbone, matrix) in zip(axes[:, 0], present.items()):
            draw_matrix(fig, ax, matrix, NER_LABELS, f"Confusion Matrix - {titles[backbone]}", limits, rotate=True)
        if suptitle:
            fig.suptitle(suptitle, fontsize=18, fontweight="bold")
        files += eda.save_figure(fig, filename, figures_dir)
        plt.close(fig)
    return files


# =============================================================================
# Orchestration
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    root = eda.project_root()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tables-dir", type=Path, default=root / "outputs" / "tables", help="folder with sup_runs.csv etc.")
    p.add_argument("--processed-dir", type=Path, default=root / "data" / "processed")
    p.add_argument("--interim-dir", type=Path, default=root / "data" / "interim")
    p.add_argument("--out-tables", type=Path, default=root / "outputs" / "tables" / "paper")
    p.add_argument("--out-figures", type=Path, default=root / "outputs" / "figures" / "paper")
    p.add_argument("--seed", type=int, default=42, help="representative run of Tables III-IV and Figs. 3-6")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="seeds of Table V (default: all seeds shared by every configuration)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = eda.load_corpus(args.processed_dir, "ade")
    if data is None:
        log(f"ADE splits not found in {args.processed_dir / 'ade'}")
        return 1
    counts = split_counts(data)
    runs = load_runs(args.tables_dir)
    check_runs(runs, counts)
    runs, seeds = restrict_to_common_seeds(runs, args.seeds)
    present = sorted({(b, s) for b, s in zip(runs.backbone_name, runs.strategy)})
    log(f"Configurations: {[config_label(c) for c in present]}; seeds: {seeds}")
    pending = [config_label(c) for c in CONFIGS if c not in present]
    if pending:
        log(f"  [pending] no runs yet for: {pending}")
    if args.seed not in seeds:
        log(f"--seed {args.seed} is not among the available seeds {seeds}")
        return 1

    pred_dir = args.interim_dir / "predictions"
    log(f"Classification runs double-checked against saved predictions: "
        f"{check_clf_predictions(runs, data['test'], pred_dir, counts)}")
    matrices, hidden = ner_matrices(runs, data["test"], pred_dir)
    long = seed_metrics(runs, counts, matrices)

    out, extra = args.out_tables, args.out_tables / "extra"
    eda.save_table(paper_table(long, "clf", args.seed), "table3_clf", out,
                   "Performance, training time, and peak GPU memory for ADE binary classification (one run).")
    eda.save_table(paper_table(long, "ner", args.seed), "table4_ner", out,
                   "Performance, training time, and peak GPU memory for entity extraction (one run).")
    eda.save_table(table_v(long), "table5_stats", out,
                   f"Wilcoxon signed-rank test results; mean (SD) over {len(seeds)} runs.")

    eda.save_table(mean_sd_table(long, "clf", table_rows("clf") + COST_ROWS), "table3_clf_mean_sd", extra,
                   f"Classification: mean (SD) over {len(seeds)} seeds.")
    eda.save_table(mean_sd_table(long, "ner", table_rows("ner") + COST_ROWS), "table4_ner_mean_sd", extra,
                   f"Entity extraction, word-level labels: mean (SD) over {len(seeds)} seeds.")
    eda.save_table(mean_sd_table(long, "ner", ENTITY_ROWS), "table4_ner_entity_mean_sd", extra,
                   f"Entity extraction, exact-match entity metrics: mean (SD) over {len(seeds)} seeds.")
    eda.save_table(effect_size_table(long), "table5_effect_size", extra,
                   "Holm-adjusted p values and rank-biserial correlation for the comparisons of Table V.", 4)
    efficiency = efficiency_table(long, args.seed)
    eda.save_table(efficiency, "efficiency", extra, "LoRA versus full fine-tuning cost.", 2)
    long.to_csv(extra / "per_seed_metrics.csv", index=False)

    figures = fig_clf(runs, counts, args.seed, args.out_figures)
    figures += fig_ner(matrices, args.seed, args.out_figures)

    shown = {k: v for k, v in hidden.items() if k[2] == args.seed}
    if shown:
        log("\nCombination of the two NER models (representative run):")
        for (backbone, strategy, _), (n, gap) in shown.items():
            log(f"  {backbone} {strategy}: {n} predictions without a cell in the 5x5 matrix; "
                f"largest precision difference with the separate models: {gap:.4f}")
    if not efficiency.empty:
        log(f"\nLoRA vs full fine-tuning, representative run (seed {args.seed}):")
        log(efficiency[["task", "backbone", "time_reduction_pct_run", "vram_reduction_pct_run",
                        "saving_gb_run", "saving_gb_predicted"]].round(2).to_string(index=False))
    log(f"\nTables : {out}\nFigures: {args.out_figures} ({len(figures)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
