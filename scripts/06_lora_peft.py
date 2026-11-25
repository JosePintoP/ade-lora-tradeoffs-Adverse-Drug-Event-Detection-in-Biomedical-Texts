#!/usr/bin/env python3
"""06 - LoRA (PEFT) versus full fine-tuning on the frozen splits (stage 3: modelling).

WHAT IT DOES
    Trains the same encoders as script 04 (same data, windows, casing, batch, epochs, seeds) but with LoRA
    adapters instead of updating every weight, then compares both regimes:

      * predictive performance  -> peft_vs_ft      (paired cluster bootstrap, Wilcoxon over seeds, effect size,
                                                      exploratory non-inferiority margins)       [H1]
      * computational efficiency -> peft_efficiency (trainable parameters, training time, peak VRAM)   [H2]

    LoRA settings follow the proposal (Table III): r = 8, lora_alpha = 32, lora_dropout = 0.1, adapters in the
    self-attention projections (query, value; the original LoRA recipe). The classification head is trained
    in full (it is new and randomly initialised, so it cannot be frozen).

THE LEARNING RATE IS NOT 2e-5
    Full fine-tuning uses 2e-5 (proposal, Table II). With LoRA that value barely moves the adapters in 3 epochs
    (the usual range is 1e-4 to 1e-3), and the proposal gives no LoRA rate. It is therefore chosen on the
    VALIDATION split, per task, from a small grid with ONE seed (stage "grid"); the remaining seeds then reuse it
    (stage "final"). Test is never used for the choice. Caveat: LoRA receives a tuned rate while full
    fine-tuning keeps the standard one, which slightly favours LoRA; keep this in the limitations.

STAGES (run them in this order; each one is cached, so an interrupted session resumes)
    python 06_lora_peft.py --stage grid                       # 3 rates x every task, seed 42
    python 06_lora_peft.py --stage final --seeds 42 43 44 45 46 47
    python 06_lora_peft.py --stage report                     # tables and figure only (no training)
    python 06_lora_peft.py --stage grid --dry-run             # print the plan

    Spanish backbone alternative (same flags as script 04, and they MUST match the full fine-tuning runs):
    python 06_lora_peft.py --stage final --corpora cwlc --model-es PlanTL-GOB-ES/bsc-bio-ehr-es --lowercase never --seeds 42 43 44 45 46 47

INPUTS   data/processed/*.jsonl (frozen), data/interim/sup_runs/*.json and predictions (script 04)
OUTPUTS  data/interim/peft_runs/*.json (+ data/interim/predictions/*_test.jsonl)
         outputs/tables/peft_runs.csv, peft_lr_selection.csv, peft_summary.csv/.tex, peft_vs_ft.csv/.tex,
         peft_efficiency.csv/.tex; outputs/figures/peft_delta_f1.{png,pdf}, peft_efficiency.{png,pdf}

NOTES
    * Efficiency numbers are only meaningful when nothing else uses the GPU during the run.
    * LoRA runs live in peft_runs/, so scripts 04 and 05 never mix them into the full fine-tuning tables.
    * The comparison drops any task whose full fine-tuning and LoRA runs differ in casing, window, batch or
      epochs, so a mismatched command line cannot silently produce a meaningless difference.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
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


sup = load_sibling("04_baseline_supervisado.py")  # training loop, data preparation, run storage
cmp = load_sibling("05_comparaciones_errores.py")  # paired bootstrap, Wilcoxon, Holm
eda = sup.eda
log = eda.log

PEFT_SUBDIR = "peft_runs"
DEFAULT_LR_GRID = (1e-4, 3e-4, 1e-3)
DEFAULT_TARGETS = ("query", "value")
NONINFERIORITY_MARGINS = (0.01, 0.02)  # exploratory (F1 points); the proposal fixes no margin, see peft_vs_ft notes
INITIALS = {"query": "q", "key": "k", "value": "v", "dense": "d"}
# Settings that must be identical in the full fine-tuning and LoRA runs of one comparison (lr differs by design).
COMPARABILITY_KEYS = ("lowercased", "max_len", "stride", "batch_size", "grad_accum", "epochs")
ALPHA = 0.05


# =============================================================================
# LoRA configuration
# =============================================================================
@dataclass(frozen=True)
class LoraSettings:
    """LoRA hyper-parameters (defaults = proposal, Table III)."""
    r: int = 8
    alpha: int = 32
    dropout: float = 0.1
    targets: tuple[str, ...] = DEFAULT_TARGETS

    @property
    def tag(self) -> str:
        """Short, file-name-safe description, e.g. 'lora-r8a32-qv'."""
        return f"lora-r{self.r}a{self.alpha}-{''.join(INITIALS.get(t, t) for t in self.targets)}"


def lr_tag(lr: float) -> str:
    """Learning rate as a compact token for run ids: 3e-4."""
    return np.format_float_scientific(lr, trim="-", exp_digits=1)


def make_wrapper(task: str, settings: LoraSettings, grad_checkpointing: bool = False) -> Callable[[Any], Any]:
    """Return the ``wrap_model`` hook of script 04: freeze the encoder and inject LoRA adapters.

    The classification head stays trainable (peft adds it to ``modules_to_save`` for TOKEN_CLS / SEQ_CLS).
    """
    from peft import LoraConfig, TaskType, get_peft_model

    def wrap(model):
        config = LoraConfig(task_type=TaskType.SEQ_CLS if task == "clf" else TaskType.TOKEN_CLS,
                            r=settings.r, lora_alpha=settings.alpha, lora_dropout=settings.dropout,
                            target_modules=list(settings.targets), bias="none")
        if grad_checkpointing:  # frozen embeddings otherwise give checkpointed blocks inputs without grad
            model.enable_input_require_grads()
        return get_peft_model(model, config)

    return wrap


def peft_run_id(corpus: str, layer: str, model_name: str, settings: LoraSettings, lr: float, seed: int) -> str:
    """Unique id of one LoRA run: {corpus}-{layer|clf}-{model}-{lora tag}-lr{lr}-s{seed}."""
    return f"{corpus}-{layer or 'clf'}-{sup.slug(model_name)}-{settings.tag}-lr{lr_tag(lr)}-s{seed}"


def model_label(model_name: str, settings: LoraSettings) -> str:
    return f"{model_name}+{settings.tag}"


# =============================================================================
# Reading finished runs
# =============================================================================
def read_rows(runs_dir: Path) -> pd.DataFrame:
    """First (primary test) row of every finished, non-smoke run in ``runs_dir``."""
    rows = []
    for path in sorted(runs_dir.glob("*.json")):
        if path.stem.endswith("-smoke"):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))["rows"][0]
        if row.get("status") == "ok":
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["layer"] = frame["layer"].fillna("-")
    return frame


def to_runs(frame: pd.DataFrame) -> list:
    """Rows -> script 05 ``Run`` objects (the id is enough to find the saved test predictions)."""
    runs = []
    for row in frame.itertuples(index=False):
        thr = getattr(row, "threshold_from_val", None)
        thr = None if thr is None or (isinstance(thr, float) and math.isnan(thr)) else float(thr)
        layer = None if row.layer == "-" else row.layer
        runs.append(cmp.Run(row.run_id, row.corpus, row.task, layer, row.model, int(row.seed), thr))
    return runs


def select_lrs(rows: pd.DataFrame, grid_seed: int, expected: Sequence[float]) -> pd.DataFrame:
    """Learning rate chosen on VALIDATION F1 for every (corpus, task, layer, backbone).

    Only the grid seed is used. Ties go to the smaller rate. ``complete`` is False when some rate of the grid
    is missing for that task (the choice would then be a comparison of unequal candidate sets).
    ``at_grid_edge`` warns that the optimum may lie outside the grid.
    """
    if rows.empty:
        return pd.DataFrame()
    out = []
    keys = ["corpus", "task", "layer", "backbone"]
    expected_sorted = sorted(float(x) for x in expected)
    for key, group in rows[rows["seed"] == grid_seed].groupby(keys):
        group = group.assign(lr=group["lr"].astype(float)).sort_values("lr")
        present = [lr for lr in expected_sorted if np.isclose(group["lr"], lr).any()]
        best = group.loc[group["val_f1"].idxmax()]  # first max = smallest lr
        for _, row in group.iterrows():
            chosen = bool(row["lr"] == best["lr"])
            out.append({**dict(zip(keys, key)), "lr": float(row["lr"]), "val_f1": float(row["val_f1"]),
                        "best_epoch": int(row["best_epoch"]), "chosen": chosen,
                        "complete": len(present) == len(expected_sorted),
                        "at_grid_edge": bool(chosen and (np.isclose(row["lr"], expected_sorted[0])
                                                         or np.isclose(row["lr"], expected_sorted[-1])))})
    return pd.DataFrame(out)


def final_rows(rows: pd.DataFrame, chosen_lr: dict[tuple, float]) -> pd.DataFrame:
    """Keep the rows trained with the chosen rate of their task."""
    keep = [np.isclose(r.lr, chosen_lr.get((r.corpus, r.task, r.layer, r.backbone), math.nan)) for r in rows.itertuples()]
    return rows[np.array(keep, dtype=bool)]


# =============================================================================
# Statistics: effect size and comparability guard
# =============================================================================
def rank_biserial(a: dict[int, float], b: dict[int, float]) -> float:
    """Matched-pairs rank-biserial correlation of A - B over the seeds both share.

    r = (W+ - W-) / (W+ + W-), W+ / W- = rank sums of the positive / negative differences (zeros dropped).
    +1: A is better in every seed with the larger gaps; 0: no systematic difference. NaN without differences.
    """
    from scipy.stats import rankdata

    common = sorted(set(a) & set(b))
    diff = np.round(np.array([a[s] - b[s] for s in common], dtype=float), 12)  # float noise must not break ties
    diff = diff[diff != 0]
    if diff.size == 0:
        return float("nan")
    ranks = rankdata(np.abs(diff))
    w_pos, w_neg = ranks[diff > 0].sum(), ranks[diff < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def comparable(ft: pd.DataFrame, lora: pd.DataFrame) -> tuple[bool, list[str]]:
    """True when both groups of runs share every setting in COMPARABILITY_KEYS (returns the offending keys)."""
    bad = []
    for key in COMPARABILITY_KEYS:
        values = {v for v in pd.concat([ft.get(key, pd.Series(dtype=object)), lora.get(key, pd.Series(dtype=object))]).dropna().tolist()}
        if len(values) > 1:
            bad.append(f"{key}={sorted(map(str, values))}")
    return not bad, bad


# =============================================================================
# Report tables
# =============================================================================
def build_summary(final: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds of the chosen-rate LoRA runs (same columns as sup_summary where they apply)."""
    if final.empty:
        return final
    keys = ["corpus", "task", "layer", "model", "lr"]
    for column in ("recall_unseen", "auroc", "n_trainable_m"):
        if column not in final:
            final = final.assign(**{column: np.nan})
    return (final.groupby(keys, dropna=False)
            .agg(n_seeds=("seed", "nunique"), precision=("precision", "mean"), recall=("recall", "mean"),
                 f1=("f1", "mean"), f1_sd=("f1", "std"), recall_unseen=("recall_unseen", "mean"), auroc=("auroc", "mean"),
                 n_trainable_m=("n_trainable_m", "first"), train_min=("train_time_s", lambda s: s.mean() / 60),
                 peak_vram_gb=("peak_vram_gb", "max"))
            .reset_index())


def compare_performance(final: pd.DataFrame, ft: pd.DataFrame, data: dict[str, Any], interim_dir: Path,
                        n_boot: int, seed: int) -> pd.DataFrame:
    """LoRA - full fine-tuning on the same backbone and task (paired on units, resampled by cluster)."""
    rows: list[dict[str, Any]] = []
    for (corpus, task, layer, backbone), lora_group in final.groupby(["corpus", "task", "layer", "backbone"]):
        ft_group = ft[(ft.corpus == corpus) & (ft.task == task) & (ft.layer == layer) & (ft.model == backbone)]
        name = f"{corpus}/{layer if layer != '-' else 'clf'}/{cmp.short_name(backbone)}"
        if ft_group.empty:
            log(f"  [skip] {name}: no full fine-tuning runs of this backbone in sup_runs")
            continue
        ok, bad = comparable(ft_group, lora_group)
        if not ok:
            log(f"  [skip] {name}: full fine-tuning and LoRA runs differ in {', '.join(bad)}; re-run with matching flags")
            continue
        lora_label = lora_group["model"].iloc[0]
        found = cmp.compare_two(corpus, task, None if layer == "-" else layer, lora_label, backbone,
                                to_runs(lora_group), to_runs(ft_group), data, interim_dir, n_boot, seed)
        f1_lora = dict(zip(lora_group["seed"].astype(int), lora_group["f1"].astype(float)))
        f1_ft = dict(zip(ft_group["seed"].astype(int), ft_group["f1"].astype(float)))
        for row in found:
            row["backbone"] = cmp.short_name(backbone)
            row["lr_lora"] = float(lora_group["lr"].iloc[0])
            row["rank_biserial"] = rank_biserial(f1_lora, f1_ft) if row["metric"] == "f1" else float("nan")
            rows.append(row)
    table = cmp.add_holm(pd.DataFrame(rows))
    if table.empty:
        return table
    is_f1 = table["metric"] == "f1"
    for margin in NONINFERIORITY_MARGINS:
        # The two-sided 95% bootstrap interval of the difference has a lower end that equals the one-sided 97.5% bound.
        table[f"noninferior_{int(margin * 100)}pt"] = pd.array(
            [bool(lo > -margin) if f else pd.NA for lo, f in zip(table["ci95_low"], is_f1)], dtype="boolean")
    return table


def compare_efficiency(final: pd.DataFrame, ft: pd.DataFrame) -> pd.DataFrame:
    """Trainable parameters, training time and peak VRAM of LoRA versus full fine-tuning (same backbone and task)."""
    rows: list[dict[str, Any]] = []
    for (corpus, task, layer, backbone), lora_group in final.groupby(["corpus", "task", "layer", "backbone"]):
        ft_group = ft[(ft.corpus == corpus) & (ft.task == task) & (ft.layer == layer) & (ft.model == backbone)]
        if ft_group.empty:
            continue
        ok, _ = comparable(ft_group, lora_group)
        if not ok:
            continue
        ft_trainable = float(ft_group["n_trainable_m"].iloc[0]) if "n_trainable_m" in ft_group and ft_group["n_trainable_m"].notna().any() \
            else float(ft_group["n_params_m"].iloc[0])  # runs made before script 04 logged it: all weights were trained
        lora_trainable = float(lora_group["n_trainable_m"].iloc[0])
        total = float(lora_group["n_params_m"].iloc[0])
        row: dict[str, Any] = {
            "corpus": corpus, "task": task, "layer": layer, "backbone": cmp.short_name(backbone),
            "n_seeds_lora": lora_group["seed"].nunique(), "n_seeds_ft": ft_group["seed"].nunique(),
            "trainable_m_lora": lora_trainable, "trainable_m_ft": ft_trainable,
            "pct_trainable": 100 * lora_trainable / ft_trainable,
            "adapter_mb_est": lora_trainable * 4,  # fp32 weights: 4 bytes per parameter (estimate, not a file size)
            "model_total_m_lora": total,
        }
        for column, label in (("train_time_s", "train_min"), ("peak_vram_gb", "peak_vram_gb")):
            scale = 60.0 if column == "train_time_s" else 1.0
            a = dict(zip(lora_group["seed"].astype(int), lora_group[column].astype(float) / scale))
            b = dict(zip(ft_group["seed"].astype(int), ft_group[column].astype(float) / scale))
            n_common, p_value, min_p = cmp.seed_wilcoxon(a, b)
            row.update({f"{label}_lora": float(np.mean(list(a.values()))), f"{label}_ft": float(np.mean(list(b.values()))),
                        f"{label}_ratio": float(np.mean(list(a.values())) / np.mean(list(b.values()))),
                        f"wilcoxon_p_{label}": p_value})
            row["n_common_seeds"], row["min_p_wilcoxon"] = n_common, min_p
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    for label in ("train_min", "peak_vram_gb"):  # one family of tests per metric (tasks x backbones)
        table[f"p_holm_{label}"] = cmp.holm_adjust(table[f"wilcoxon_p_{label}"].to_numpy())
    return table


# =============================================================================
# Figures
# =============================================================================
def fig_peft_delta(table: pd.DataFrame, figures_dir: Path) -> list[str]:
    """F1(LoRA) - F1(full fine-tuning) with its paired 95% interval. Filled marker = Holm-significant difference."""
    plt = eda.setup_matplotlib()
    frame = table[table["primary"]].reset_index(drop=True)
    frame["label"] = frame.apply(lambda r: f"{cmp.task_label(r)}\n({r['backbone']})", axis=1)
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(frame) + 1.6))
    for y, (_, row) in enumerate(frame.iterrows()):
        significant = bool(row["significant_holm"]) if pd.notna(row["significant_holm"]) else False
        ax.plot([row["ci95_low"], row["ci95_high"]], [y, y], color="black", linewidth=1.2, solid_capstyle="butt")
        ax.plot(row["delta"], y, marker="o", markersize=6.5, color="black",
                markerfacecolor="black" if significant else "white", markeredgewidth=1.2)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
    for margin, style in zip(NONINFERIORITY_MARGINS, (":", "-.")):
        ax.axvline(-margin, color="0.35", linestyle=style, linewidth=0.9)
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame["label"])
    ax.invert_yaxis()
    ax.set_xlabel("F1 difference: LoRA minus full fine-tuning (F1 points, 0-1 scale)")
    ax.plot([], [], "o", color="black", markerfacecolor="black", label="Holm-adjusted p < 0.05")
    ax.plot([], [], "o", color="black", markerfacecolor="white", label="not significant")
    ax.plot([], [], color="0.35", linestyle=":", label="-0.01 margin (exploratory)")
    ax.plot([], [], color="0.35", linestyle="-.", label="-0.02 margin (exploratory)")
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)
    saved = eda.save_figure(fig, "peft_delta_f1", figures_dir)
    plt.close(fig)
    return saved


def fig_peft_efficiency(table: pd.DataFrame, figures_dir: Path) -> list[str]:
    """Training time and peak VRAM of LoRA as a fraction of full fine-tuning (1 = same cost)."""
    plt = eda.setup_matplotlib()
    frame = table.reset_index(drop=True)
    frame["label"] = frame.apply(lambda r: f"{cmp.task_label(r)}\n({r['backbone']})", axis=1)
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(frame) + 1.6))
    for y, (_, row) in enumerate(frame.iterrows()):
        ax.plot(row["train_min_ratio"], y, marker="o", markersize=6.5, color="black", markerfacecolor="white", markeredgewidth=1.2)
        ax.plot(row["peak_vram_gb_ratio"], y, marker="s", markersize=6, color="black", markerfacecolor="black")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
    top = np.nanmax(frame[["train_min_ratio", "peak_vram_gb_ratio"]].to_numpy(dtype=float)) if len(frame) else 1.0
    ax.set_xlim(0, max(1.1, 1.05 * (top if np.isfinite(top) else 1.0)))  # a ratio axis starts at 0
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame["label"])
    ax.invert_yaxis()
    ax.set_xlabel("LoRA cost relative to full fine-tuning (ratio; 1 = same)")
    ax.plot([], [], "o", color="black", markerfacecolor="white", label="training time")
    ax.plot([], [], "s", color="black", markerfacecolor="black", label="peak VRAM")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    saved = eda.save_figure(fig, "peft_efficiency", figures_dir)
    plt.close(fig)
    return saved


# =============================================================================
# Report orchestration
# =============================================================================
def build_report(args, settings: LoraSettings) -> None:
    """Rebuild every table and figure from the runs on disk (no training)."""
    lora_all = read_rows(args.interim_dir / PEFT_SUBDIR)
    if lora_all.empty:
        log("No finished LoRA runs yet; nothing to report.")
        return
    lora_all = lora_all[lora_all["lora_tag"] == settings.tag]
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    lora_all.to_csv(args.tables_dir / "peft_runs.csv", index=False, encoding="utf-8-sig")

    selection = select_lrs(lora_all, args.grid_seed, args.lr_grid)
    if not selection.empty:
        selection.to_csv(args.tables_dir / "peft_lr_selection.csv", index=False, encoding="utf-8-sig")
    chosen: dict[tuple, float] = {}
    if args.lr is not None:  # an explicit rate overrides the grid for every task
        for r in lora_all[["corpus", "task", "layer", "backbone"]].drop_duplicates().itertuples(index=False):
            chosen[tuple(r)] = float(args.lr)
    elif not selection.empty:
        for r in selection[selection.chosen & selection.complete].itertuples():
            chosen[(r.corpus, r.task, r.layer, r.backbone)] = r.lr
    final = final_rows(lora_all, chosen)
    if final.empty:
        log("No LoRA runs with a selected learning rate yet (finish stage 'grid' first).")
        return
    summary = build_summary(final)
    summary.to_csv(args.tables_dir / "peft_summary.csv", index=False, encoding="utf-8-sig")
    compact = summary[["corpus", "layer", "model", "lr", "n_seeds", "precision", "recall", "f1", "f1_sd"]].copy()
    compact["model"] = compact["model"].str.split("/").str[-1]
    eda.save_table(compact, "peft_summary_thesis", args.tables_dir,
                   f"LoRA ({settings.tag}) on the frozen test splits: mean over seeds, exact-match entity F1 (clf: sentence F1).", 3)

    ft = read_rows(args.interim_dir / "sup_runs")
    if ft.empty:
        log("No full fine-tuning runs in sup_runs; the comparison with LoRA is skipped.")
        return
    corpora = sorted(final["corpus"].unique())
    data = {c: eda.load_corpus(args.processed_dir, c) for c in corpora}
    log("[report] LoRA vs full fine-tuning: performance")
    perf = compare_performance(final, ft, data, args.interim_dir, args.n_boot, args.boot_seed)
    if not perf.empty:
        perf.to_csv(args.tables_dir / "peft_vs_ft.csv", index=False, encoding="utf-8-sig")
        thesis = perf[perf["primary"]][["corpus", "layer", "backbone", "n_seeds_a", "n_seeds_b", "value_a", "value_b", "delta",
                                       "ci95_low", "ci95_high", "p_holm", "noninferior_1pt", "noninferior_2pt"]].copy()
        thesis = thesis.rename(columns={"value_a": "f1_lora", "value_b": "f1_ft", "n_seeds_a": "seeds_lora", "n_seeds_b": "seeds_ft"})
        eda.save_table(thesis, "peft_vs_ft_thesis", args.tables_dir,
                       "LoRA minus full fine-tuning (F1, paired cluster bootstrap 95% CI; Holm-adjusted p).", 3)
        saved = fig_peft_delta(perf, args.figures_dir)
        log(f"  figure: {', '.join(saved)}")
        m = int(perf["primary"].sum())
        n_seeds = int(perf["n_common_seeds"].min()) if m else 0
        if n_seeds >= 3:
            floor = 2.0 / 2 ** n_seeds
            if floor * m >= ALPHA:
                log(f"  NOTE: with {n_seeds} common seeds the smallest attainable Wilcoxon p is {floor:.5f}; multiplied by the "
                    f"{m} tests of the table (Holm) it cannot fall below {ALPHA}. Rely on the bootstrap columns, or use more seeds "
                    f"(the proposal targets N >= 10).")
        elif m:
            log(f"  NOTE: {n_seeds} common seeds; the Wilcoxon test needs at least 3 (the bootstrap columns are still valid).")
    log("[report] LoRA vs full fine-tuning: efficiency")
    eff = compare_efficiency(final, ft)
    if not eff.empty:
        eff.to_csv(args.tables_dir / "peft_efficiency.csv", index=False, encoding="utf-8-sig")
        compact_eff = eff[["corpus", "layer", "backbone", "pct_trainable", "train_min_lora", "train_min_ft", "train_min_ratio",
                           "peak_vram_gb_lora", "peak_vram_gb_ft", "peak_vram_gb_ratio"]]
        eda.save_table(compact_eff, "peft_efficiency_thesis", args.tables_dir,
                       "Training cost of LoRA versus full fine-tuning (means over seeds; same GPU, same precision).", 2)
        saved = fig_peft_efficiency(eff, args.figures_dir)
        log(f"  figure: {', '.join(saved)}")
    log(f"\nTables in {args.tables_dir}: peft_runs, peft_lr_selection, peft_summary, peft_vs_ft, peft_efficiency")


# =============================================================================
# Training orchestration
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    """Arguments of script 04 (so the commands match) plus the LoRA-specific ones."""
    p = sup.build_parser()
    p.description = __doc__
    p.set_defaults(lr=None)
    p.add_argument("--stage", choices=["grid", "final", "report"], default="final",
                   help="grid = pick the learning rate on validation; final = train the seeds; report = tables only")
    p.add_argument("--lr-grid", type=float, nargs="*", default=list(DEFAULT_LR_GRID), help="learning rates tried in stage grid")
    p.add_argument("--grid-seed", type=int, default=42, help="the single seed used to choose the learning rate")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--target-modules", nargs="*", default=list(DEFAULT_TARGETS),
                   help="attention projections that receive adapters (query key value; 'dense' adds the output/FFN layers)")
    p.add_argument("--figures-dir", type=Path, default=eda.project_root() / "outputs" / "figures")
    p.add_argument("--boot-seed", type=int, default=42, help="seed of the paired bootstrap")
    return p


def train_stage(args, settings: LoraSettings) -> int:
    """Run (or skip, when cached) every planned LoRA training; returns a process exit code."""
    if not args.no_verify and not args.dry_run:
        log("[0] Verifying processed files against the manifest ...")
        eda.verify_manifest(args.processed_dir)
    data = {c: eda.load_corpus(args.processed_dir, c) for c in args.corpora}
    missing = [c for c, d in data.items() if d is None]
    if missing:
        log(f"Processed files not found for: {missing}. Run script 02 first.")
        return 1
    available = {c: sorted(eda.FIXED_LAYERS[c]) if c in eda.FIXED_LAYERS else sorted({n for r in d["train"] for n in r["tags"]})
                 for c, d in data.items()}
    plan = sup.plan_runs(args, available)

    existing = read_rows(args.interim_dir / PEFT_SUBDIR)
    if not existing.empty:
        existing = existing[existing["lora_tag"] == settings.tag]
    selection = select_lrs(existing, args.grid_seed, args.lr_grid) if not existing.empty else pd.DataFrame()

    device = sup.torch.device("cuda" if sup.torch.cuda.is_available() else "cpu")
    amp_dtype = sup.pick_amp(device, args.amp)
    if device.type == "cuda":
        props = sup.torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}, {props.total_memory / 1024 ** 3:.1f} GB, autocast={amp_dtype}")
    else:
        log("WARNING: no CUDA GPU detected; training on CPU is very slow. Use --max-train-samples for a smoke test.")
    log(f"LoRA {settings.tag} (dropout {settings.dropout}); stage {args.stage}")

    jobs: list[tuple[dict[str, str], str, float, int]] = []  # (item, backbone, lr, seed)
    for item in plan:
        backbone = args.model_es if sup.CORPUS_LANG[item["corpus"]] == "es" else args.model_en
        if args.stage == "grid":
            jobs += [(item, backbone, float(lr), args.grid_seed) for lr in args.lr_grid]
            continue
        lr = args.lr
        if lr is None:
            pick = selection[(selection.corpus == item["corpus"]) & (selection.task == item["task"])
                             & (selection.layer == (item["layer"] or "-")) & (selection.backbone == backbone)
                             & selection.chosen & selection.complete] if not selection.empty else selection
            if pick.empty:
                log(f"  [skip] {item['corpus']}/{item['layer'] or 'clf'}: the learning-rate grid is missing or incomplete "
                    f"(run --stage grid first, or pass --lr)")
                continue
            lr = float(pick["lr"].iloc[0])
        jobs += [(item, backbone, float(lr), seed) for seed in args.seeds]
    log("Planned runs: " + ", ".join(f"{i['corpus']}/{i['layer'] or 'clf'}@{lr_tag(lr)}-s{s}" for i, _, lr, s in jobs))
    if args.dry_run:
        return 0

    tokenizers: dict[str, Any] = {}
    for item, backbone, lr, seed in jobs:
        corpus, task, layer = item["corpus"], item["task"], item["layer"]
        lowercase = args.lowercase == "always" or (args.lowercase == "auto" and corpus == "cwlc")
        window = (args.max_len or sup.WINDOW[corpus][0], sup.WINDOW[corpus][1])
        if args.max_len and window[1] >= window[0] - 2:
            window = (window[0], max(0, window[0] // 4))
        if backbone not in tokenizers:
            tokenizers[backbone] = sup.load_tokenizer(backbone)
        run_id = peft_run_id(corpus, layer, backbone, settings, lr, seed)
        if args.max_train_samples or args.max_eval_samples:
            run_id += "-smoke"
        result_path, _ = sup.run_paths(args.interim_dir, run_id, PEFT_SUBDIR)
        if result_path.exists() and not args.overwrite:
            log(f"[skip] {run_id} (already done)")
            continue
        log(f"[run] {run_id}  (lowercase={lowercase}, window={window})")
        cfg = sup.RunConfig(epochs=args.epochs, lr=lr, batch_size=args.batch_size, grad_accum=args.grad_accum,
                            max_len=window[0], stride=window[1], seed=seed, n_boot=args.n_boot,
                            grad_checkpointing=args.grad_checkpointing, amp=args.amp)
        meta = {"corpus": corpus, "task": task, "layer": layer or None, "lowercased": lowercase, "run_id": run_id,
                "peft": "lora", "lora_tag": settings.tag, "lora_r": settings.r, "lora_alpha": settings.alpha,
                "lora_dropout": settings.dropout, "lora_targets": ",".join(settings.targets), "backbone": backbone,
                "stage": args.stage}
        wrapper = make_wrapper(task, settings, args.grad_checkpointing)
        try:
            if task == "clf":
                splits = sup.build_clf_splits(data[corpus], args.max_train_samples, args.max_eval_samples)
                result = sup.run_classifier(splits["train"], splits["val"], splits["test"], tokenizers[backbone], backbone, cfg,
                                            device, amp_dtype, run_id, args.save_model, wrap_model=wrapper)
            else:
                train, val, test, variants = sup.build_ner_splits(corpus, layer, data[corpus], lowercase,
                                                                  args.max_train_samples, args.max_eval_samples)
                result = sup.run_ner(train, val, test, tokenizers[backbone], backbone, cfg, variants, device, amp_dtype,
                                     window, run_id, args.save_model, wrap_model=wrapper)
        except Exception as error:  # noqa: BLE001 - one failed run must not kill an unattended session
            if not sup.is_oom(error):
                raise
            sup.free_memory()
            log(f"  OUT OF MEMORY in {run_id}. Try --batch-size {max(1, args.batch_size // 2)} --grad-accum "
                f"{args.grad_accum * 2} (this changes the comparison with script 04: re-run those too) or --grad-checkpointing.")
            result = {"rows": [{"status": "oom", "model": backbone, "seed": seed, "split": "test", "primary": True, "lr": lr}],
                      "history": [], "predictions": []}
        for row in result["rows"]:
            row["model"] = model_label(backbone, settings)
        sup.save_run(args.interim_dir, run_id, meta, result, PEFT_SUBDIR)
        top = result["rows"][0]
        if top.get("status") == "ok":
            log(f"  -> val F1={top['val_f1']:.4f}  test F1={top['f1']:.4f}  trainable={top['n_trainable_m']:.2f}M "
                f"of {top['n_params_m']:.1f}M  peak VRAM={top['peak_vram_gb']:.2f} GB  time={top['train_time_s'] / 60:.1f} min")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Train according to --stage, then rebuild the report."""
    args = build_parser().parse_args(argv)
    settings = LoraSettings(args.lora_r, args.lora_alpha, args.lora_dropout, tuple(args.target_modules))
    if args.stage != "report":
        code = train_stage(args, settings)
        if code or args.dry_run:
            return code
    build_report(args, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
