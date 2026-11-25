#!/usr/bin/env python3
"""Script 04 - Supervised baselines: full fine-tuning of small encoders on a 4 GB GPU.

Stage 3 (modelling), first step. This is the "serious floor" that LoRA / PEFT and the
LLM-based pseudo-labelling will later be compared against. It reads ONLY the frozen files
of script 02 (data/processed/**), verifies them against the manifest, and writes results
to outputs/tables and data/interim.

Experiments (one model per row, never mixing corpora, as agreed in the thesis design)
------------------------------------------------------------------------------------
* ADE-Corpus V2, classification (``clf``): does the sentence contain a drug-effect relation?
  Sequence-classification head, threshold tuned on val (same protocol as the TF-IDF baseline
  of script 03, so the numbers are comparable).
* NER (``ner``), one flat BIO tagger per entity type (nesting in CWLC is across types, so
  one model per type avoids it):
    - ADE: drug, effect          (trained/evaluated on POSITIVE sentences only, because the
                                  negative sentences carry no entity annotation at all)
    - BC5CDR: chemical, disease  (abstracts, so long documents are windowed)
    - CWLC: Medication, Disease, Finding (the three types of the full design)
  Metric: entity-level exact-match micro P/R/F1, the same definition as the dictionary
  baseline, with a cluster bootstrap CI (PMID / document) and recall split into mentions
  SEEN vs UNSEEN as gold mentions in train. The unseen recall is the number that tells whether
  the model generalises beyond memorisation (the dictionary has 0 by construction).

Design for a low-end GPU (RTX 2050, 4 GB)
-----------------------------------------
* Base-size encoders (~110-125M parameters), full fine-tuning, mixed precision (bf16 when the
  GPU supports it, otherwise fp16 with loss scaling), micro-batch 8 + gradient accumulation 2
  (effective batch 16, the value of the thesis proposal), optional gradient checkpointing.
* Dynamic padding, short max length (128 sub-words; 256 for BC5CDR abstracts), and a sliding
  window with overlap for NER so long inputs are never truncated (truncating would silently
  drop gold mentions and invalidate the evaluation). Window predictions are averaged.
* A CUDA out-of-memory error aborts only that run (recorded as status ``oom`` with advice)
  instead of the whole session. Finished runs are cached and skipped on re-runs.
* Peak allocated VRAM and training time are recorded per run (efficiency analysis of the plan).

Text normalisation
------------------
CWLC is lower-cased (49.8% of its documents are all caps) and paired with an UNCASED Spanish
model; ADE/BC5CDR keep their original casing with a cased English model.

Models (chosen by --model-es / --model-en; any Hugging Face id or local folder works)
----------------------------------------------------------------------------------
    Spanish default : dccuchile/bert-base-spanish-wwm-uncased   (BETO uncased, ~110M)
    English default : dmis-lab/biobert-base-cased-v1.1          (BioBERT, ~110M)
    Spanish alternatives worth a controlled comparison:
        PlanTL-GOB-ES/bsc-bio-ehr-es  (biomedical + EHR RoBERTa, ~125M, CASED: it is fed
                                       lower-cased text here, which is a mismatch to be tested)
        dccuchile/distilbert-base-spanish-uncased (DistilBETO, ~67M, fallback if memory is tight)

Limitations
-----------
* Seeds: one by default (--seeds to add more). With a single seed the CI only reflects test
  sampling, not training variance.
* Hyper-parameters are the thesis-proposal values (lr 2e-5, wd 0.01, 3 epochs); they are NOT
  tuned per task. Epoch selection uses val only; the test split is scored once per run.
* GPU kernels are not bit-wise deterministic, so re-running can change the third decimal.

Usage
-----
    python 04_baseline_supervisado.py --dry-run                 # list planned runs
    python 04_baseline_supervisado.py --corpora cwlc --layers Medication --max-train-samples 200
    python 04_baseline_supervisado.py                           # everything, 1 seed
    python 04_baseline_supervisado.py --model-es PlanTL-GOB-ES/bsc-bio-ehr-es --corpora cwlc
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification,
                          AutoTokenizer, BertConfig)


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


# Single source of truth for BIO decoding, mention keys, bootstrap, metrics and table writers.
eda = load_sibling("03_exploracion.py")
log = eda.log

DEFAULT_ES_MODEL = "dccuchile/bert-base-spanish-wwm-uncased"
DEFAULT_EN_MODEL = "dmis-lab/biobert-base-cased-v1.1"
DEFAULT_LAYERS = {"ade": ["drug", "effect"], "bc5cdr": ["chemical", "disease"], "cwlc": ["Medication", "Disease", "Finding"]}
CORPUS_LANG = {"ade": "en", "bc5cdr": "en", "cwlc": "es"}
# (max sub-words per window, overlap in sub-words) by corpus
WINDOW = {"ade": (128, 32), "bc5cdr": (256, 64), "cwlc": (128, 32)}
TAG2ID = {"O": 0, "B": 1, "I": 2}
ID2TAG = {0: "O", 1: "B-X", 2: "I-X"}  # decoded with eda.extract_spans (lenient BIO)


# =============================================================================
# Configuration and small utilities
# =============================================================================
@dataclass
class RunConfig:
    """Hyper-parameters of one run (defaults = thesis proposal + low-VRAM settings)."""
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    batch_size: int = 8
    grad_accum: int = 2
    max_len: int = 128
    stride: int = 32
    seed: int = 42
    n_boot: int = 1000
    grad_checkpointing: bool = False
    amp: str = "auto"  # auto | bf16 | fp16 | off


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slug(text: str) -> str:
    """File-system safe short name of a model id or path."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(text).name if Path(text).exists() else text).strip("_")


def pick_amp(device: torch.device, mode: str) -> torch.dtype | None:
    """Autocast dtype: bf16 if the GPU supports it (no loss scaling needed), else fp16; None on CPU/off."""
    if device.type != "cuda" or mode == "off":
        return None
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def free_memory() -> None:
    """Release Python and CUDA caches between runs (matters on 4 GB)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(error: BaseException) -> bool:
    """True for CUDA out-of-memory errors across torch versions."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(error, oom_type) or "out of memory" in str(error).lower()


# =============================================================================
# Encoding: words -> sub-words with first-sub-word labels and overlapping windows
# =============================================================================
def tag_to_id(tag: str) -> int:
    """'B-DRUG' / 'I-DRUG' / 'O' -> 1 / 2 / 0 (single type per model, so the type is dropped)."""
    return TAG2ID[tag[0]]


def load_config(model_name: str, **overrides: Any):
    """Load a model config, tolerating repos whose config.json has no ``model_type`` key.

    The original BioBERT checkpoints (``dmis-lab/biobert-*``) ship such a config; they are plain
    BERT models, so we fall back to ``BertConfig`` in that single case. Overrides (num_labels,
    id2label, ...) are set on the object, because passing both ``config`` and those kwargs to
    ``from_pretrained`` is not allowed.
    """
    try:
        config = AutoConfig.from_pretrained(model_name)
    except ValueError as err:
        if "model_type" not in str(err):
            raise
        log(f"  note: {model_name} has no 'model_type' in config.json; treating it as BERT")
        config = BertConfig.from_pretrained(model_name)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def is_cased_name(model_name: str) -> bool:
    """True for checkpoints named '...cased...' (and not 'uncased')."""
    low = model_name.lower()
    return "cased" in low and "uncased" not in low


def load_tokenizer(model_name: str):
    """Fast tokenizer (needed for word_ids). RoBERTa-style BPE requires add_prefix_space for pre-split words.

    Cased BERT checkpoints without a tokenizer_config.json (e.g. BioBERT) would silently get the
    library default ``do_lower_case=True``; we force False and verify it.
    """
    config = load_config(model_name)
    kwargs: dict[str, Any] = {"config": config}
    if config.model_type == "roberta":
        kwargs["add_prefix_space"] = True
    if config.model_type == "bert" and is_cased_name(model_name):
        kwargs["do_lower_case"] = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(f"{model_name}: a fast tokenizer is required (word alignment); none available")
    if is_cased_name(model_name):
        first = tokenizer.tokenize("Aspirin")[0]
        if first[:1].islower():
            raise RuntimeError(f"{model_name} is cased but its tokenizer lower-cases the input (got {first!r}); "
                               "this would degrade results silently")
    return tokenizer


def encode_words(tokenizer, words: Sequence[Sequence[str]], tags: Sequence[Sequence[str]] | None,
                 max_len: int, stride: int) -> list[dict[str, Any]]:
    """Tokenise pre-split words into overlapping windows.

    Every window gets: input ids, mask, optional token types, labels (first sub-word of each word
    carries the tag, the rest -100), ``sample`` (index of the source unit), and ``first``, the list
    of (position, word index) pairs used to read predictions back at word level.
    A word whose first piece falls in the overlap is labelled/predicted in every window that
    contains it; predictions are averaged later.
    """
    if stride >= max_len - 2:
        raise ValueError("stride must be smaller than max_len - 2")
    enc = tokenizer([list(w) for w in words], is_split_into_words=True, truncation=True, max_length=max_len,
                    stride=stride, return_overflowing_tokens=True, padding=False)
    mapping = enc["overflow_to_sample_mapping"]
    features = []
    for i in range(len(enc["input_ids"])):
        sample, word_ids = mapping[i], enc.word_ids(i)
        labels, first, previous = [], [], None
        for pos, word in enumerate(word_ids):
            if word is None or word == previous:
                labels.append(-100)
            else:
                labels.append(tag_to_id(tags[sample][word]) if tags is not None else 0)
                first.append((pos, word))
            previous = word
        feature = {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
                   "labels": labels, "sample": sample, "first": first}
        if "token_type_ids" in enc:
            feature["token_type_ids"] = enc["token_type_ids"][i]
        features.append(feature)
    return features


def merge_word_probs(features: Sequence[dict[str, Any]], probs: Sequence[np.ndarray],
                     n_words: Sequence[int]) -> list[np.ndarray]:
    """Average window-level class probabilities into one (n_words, 3) array per unit.

    Words never seen by any window (e.g. a token the tokenizer drops entirely) default to 'O'.
    """
    acc = [np.zeros((n, len(TAG2ID))) for n in n_words]
    cnt = [np.zeros(n) for n in n_words]
    for feature, p in zip(features, probs):
        for pos, word in feature["first"]:
            acc[feature["sample"]][word] += p[pos]
            cnt[feature["sample"]][word] += 1
    merged = []
    for a, c in zip(acc, cnt):
        out = np.zeros_like(a)
        seen = c > 0
        out[seen] = a[seen] / c[seen, None]
        out[~seen, TAG2ID["O"]] = 1.0
        merged.append(out)
    return merged


# =============================================================================
# Batching, prediction, training loop
# =============================================================================
def collate(batch: Sequence[dict[str, Any]], pad_id: int, with_labels: bool = True) -> dict[str, torch.Tensor]:
    """Dynamic padding of the tensor fields; labels padded with -100 (ignored by the loss)."""
    longest = max(len(b["input_ids"]) for b in batch)
    out: dict[str, torch.Tensor] = {}
    fields = [("input_ids", pad_id), ("attention_mask", 0)]
    if "token_type_ids" in batch[0]:
        fields.append(("token_type_ids", 0))
    for name, fill in fields:
        out[name] = torch.tensor([list(b[name]) + [fill] * (longest - len(b[name])) for b in batch], dtype=torch.long)
    if with_labels:
        labels = [b["labels"] for b in batch]
        if isinstance(labels[0], list):  # token level
            out["labels"] = torch.tensor([l + [-100] * (longest - len(l)) for l in labels], dtype=torch.long)
        else:  # sequence level
            out["labels"] = torch.tensor(labels, dtype=torch.long)
    return out


@torch.no_grad()
def predict_probs(model, features: Sequence[dict[str, Any]], device: torch.device, amp_dtype: torch.dtype | None,
                  pad_id: int, batch_size: int, token_level: bool) -> list[np.ndarray]:
    """Softmax probabilities per feature (token level: (seq, 3) trimmed to length; else (2,)).

    Batches are sorted by length to minimise padding; results are returned in the original order.
    """
    model.eval()
    order = sorted(range(len(features)), key=lambda i: len(features[i]["input_ids"]))
    out: list[np.ndarray | None] = [None] * len(features)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = collate([features[i] for i in idx], pad_id, with_labels=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
        with ctx:
            logits = model(**batch).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = probs[j, :len(features[i]["input_ids"])] if token_level else probs[j]
    return out  # type: ignore[return-value]


def fit(model, train_features: Sequence[dict[str, Any]], evaluate: Callable[[Any], float], cfg: RunConfig,
        device: torch.device, amp_dtype: torch.dtype | None, pad_id: int, name: str) -> tuple[list[dict[str, float]], int]:
    """Fine-tune with AdamW + linear warm-up/decay; keep the weights of the best validation epoch.

    ``evaluate(model)`` returns the validation score to maximise (entity F1 / average precision).
    Gradient accumulation emulates the effective batch of the proposal with a small micro-batch.
    """
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(list(train_features), batch_size=cfg.batch_size, shuffle=True, generator=generator,
                        collate_fn=lambda b: collate(b, pad_id))
    updates_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_updates = max(1, updates_per_epoch * cfg.epochs)
    warmup = int(cfg.warmup_ratio * total_updates)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]  # frozen weights (PEFT) are skipped
    groups = [
        {"params": [p for n, p in trainable if not any(k in n for k in no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in trainable if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    all_trainable = len(trainable) == sum(1 for _ in model.parameters())
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / max(1, warmup) if step < warmup else max(0.0, (total_updates - step) / max(1, total_updates - warmup)))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    best_score, best_state, best_epoch, history = -math.inf, None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        started, running, seen_batches = time.time(), 0.0, 0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
            with ctx:
                loss = model(**batch).loss / cfg.grad_accum
            scaler.scale(loss).backward()
            running, seen_batches = running + loss.item() * cfg.grad_accum, seen_batches + 1
            if (step + 1) % cfg.grad_accum == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        score = evaluate(model)
        history.append({"epoch": epoch, "train_loss": running / max(1, seen_batches), "val_score": score,
                        "epoch_seconds": time.time() - started})
        log(f"    [{name}] epoch {epoch}/{cfg.epochs}  loss={history[-1]['train_loss']:.4f}  val={score:.4f}  "
            f"({history[-1]['epoch_seconds']:.0f}s)")
        if score > best_score:
            best_score, best_epoch = score, epoch
            if all_trainable:  # full fine-tuning: keep the whole state dict, exactly as before
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:  # PEFT: only the adapters and the head change, so only they are copied (~1 MB instead of ~440 MB)
                best_state = {n: p.detach().cpu().clone() for n, p in trainable}
    outcome = model.load_state_dict(best_state, strict=all_trainable)
    if not all_trainable and outcome.unexpected_keys:  # strict=False would hide a naming mismatch: fail loudly instead
        raise RuntimeError(f"Could not restore the best epoch, unknown keys: {outcome.unexpected_keys[:3]}")
    return history, best_epoch


# =============================================================================
# NER evaluation
# =============================================================================
@dataclass
class NerData:
    """One split of a single-type NER task (words may already be lower-cased)."""
    unit_ids: list[str]
    words: list[list[str]]
    tags: list[list[str]]
    clusters: list[str]


def ner_counts(words: Sequence[Sequence[str]], gold_tags: Sequence[Sequence[str]], pred_tags: Sequence[Sequence[str]],
               train_keys: set[tuple[str, ...]], mask: np.ndarray | None = None) -> dict[str, Any]:
    """Entity-level exact-match counts per unit plus seen/unseen recall bookkeeping.

    Same definition as the dictionary baseline: a mention is *seen* when its lower-cased string is
    a gold mention somewhere in train. ``mask`` restricts the evaluation to a subset of units.
    """
    n = len(words)
    tp, fp, fn = (np.zeros(n, dtype=np.int64) for _ in range(3))
    seen_total = seen_hit = unseen_total = unseen_hit = 0
    for u in range(n):
        if mask is not None and not mask[u]:
            continue
        gold, pred = set(eda.extract_spans(gold_tags[u])), set(eda.extract_spans(pred_tags[u]))
        tp[u], fp[u], fn[u] = len(gold & pred), len(pred - gold), len(gold - pred)
        for span in gold:
            hit = span in pred
            if eda.mention_key(words[u], span) in train_keys:
                seen_total, seen_hit = seen_total + 1, seen_hit + hit
            else:
                unseen_total, unseen_hit = unseen_total + 1, unseen_hit + hit
    return {"tp": tp, "fp": fp, "fn": fn, "seen_total": seen_total, "seen_hit": seen_hit,
            "unseen_total": unseen_total, "unseen_hit": unseen_hit}


def decode_tags(merged: Sequence[np.ndarray]) -> list[list[str]]:
    """Argmax per word -> tag strings (O / B-X / I-X)."""
    return [[ID2TAG[int(k)] for k in probs.argmax(axis=1)] for probs in merged]


def train_keys_of(data: NerData) -> set[tuple[str, ...]]:
    """Lower-cased gold mention strings of a split (used for the seen/unseen split of recall)."""
    return {eda.mention_key(w, span) for w, t in zip(data.words, data.tags) for span in eda.extract_spans(t)}


def run_ner(train: NerData, val: NerData, test: NerData, tokenizer, model_name: str, cfg: RunConfig,
            variants: dict[str, np.ndarray | None], device: torch.device, amp_dtype: torch.dtype | None,
            window: tuple[int, int], run_name: str, save_dir: Path | None = None,
            wrap_model: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    """Fine-tune a token classifier on one entity type and evaluate it on test.

    Returns {rows, history, predictions}; one row per test variant (first variant = primary).
    ``wrap_model`` (optional) receives the freshly loaded model and returns the model to train, e.g. a
    PEFT/LoRA wrapper (script 06); by default every weight is trained (full fine-tuning).
    """
    set_seed(cfg.seed)
    max_len, stride = window
    feats = {name: encode_words(tokenizer, d.words, d.tags, max_len, stride)
             for name, d in (("train", train), ("val", val), ("test", test))}
    id2label = {i: t for i, t in ID2TAG.items()}
    config = load_config(model_name, num_labels=len(TAG2ID), id2label=id2label,
                         label2id={v: k for k, v in id2label.items()})
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    if wrap_model is not None:
        model = wrap_model(model)
    model = model.to(device)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eval_bs = max(8, cfg.batch_size * 2)
    keys = train_keys_of(train)

    def predict_split(m, split: str, data: NerData) -> list[list[str]]:
        probs = predict_probs(m, feats[split], device, amp_dtype, pad_id, eval_bs, token_level=True)
        return decode_tags(merge_word_probs(feats[split], probs, [len(w) for w in data.words]))

    def val_f1(m) -> float:
        c = ner_counts(val.words, val.tags, predict_split(m, "val", val), keys)
        return eda.prf(int(c["tp"].sum()), int(c["fp"].sum()), int(c["fn"].sum()))[2]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    history, best_epoch = fit(model, feats["train"], val_f1, cfg, device, amp_dtype, pad_id, run_name)
    train_seconds = time.time() - started
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else float("nan")

    pred_val = predict_split(model, "val", val)
    pred_test = predict_split(model, "test", test)
    val_counts = ner_counts(val.words, val.tags, pred_val, keys)
    base = {
        "model": model_name, "seed": cfg.seed, "best_epoch": best_epoch, "val_f1": history[best_epoch - 1]["val_score"],
        "train_time_s": train_seconds, "peak_vram_gb": peak_gb,
        "n_params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "n_trainable_m": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        "max_len": max_len, "stride": stride,
        "batch_size": cfg.batch_size, "grad_accum": cfg.grad_accum, "lr": cfg.lr, "epochs": cfg.epochs,
        "n_train_units": len(train.words), "n_train_windows": len(feats["train"]), "status": "ok",
    }
    rows = []
    for k, (variant, mask) in enumerate(variants.items()):
        c = ner_counts(test.words, test.tags, pred_test, keys, mask)
        p, r, f1 = eda.prf(int(c["tp"].sum()), int(c["fp"].sum()), int(c["fn"].sum()))
        row = {**base, "split": "test", "variant": variant, "primary": k == 0, "precision": p, "recall": r, "f1": f1,
               "n_gold": c["seen_total"] + c["unseen_total"],
               "recall_seen": c["seen_hit"] / c["seen_total"] if c["seen_total"] else float("nan"),
               "recall_unseen": c["unseen_hit"] / c["unseen_total"] if c["unseen_total"] else float("nan"),
               "unseen_share": c["unseen_total"] / max(1, c["seen_total"] + c["unseen_total"]),
               "ci95_f1_low": float("nan"), "ci95_f1_high": float("nan")}
        if cfg.n_boot > 0:
            sel = np.ones(len(test.words), bool) if mask is None else mask
            ci = eda.bootstrap_micro_prf(np.asarray(test.clusters)[sel], c["tp"][sel], c["fp"][sel], c["fn"][sel], cfg.n_boot, cfg.seed)
            row["ci95_f1_low"], row["ci95_f1_high"] = ci["f1"]
        rows.append(row)
    rows.append({**base, "split": "val", "variant": "all", "primary": False,
                 **dict(zip(("precision", "recall", "f1"), eda.prf(int(val_counts["tp"].sum()), int(val_counts["fp"].sum()), int(val_counts["fn"].sum())))),
                 "n_gold": val_counts["seen_total"] + val_counts["unseen_total"]})

    predictions = [{"unit_id": uid, "gold": eda.extract_spans(g), "pred": eda.extract_spans(p)}
                   for uid, g, p in zip(test.unit_ids, test.tags, pred_test)]
    if save_dir is not None:
        model.save_pretrained(save_dir / run_name)
        tokenizer.save_pretrained(save_dir / run_name)
    del model
    free_memory()
    return {"rows": rows, "history": history, "predictions": predictions}


# =============================================================================
# ADE sentence classification
# =============================================================================
def run_classifier(train: dict[str, Any], val: dict[str, Any], test: dict[str, Any], tokenizer, model_name: str,
                   cfg: RunConfig, device: torch.device, amp_dtype: torch.dtype | None, run_name: str,
                   save_dir: Path | None = None, wrap_model: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    """Fine-tune a sequence classifier (has-ADE vs not). Each split dict: texts, labels, clusters, unit_ids.

    The epoch is chosen by validation average precision (threshold-free); the decision threshold is then
    the one maximising val F1, exactly as in the TF-IDF baseline. Test is scored once.
    """
    from sklearn.metrics import average_precision_score

    set_seed(cfg.seed)

    def encode(d: dict[str, Any]) -> list[dict[str, Any]]:
        enc = tokenizer(list(d["texts"]), truncation=True, max_length=cfg.max_len, padding=False)
        feats = []
        for i in range(len(d["texts"])):
            f = {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": int(d["labels"][i])}
            if "token_type_ids" in enc:
                f["token_type_ids"] = enc["token_type_ids"][i]
            feats.append(f)
        return feats

    feats = {"train": encode(train), "val": encode(val), "test": encode(test)}
    truncated = float(np.mean([len(f["input_ids"]) >= cfg.max_len for f in feats["test"]]))
    config = load_config(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    if wrap_model is not None:
        model = wrap_model(model)
    model = model.to(device)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eval_bs = max(8, cfg.batch_size * 2)
    y_val, y_test = np.asarray(val["labels"]), np.asarray(test["labels"])

    def scores(m, split: str) -> np.ndarray:
        return np.array([p[1] for p in predict_probs(m, feats[split], device, amp_dtype, pad_id, eval_bs, token_level=False)])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    history, best_epoch = fit(model, feats["train"], lambda m: float(average_precision_score(y_val, scores(m, "val"))),
                              cfg, device, amp_dtype, pad_id, run_name)
    train_seconds = time.time() - started
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else float("nan")

    s_val, s_test = scores(model, "val"), scores(model, "test")
    threshold = eda.best_f1_threshold(y_val, s_val)
    point = eda.classification_metrics(y_test, s_test, threshold)
    draws: dict[str, list[float]] = {k: [] for k in point}
    if cfg.n_boot > 0:
        for idx in eda.cluster_bootstrap_indices(test["clusters"], cfg.n_boot, cfg.seed):
            if y_test[idx].min() == y_test[idx].max():
                continue
            for k, v in eda.classification_metrics(y_test[idx], s_test[idx], threshold).items():
                draws[k].append(v)
    lo, hi = eda.percentile_ci(np.array(draws["f1"])) if draws["f1"] else (float("nan"), float("nan"))
    row = {
        "model": model_name, "seed": cfg.seed, "split": "test", "variant": "all", "primary": True,
        "precision": point["precision"], "recall": point["recall"], "f1": point["f1"],
        "auroc": point["auroc"], "avg_precision": point["avg_precision"], "ci95_f1_low": lo, "ci95_f1_high": hi,
        "threshold_from_val": threshold, "n_gold": int(y_test.sum()), "best_epoch": best_epoch,
        "val_f1": float(eda.classification_metrics(y_val, s_val, threshold)["f1"]), "train_time_s": train_seconds,
        "peak_vram_gb": peak_gb, "n_params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "n_trainable_m": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        "max_len": cfg.max_len, "pct_test_truncated": 100 * truncated, "batch_size": cfg.batch_size,
        "grad_accum": cfg.grad_accum, "lr": cfg.lr, "epochs": cfg.epochs, "n_train_units": len(train["texts"]), "status": "ok",
    }
    predictions = [{"unit_id": u, "label": int(l), "score": float(s)} for u, l, s in zip(test["unit_ids"], y_test, s_test)]
    if save_dir is not None:
        model.save_pretrained(save_dir / run_name)
        tokenizer.save_pretrained(save_dir / run_name)
    del model
    free_memory()
    return {"rows": [row], "history": history, "predictions": predictions}


# =============================================================================
# Data preparation from the frozen processed files
# =============================================================================
def build_ner_splits(corpus: str, layer: str, data: dict[str, list[dict[str, Any]]], lowercase: bool,
                     max_train: int | None, max_eval: int | None) -> tuple[NerData, NerData, NerData, dict[str, np.ndarray | None]]:
    """Single-layer NER splits plus the test variants to report.

    * ADE keeps only positive sentences (negatives have no annotation, see module docstring).
    * CWLC adds the 'novel_sentences' variant (test sentences never seen verbatim in train).
    """
    splits: dict[str, NerData] = {}
    novel_mask: np.ndarray | None = None
    for name in eda.SPLITS:
        records = data[name]
        if corpus == "ade":
            records = [r for r in records if r["label"] == 1]
        tags = eda.layer_tags(corpus, records)[layer]
        ids = [str(r.get("sentence_id") or r["pmid"]) for r in records]
        clusters = eda.cluster_ids(corpus, records)
        words = [[t.lower() for t in r["tokens"]] if lowercase else list(r["tokens"]) for r in records]
        if corpus == "cwlc" and name == "test":
            novel_mask = np.array([not r["seen_in_train"] for r in records])
        limit = max_train if name == "train" else max_eval
        if limit:
            words, tags, ids, clusters = words[:limit], tags[:limit], ids[:limit], clusters[:limit]
            if novel_mask is not None and name == "test":
                novel_mask = novel_mask[:limit]
        splits[name] = NerData(ids, words, [list(t) for t in tags], clusters)
    variants: dict[str, np.ndarray | None] = {"positive_sentences" if corpus == "ade" else "all": None}
    if corpus == "cwlc" and novel_mask is not None:
        variants["novel_sentences"] = novel_mask
    return splits["train"], splits["val"], splits["test"], variants


def build_clf_splits(data: dict[str, list[dict[str, Any]]], max_train: int | None, max_eval: int | None) -> dict[str, dict[str, Any]]:
    """ADE sentence-classification splits (all sentences, original casing)."""
    out = {}
    for name in eda.SPLITS:
        records = data[name]
        limit = max_train if name == "train" else max_eval
        if limit:  # keep both classes in small smoke subsets: interleave positives and negatives
            pos = [r for r in records if r["label"] == 1][: limit // 2]
            neg = [r for r in records if r["label"] == 0][: limit - len(pos)]
            records = pos + neg
        out[name] = {"texts": [r["text"] for r in records], "labels": [r["label"] for r in records],
                     "clusters": eda.cluster_ids("ade", records), "unit_ids": [r["sentence_id"] for r in records]}
    return out


# =============================================================================
# Result storage and aggregation
# =============================================================================
def run_paths(interim_dir: Path, run_id: str, runs_subdir: str = "sup_runs") -> tuple[Path, Path]:
    """(result json, predictions jsonl) of one run. PEFT runs (script 06) use their own ``runs_subdir``."""
    return interim_dir / runs_subdir / f"{run_id}.json", interim_dir / "predictions" / f"{run_id}_test.jsonl"


def save_run(interim_dir: Path, run_id: str, meta: dict[str, Any], result: dict[str, Any],
             runs_subdir: str = "sup_runs") -> None:
    """Persist rows + history (json) and test predictions (jsonl, for the error analysis of stage 4)."""
    result_path, pred_path = run_paths(interim_dir, run_id, runs_subdir)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**meta, **row} for row in result["rows"]]
    result_path.write_text(json.dumps({"run_id": run_id, "rows": rows, "history": result["history"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    with pred_path.open("w", encoding="utf-8") as handle:
        for record in result["predictions"]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def aggregate(interim_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect every finished run: (all rows, summary of the primary test rows by model across seeds)."""
    rows = []
    for path in sorted((interim_dir / "sup_runs").glob("*.json")):
        if path.stem.endswith("-smoke"):
            continue
        rows += json.loads(path.read_text(encoding="utf-8"))["rows"]
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs, runs
    ok = runs[(runs.status == "ok") & (runs.split == "test") & (runs.primary == True)]  # noqa: E712
    keys = ["corpus", "task", "layer", "model", "variant"]
    ok = ok.assign(layer=ok["layer"].fillna("-"))
    for column in ("auroc", "recall_seen", "recall_unseen", "unseen_share"):  # absent when only one task type ran
        if column not in ok:
            ok[column] = np.nan
    summary = (ok.groupby(keys, dropna=False)
               .agg(n_seeds=("seed", "nunique"), n_gold=("n_gold", "first"),
                    precision=("precision", "mean"), recall=("recall", "mean"),
                    f1=("f1", "mean"), f1_sd=("f1", "std"),
                    ci95_f1_low=("ci95_f1_low", "first"), ci95_f1_high=("ci95_f1_high", "first"),
                    recall_seen=("recall_seen", "mean"), recall_unseen=("recall_unseen", "mean"),
                    unseen_share=("unseen_share", "mean"), auroc=("auroc", "mean"),
                    train_min=("train_time_s", lambda s: s.mean() / 60), peak_vram_gb=("peak_vram_gb", "max"),
                    n_params_m=("n_params_m", "first"))
               .reset_index())
    return runs, summary


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
    p.add_argument("--save-model", type=Path, default=None, help="folder where the best weights of every run are saved (off by default)")
    p.add_argument("--corpora", nargs="*", default=list(DEFAULT_LAYERS), choices=list(DEFAULT_LAYERS))
    p.add_argument("--tasks", nargs="*", default=["clf", "ner"], choices=["clf", "ner"])
    p.add_argument("--layers", nargs="*", default=None, help="entity types to train (default: the thesis design)")
    p.add_argument("--model-es", default=DEFAULT_ES_MODEL)
    p.add_argument("--model-en", default=DEFAULT_EN_MODEL)
    p.add_argument("--lowercase", choices=["auto", "always", "never"], default="auto", help="auto = lower-case CWLC only")
    p.add_argument("--seeds", type=int, nargs="*", default=[42])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=8, help="micro-batch (lower it on out-of-memory)")
    p.add_argument("--grad-accum", type=int, default=2, help="micro-batches per update (effective batch = batch-size x grad-accum)")
    p.add_argument("--max-len", type=int, default=None, help="override the sub-word window (default 128; 256 for BC5CDR)")
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto")
    p.add_argument("--grad-checkpointing", action="store_true", help="trade ~30%% speed for much less activation memory")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--max-train-samples", type=int, default=None, help="smoke test: first N training units")
    p.add_argument("--max-eval-samples", type=int, default=None, help="smoke test: first N val/test units")
    p.add_argument("--overwrite", action="store_true", help="re-run even if a result already exists")
    p.add_argument("--dry-run", action="store_true", help="print the planned runs and exit")
    p.add_argument("--no-verify", action="store_true")
    return p


def plan_runs(args, available_layers: dict[str, list[str]]) -> list[dict[str, str]]:
    """Cartesian plan of (corpus, task, layer) honouring --corpora, --tasks and --layers."""
    plan = []
    for corpus in args.corpora:
        if "clf" in args.tasks and corpus == "ade":
            plan.append({"corpus": corpus, "task": "clf", "layer": ""})
        if "ner" in args.tasks:
            if args.layers:  # user selection, matched case-insensitively against the layers that exist
                lookup = {l.lower(): l for l in available_layers[corpus]}
                chosen = [lookup[w.lower()] for w in args.layers if w.lower() in lookup]
            else:  # default: exactly the layers of the thesis design
                chosen = DEFAULT_LAYERS[corpus]
            plan += [{"corpus": corpus, "task": "ner", "layer": l} for l in chosen]
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    """Run the planned experiments, cache each one, then rebuild the summary tables."""
    args = build_parser().parse_args(argv)
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
    plan = plan_runs(args, available)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = pick_amp(device, args.amp)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}, {props.total_memory / 1024 ** 3:.1f} GB, autocast={amp_dtype}")
    else:
        log("WARNING: no CUDA GPU detected; training on CPU is very slow. Use --max-train-samples for a smoke test.")
    log("Planned runs: " + ", ".join(f"{r['corpus']}/{r['layer'] or 'clf'}" for r in plan) + f"  x seeds {args.seeds}")
    if args.dry_run:
        return 0

    tokenizers: dict[str, Any] = {}
    for item in plan:
        corpus, task, layer = item["corpus"], item["task"], item["layer"]
        model_name = args.model_es if CORPUS_LANG[corpus] == "es" else args.model_en
        lowercase = args.lowercase == "always" or (args.lowercase == "auto" and corpus == "cwlc")
        window = (args.max_len or WINDOW[corpus][0], WINDOW[corpus][1])
        if args.max_len and window[1] >= window[0] - 2:
            window = (window[0], max(0, window[0] // 4))
        if model_name not in tokenizers:
            tokenizers[model_name] = load_tokenizer(model_name)
            sample = "Paciente con dolor abdominal y cefalea" if CORPUS_LANG[corpus] == "es" else "Patient developed rash after aspirin"
            log(f"  tokenizer check [{model_name}]: {tokenizers[model_name].tokenize(sample.lower() if lowercase else sample)}")
        tokenizer = tokenizers[model_name]

        for seed in args.seeds:
            cfg = RunConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, grad_accum=args.grad_accum,
                            max_len=window[0], stride=window[1], seed=seed, n_boot=args.n_boot,
                            grad_checkpointing=args.grad_checkpointing, amp=args.amp)
            run_id = f"{corpus}-{layer or 'clf'}-{slug(model_name)}-s{seed}"
            if args.max_train_samples or args.max_eval_samples:
                run_id += "-smoke"  # smoke runs never collide with (or get summarised as) real runs
            result_path, _ = run_paths(args.interim_dir, run_id)
            if result_path.exists() and not args.overwrite:
                log(f"[skip] {run_id} (already done)")
                continue
            log(f"[run] {run_id}  (lowercase={lowercase}, window={window})")
            meta = {"corpus": corpus, "task": task, "layer": layer or None, "lowercased": lowercase, "run_id": run_id}
            try:
                if task == "clf":
                    splits = build_clf_splits(data[corpus], args.max_train_samples, args.max_eval_samples)
                    result = run_classifier(splits["train"], splits["val"], splits["test"], tokenizer, model_name, cfg,
                                            device, amp_dtype, run_id, args.save_model)
                else:
                    train, val, test, variants = build_ner_splits(corpus, layer, data[corpus], lowercase,
                                                                  args.max_train_samples, args.max_eval_samples)
                    result = run_ner(train, val, test, tokenizer, model_name, cfg, variants, device, amp_dtype, window,
                                     run_id, args.save_model)
            except Exception as error:  # noqa: BLE001 - one failed run must not kill an unattended session
                if not is_oom(error):
                    raise
                free_memory()
                log(f"  OUT OF MEMORY in {run_id}. Try --batch-size {max(1, args.batch_size // 2)} --grad-accum "
                    f"{args.grad_accum * 2}, add --grad-checkpointing, or use a smaller model (DistilBETO).")
                result = {"rows": [{"status": "oom", "model": model_name, "seed": seed, "split": "test", "primary": True}],
                          "history": [], "predictions": []}
            save_run(args.interim_dir, run_id, meta, result)
            top = result["rows"][0]
            if top.get("status") == "ok":
                log(f"  -> test F1={top['f1']:.4f} [{top['ci95_f1_low']:.4f}, {top['ci95_f1_high']:.4f}]  "
                    f"peak VRAM={top['peak_vram_gb']:.2f} GB  time={top['train_time_s'] / 60:.1f} min")

    runs, summary = aggregate(args.interim_dir)
    if not runs.empty:
        args.tables_dir.mkdir(parents=True, exist_ok=True)
        runs.to_csv(args.tables_dir / "sup_runs.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(args.tables_dir / "sup_summary.csv", index=False, encoding="utf-8-sig")
        # Compact version that fits a thesis page (the CSV above keeps every column)
        compact = summary[["corpus", "layer", "model", "n_gold", "precision", "recall", "f1", "ci95_f1_low", "ci95_f1_high",
                           "recall_unseen"]].copy()
        compact["model"] = compact["model"].str.split("/").str[-1]
        eda.save_table(compact, "sup_summary_thesis", args.tables_dir,
                       "Supervised baselines (full fine-tuning), entity-level exact-match scores on test (CI95: cluster bootstrap).", 3)
        log(f"\nTables: sup_summary.csv, sup_summary_thesis.csv/.tex and sup_runs.csv in {args.tables_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
