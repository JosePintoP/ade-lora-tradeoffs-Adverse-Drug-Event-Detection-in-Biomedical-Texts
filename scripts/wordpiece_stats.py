"""Sentence length statistics of the ADE splits (paper, preprocessing section).

Prints, for the training set, the length in characters, whitespace-separated words and
WordPiece tokens (without special tokens), and counts the sentences of all splits that
would be truncated at the maximum sequence length.

Usage:
    python scripts/wordpiece_stats.py data/processed/ade bert-base-cased --max-length 128
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

SPLITS = ("train", "val", "test")


def load_texts(path: Path) -> list[str]:
    """Read the sentence texts of one JSONL split."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["text"] for line in handle]


def token_lengths(texts: list[str], tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
    """Return the WordPiece length of each sentence, without special tokens."""
    ids = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    return np.array([len(x) for x in ids])


def summary(name: str, lengths: np.ndarray) -> str:
    """Format mean, median, 95th percentile and maximum of a length distribution."""
    return (f"{name}: mean={lengths.mean():.1f} median={np.median(lengths):.0f} "
            f"p95={np.percentile(lengths, 95):.0f} max={lengths.max()}")


def main() -> None:
    """Print training-set length statistics and truncation counts over all splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("model")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    texts = {s: load_texts(args.data_dir / f"{s}.jsonl") for s in SPLITS}
    train = texts["train"]
    print(summary("train characters", np.array([len(t) for t in train])))
    print(summary("train words", np.array([len(t.split()) for t in train])))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lengths = {s: token_lengths(texts[s], tokenizer) for s in SPLITS}
    print(summary("train wordpieces", lengths["train"]))
    everything = np.concatenate([lengths[s] for s in SPLITS])
    limit = args.max_length - tokenizer.num_special_tokens_to_add()
    truncated = int((everything > limit).sum())
    print(f"truncated at {args.max_length}: {truncated} of {len(everything)} "
          f"({100 * truncated / len(everything):.2f}%)")


if __name__ == "__main__":
    main()
