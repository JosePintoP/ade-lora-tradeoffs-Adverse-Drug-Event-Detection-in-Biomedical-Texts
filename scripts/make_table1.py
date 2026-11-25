"""Build Table I (ADE-Corpus statistics by split) as CSV and LaTeX.

Usage:
    python make_table1.py --data-dir data/processed/ade --out-dir outputs/tables
"""
import argparse
import csv
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

SPLITS = ("train", "val", "test")
HEADER = ["", "Train", "Validation", "Test"]
CAPTION = (
    "Descriptive statistics of the ADE corpus by split. Splits are grouped by case report (PMID); "
    "no PMID or sentence is shared between sets. An instance is a unique sentence per case report. "
    "NER uses ADE-related sentences and counts word-level tokens. Drug and effect entities are "
    "tagged in separate BIO layers: the {both} tokens in both are counted in each, and O denotes "
    "tokens outside both."
)


def load_split(path: Path) -> list[dict]:
    """Read one JSONL split."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def assert_no_leakage(data: dict[str, list[dict]]) -> None:
    """Raise if any PMID or normalized sentence appears in more than one split."""
    keys = {
        "pmid": lambda r: r["pmid"],
        "sentence": lambda r: re.sub(r"[^a-z0-9]+", "", r["text"].lower()),
    }
    for (a, rows_a), (b, rows_b) in combinations(data.items(), 2):
        for level, key in keys.items():
            shared = {key(r) for r in rows_a} & {key(r) for r in rows_b}
            if shared:
                raise ValueError(f"Leakage between {a} and {b} by {level}: {len(shared)}")


def split_stats(rows: list[dict]) -> Counter:
    """Count sentences, case reports, relations and word-level BIO tags."""
    positives = [r for r in rows if r["label"] == 1]
    stats = Counter(
        sentences=len(rows),
        reports=len({r["pmid"] for r in rows}),
        ade=len(positives),
        non_ade=len(rows) - len(positives),
        ner_sentences=len(positives),
        relations=sum(r["n_relations"] for r in positives),
    )
    for row in positives:
        for drug, effect in zip(row["tags_drug"], row["tags_effect"], strict=True):
            stats["tokens"] += 1
            if drug != "O":
                stats[drug] += 1
            if effect != "O":
                stats[effect] += 1
            stats["O"] += drug == "O" and effect == "O"
            stats["both"] += drug != "O" and effect != "O"
    return stats


def build_rows(stats: list[Counter]) -> list[list[str]]:
    """Lay out the table rows in the order used in the paper."""

    def counts(label: str, key: str) -> list[str]:
        return [label, *(f"{s[key]:,}" for s in stats)]

    def shares(label: str, key: str, total: str) -> list[str]:
        return [label, *(f"{s[key]:,} ({100 * s[key] / s[total]:.1f}%)" for s in stats)]

    return [
        ["Sentence classification", "", "", ""],
        counts("Case reports", "reports"),
        counts("Sentences", "sentences"),
        shares("ADE", "ade", "sentences"),
        shares("non-ADE", "non_ade", "sentences"),
        ["NER (ADE sentences)", "", "", ""],
        counts("Sentences", "ner_sentences"),
        counts("Relations", "relations"),
        *(shares(tag, tag, "tokens") for tag in ("O", "B-DRUG", "I-DRUG", "B-EFFECT", "I-EFFECT")),
        counts("Total tokens", "tokens"),
    ]


def to_latex(rows: list[list[str]], caption: str) -> str:
    """Render the table as a booktabs LaTeX table."""

    def tex(value: str) -> str:
        return value.replace("%", r"\%")

    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
             r"\label{tab:ade_stats}", r"\begin{tabular}{lrrr}", r"\toprule",
             " & ".join(HEADER) + r" \\", r"\midrule"]
    for row in rows:
        if row[1] == "":
            if lines[-1] != r"\midrule":
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{row[0]}}}}} \\")
        else:
            lines.append(" & ".join(tex(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Load the splits, check leakage and write the table files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/ade"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tables"))
    args = parser.parse_args()

    data = {name: load_split(args.data_dir / f"{name}.jsonl") for name in SPLITS}
    assert_no_leakage(data)
    stats = [split_stats(data[name]) for name in SPLITS]
    rows = build_rows(stats)
    caption = CAPTION.format(both=sum(s["both"] for s in stats))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "table1_ade.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([HEADER, *rows])
    (args.out_dir / "table1_ade.tex").write_text(to_latex(rows, caption), encoding="utf-8")
    (args.out_dir / "table1_ade_caption.txt").write_text(caption, encoding="utf-8")
    for row in [HEADER, *rows]:
        print(" | ".join(row))


if __name__ == "__main__":
    main()
