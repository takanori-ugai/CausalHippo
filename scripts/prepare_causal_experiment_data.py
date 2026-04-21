#!/usr/bin/env python3
"""Prepare a balanced causal QA dataset from eval/causal CSV files.

Input CSV rows are expected as:
  e1, e2, paragraph(with optional <e1>/<e2> tags), answer(true/false), category

Outputs:
- causal_qa_all.jsonl: all converted rows
- causal_qa_balanced.jsonl: label-balanced subset (true/false)
- causal_qa_manifest.jsonl: metadata manifest per row
- causal_qa_summary.json: summary statistics
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ENTITY_TAG_RE = re.compile(r"</?e[12]>", re.IGNORECASE)
UNKNOWN_ANSWER = "unknown"
UNKNOWN_ALIASES = ["unknown", "none", "no direct cause", "not enough information"]


@dataclass(frozen=True)
class CausalRow:
    id: str
    source_file: str
    row_index: int
    category: str
    e1: str
    e2: str
    paragraph: str
    label_true: bool


def strip_entity_tags(text: str) -> str:
    return ENTITY_TAG_RE.sub("", text)


def parse_bool(raw: str) -> bool:
    return raw.strip().lower() == "true"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_causal_rows(paths: Sequence[Path]) -> List[CausalRow]:
    rows: List[CausalRow] = []
    for path in sorted(paths):
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            row_index = 0
            for fields in reader:
                if not fields or all(not c.strip() for c in fields):
                    continue
                if len(fields) < 4:
                    raise ValueError(
                        f"{path}: row {row_index} expected >=4 fields (e1,e2,paragraph,answer[,category]), got {len(fields)}"
                    )

                e1 = fields[0].strip()
                e2 = fields[1].strip()
                paragraph = strip_entity_tags(fields[2].strip())
                label_true = parse_bool(fields[3])
                category = fields[4].strip() if len(fields) > 4 and fields[4].strip() else "unknown"

                sample_id = f"{path.stem}_{row_index}"
                rows.append(
                    CausalRow(
                        id=sample_id,
                        source_file=path.name,
                        row_index=row_index,
                        category=category,
                        e1=e1,
                        e2=e2,
                        paragraph=paragraph,
                        label_true=label_true,
                    )
                )
                row_index += 1
    return rows


def build_noise_pool(rows: Sequence[CausalRow]) -> List[tuple[str, str]]:
    pool: List[tuple[str, str]] = []
    for row in rows:
        paragraph = normalize_text(row.paragraph)
        if paragraph:
            pool.append((row.id, paragraph))
    return pool


def sample_noise_texts(
    *,
    row_id: str,
    noise_pool: Sequence[tuple[str, str]],
    existing_texts: set[str],
    count: int,
    rng: random.Random,
) -> List[str]:
    if count <= 0 or not noise_pool:
        return []

    selected: List[str] = []
    seen = set(existing_texts)
    attempts = 0
    max_attempts = max(50, count * 30)

    while len(selected) < count and attempts < max_attempts:
        source_id, text = noise_pool[rng.randrange(len(noise_pool))]
        attempts += 1
        if source_id == row_id or text in seen:
            continue
        selected.append(text)
        seen.add(text)

    if len(selected) < count:
        start = rng.randrange(len(noise_pool))
        for offset in range(len(noise_pool)):
            source_id, text = noise_pool[(start + offset) % len(noise_pool)]
            if source_id == row_id or text in seen:
                continue
            selected.append(text)
            seen.add(text)
            if len(selected) >= count:
                break

    return selected


def to_jsonl_record(
    row: CausalRow,
    noise_pool: Sequence[tuple[str, str]],
    noise_texts_per_sample: int,
    rng: random.Random,
) -> dict:
    if row.label_true:
        answer = row.e1
        aliases: List[str] = []
    else:
        answer = UNKNOWN_ANSWER
        aliases = UNKNOWN_ALIASES

    source_paragraph = normalize_text(row.paragraph)
    paragraphs = [
        {
            "idx": 0,
            "title": row.e1,
            "paragraph_text": source_paragraph,
            "is_supporting": row.label_true,
        }
    ]

    if noise_texts_per_sample > 0 and noise_pool:
        noise_texts = sample_noise_texts(
            row_id=row.id,
            noise_pool=noise_pool,
            existing_texts={source_paragraph},
            count=noise_texts_per_sample,
            rng=rng,
        )
        for noise_text in noise_texts:
            paragraphs.append(
                {
                    "idx": len(paragraphs),
                    "title": "noise",
                    "paragraph_text": noise_text,
                    "is_supporting": False,
                }
            )

    return {
        "id": row.id,
        "paragraphs": paragraphs,
        "question": f"What is the cause of {row.e2}?",
        "answer": answer,
        "answer_aliases": aliases,
        "answerable": True,
        "metadata": {
            "label_true": row.label_true,
            "category": row.category,
            "source_file": row.source_file,
            "effect": row.e2,
            "cause_candidate": row.e1,
        },
    }


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(path: Path, rows: Iterable[CausalRow]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            manifest = {
                "id": row.id,
                "source_file": row.source_file,
                "row_index": row.row_index,
                "category": row.category,
                "label_true": row.label_true,
                "e1": row.e1,
                "e2": row.e2,
            }
            f.write(json.dumps(manifest, ensure_ascii=False) + "\n")


def build_balanced(rows: Sequence[CausalRow], seed: int, target_size: int | None) -> List[CausalRow]:
    by_label: Dict[bool, List[CausalRow]] = defaultdict(list)
    for row in rows:
        by_label[row.label_true].append(row)

    true_rows = by_label.get(True, [])
    false_rows = by_label.get(False, [])
    if not true_rows or not false_rows:
        raise ValueError("Need both true and false rows to build a balanced dataset.")

    rng = random.Random(seed)
    rng.shuffle(true_rows)
    rng.shuffle(false_rows)

    per_label_max = min(len(true_rows), len(false_rows))
    if target_size is None:
        per_label = per_label_max
    else:
        if target_size < 2:
            raise ValueError("--target-size must be >= 2")
        per_label = min(per_label_max, target_size // 2)

    selected_true = sample_label_category_balanced(true_rows, target=per_label, rng=rng)
    selected_false = sample_label_category_balanced(false_rows, target=per_label, rng=rng)
    selected = selected_true + selected_false
    rng.shuffle(selected)
    return selected


def sample_label_category_balanced(
    rows: Sequence[CausalRow],
    target: int,
    rng: random.Random,
) -> List[CausalRow]:
    by_category: Dict[str, List[CausalRow]] = defaultdict(list)
    for row in rows:
        by_category[row.category].append(row)

    for cat_rows in by_category.values():
        rng.shuffle(cat_rows)

    category_sizes = {cat: len(cat_rows) for cat, cat_rows in by_category.items()}
    quotas = equal_quota_allocation(category_sizes, target)

    selected: List[CausalRow] = []
    for category in sorted(by_category.keys()):
        selected.extend(by_category[category][: quotas[category]])

    return selected


def equal_quota_allocation(group_sizes: Dict[str, int], target: int) -> Dict[str, int]:
    """Allocate target almost equally across groups with capacity limits."""
    groups = sorted(group_sizes.keys())
    quotas = {group: 0 for group in groups}
    remaining = target

    while remaining > 0:
        eligible = [group for group in groups if quotas[group] < group_sizes[group]]
        if not eligible:
            break

        per_group = max(1, remaining // len(eligible))
        used = 0
        for group in eligible:
            remaining_now = remaining - used
            if remaining_now <= 0:
                break
            add = min(per_group, group_sizes[group] - quotas[group], remaining_now)
            quotas[group] += add
            used += add

        if used == 0:
            break
        remaining -= used

    if remaining > 0:
        for group in groups:
            while remaining > 0 and quotas[group] < group_sizes[group]:
                quotas[group] += 1
                remaining -= 1

    return quotas


def summarize(rows: Sequence[CausalRow]) -> dict:
    label_counts = Counter(row.label_true for row in rows)
    by_category: Dict[str, Counter] = defaultdict(Counter)
    by_file: Dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        by_category[row.category][row.label_true] += 1
        by_file[row.source_file][row.label_true] += 1

    def format_counter(counter: Counter) -> dict:
        return {
            "true": counter.get(True, 0),
            "false": counter.get(False, 0),
            "total": counter.get(True, 0) + counter.get(False, 0),
        }

    return {
        "total": len(rows),
        "labels": format_counter(label_counts),
        "categories": {k: format_counter(v) for k, v in sorted(by_category.items())},
        "files": {k: format_counter(v) for k, v in sorted(by_file.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default="eval/causal/*.csv",
        help="Input CSV glob (default: eval/causal/*.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/causal_experiment",
        help="Output directory (default: data/causal_experiment)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balancing/shuffling (default: 42)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Optional target total size of balanced set (must be >=2; rounded down to even)",
    )
    parser.add_argument(
        "--noise-texts-per-sample",
        type=int,
        default=5,
        help="Number of distractor/noise texts to append per sample (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in sorted(Path().glob(args.input_glob))]
    if not input_paths:
        raise SystemExit(f"No input files matched: {args.input_glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_causal_rows(input_paths)
    balanced_rows = build_balanced(all_rows, seed=args.seed, target_size=args.target_size)
    if args.noise_texts_per_sample < 0:
        raise ValueError("--noise-texts-per-sample must be >= 0")

    noise_pool = build_noise_pool(all_rows)
    all_rng = random.Random(args.seed + 101)
    balanced_rng = random.Random(args.seed + 202)

    all_records = [
        to_jsonl_record(row, noise_pool, args.noise_texts_per_sample, all_rng)
        for row in all_rows
    ]
    balanced_records = [
        to_jsonl_record(row, noise_pool, args.noise_texts_per_sample, balanced_rng)
        for row in balanced_rows
    ]

    all_path = output_dir / "causal_qa_all.jsonl"
    balanced_path = output_dir / "causal_qa_balanced.jsonl"
    manifest_path = output_dir / "causal_qa_manifest.jsonl"
    summary_path = output_dir / "causal_qa_summary.json"

    write_jsonl(all_path, all_records)
    write_jsonl(balanced_path, balanced_records)
    write_manifest(manifest_path, balanced_rows)

    summary = {
        "input_glob": args.input_glob,
        "seed": args.seed,
        "target_size": args.target_size,
        "noise_texts_per_sample": args.noise_texts_per_sample,
        "all": summarize(all_rows),
        "balanced": summarize(balanced_rows),
        "outputs": {
            "all_jsonl": str(all_path),
            "balanced_jsonl": str(balanced_path),
            "manifest_jsonl": str(manifest_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[prepare_causal_experiment_data] all: {len(all_rows)} -> {all_path}")
    print(f"[prepare_causal_experiment_data] balanced: {len(balanced_rows)} -> {balanced_path}")
    print(f"[prepare_causal_experiment_data] manifest: {manifest_path}")
    print(f"[prepare_causal_experiment_data] summary: {summary_path}")


if __name__ == "__main__":
    main()
