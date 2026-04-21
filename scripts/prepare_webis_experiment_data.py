#!/usr/bin/env python3
"""Prepare Webis-CausalQA experiment data for MuSiQue-style evaluation.

Input format:
  Webis/Webis-CausalQA-22-v-2.0/input/original_splits/*_train_*.csv

This script converts Webis CSV rows into the JSONL schema used by
`MultiConditionExperiment` and generates:
- balanced source-aware subset
- optional full/source/overlap splits (large outputs)
- optional hard-negative augmented balanced subset
- manifest and summary

Design note:
Webis train data is large, so processing is done in streaming two-pass mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
HTML_SENTENCE_BREAK_RE = re.compile(r"(?i)</li>|</p>|<br\\s*/?>")
OVERLAP_BUCKETS = ("low", "mid", "high")
STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}


@dataclass(frozen=True)
class WebisRow:
    source: str
    source_file: str
    row_index: int
    original_id: str
    question: str
    answer: str
    answer_processed: str
    paragraph_text: str
    paragraph_source: str
    context_missing: bool


@dataclass(frozen=True)
class ScanStats:
    total_rows: int
    source_counts: Dict[str, int]
    context_missing_counts: Dict[str, int]
    q1: float
    q2: float


@dataclass(frozen=True)
class NegativeCandidate:
    sample_id: str
    source: str
    paragraph_text: str
    title: str
    tokens: Set[str]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> Set[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return {tok for tok in tokens if tok and tok not in STOPWORDS}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def percentile(sorted_values: Sequence[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * ratio
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


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
            add = min(per_group, group_sizes[group] - quotas[group], remaining - used)
            if add <= 0:
                continue
            quotas[group] += add
            used += add
            if used >= remaining:
                break

        if used == 0:
            break
        remaining -= used

    if remaining > 0:
        for group in groups:
            while remaining > 0 and quotas[group] < group_sizes[group]:
                quotas[group] += 1
                remaining -= 1

    return quotas


def overlap_bucket(score: float, q1: float, q2: float) -> str:
    if score <= q1:
        return "low"
    if score <= q2:
        return "mid"
    return "high"


def derive_source_name(path: Path) -> str:
    stem = path.stem
    marker = "_train_"
    pos = stem.find(marker)
    if pos > 0:
        return stem[:pos]
    return stem


def choose_paragraph_text(row: dict) -> Tuple[str, str, bool]:
    context = normalize_text(row.get("context", ""))
    context_processed = normalize_text(row.get("context_processed", ""))
    answer = normalize_text(row.get("answer", ""))

    if context:
        return context, "context", False
    if context_processed:
        return context_processed, "context_processed", False
    if answer:
        return answer, "answer_fallback", True
    return "", "missing", True


def split_text_by_words(text: str, max_words: int) -> List[str]:
    if not text.strip():
        return []

    def split_oversized_unit(unit: str) -> List[str]:
        words = unit.split()
        if len(words) <= max_words:
            return [unit]

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            cut = end
            for i in range(end - 1, start - 1, -1):
                if words[i].endswith((".", "!", "?", ";", ":")):
                    cut = i + 1
                    break
            if cut <= start:
                cut = end
            chunks.append(" ".join(words[start:cut]))
            start = cut
        return chunks

    prepared = normalize_text(text)
    prepared = HTML_SENTENCE_BREAK_RE.sub(". ", prepared)

    sentences = [
        normalize_text(s)
        for s in SENTENCE_BOUNDARY_RE.split(prepared)
        if normalize_text(s)
    ]
    if not sentences:
        return split_oversized_unit(prepared)

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_words = 0

    for sentence in sentences:
        for unit in split_oversized_unit(sentence):
            unit_words = len(unit.split())
            if current_sentences and current_words + unit_words > max_words:
                chunks.append(" ".join(current_sentences))
                current_sentences = [unit]
                current_words = unit_words
            else:
                current_sentences.append(unit)
                current_words += unit_words

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def detect_support_indices(chunks: Sequence[str], answer_candidates: Sequence[str]) -> Set[int]:
    if not chunks:
        return set()

    normalized_answers = [normalize_text(a).lower() for a in answer_candidates if normalize_text(a)]
    tokenized_answers = [tokenize(a) for a in normalized_answers]
    normalized_chunks = [normalize_text(c).lower() for c in chunks]
    tokenized_chunks = [tokenize(c) for c in chunks]

    best_idx = 0
    best_score = -1.0

    for idx, chunk_text in enumerate(normalized_chunks):
        chunk_tokens = tokenized_chunks[idx]
        for answer_text, answer_tokens in zip(normalized_answers, tokenized_answers):
            if not answer_tokens:
                continue
            phrase_hit = 1.0 if answer_text in chunk_text else 0.0
            token_overlap = len(answer_tokens & chunk_tokens) / len(answer_tokens)
            score = (phrase_hit * 2.0) + token_overlap
            if score > best_score:
                best_score = score
                best_idx = idx

    return {best_idx}


def iter_webis_rows(paths: Sequence[Path], max_rows_per_file: int | None = None) -> Iterable[WebisRow]:
    csv.field_size_limit(1_000_000_000)

    for path in sorted(paths):
        source = derive_source_name(path)
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue

            yielded = 0
            for raw_index, raw in enumerate(reader):
                if max_rows_per_file is not None and yielded >= max_rows_per_file:
                    break
                if not raw:
                    continue

                question_raw = normalize_text(raw.get("question", ""))
                question_processed = normalize_text(raw.get("question_processed", ""))
                question = question_raw if question_raw else question_processed

                answer_raw = normalize_text(raw.get("answer", ""))
                answer_processed = normalize_text(raw.get("answer_processed", ""))
                answer = answer_raw if answer_raw else answer_processed

                paragraph_text, paragraph_source, context_missing = choose_paragraph_text(raw)

                if not question:
                    continue
                if not answer:
                    continue
                if not paragraph_text:
                    continue

                original_id = normalize_text(raw.get("id", "")) or str(raw_index)

                yield WebisRow(
                    source=source,
                    source_file=path.name,
                    row_index=raw_index,
                    original_id=original_id,
                    question=question,
                    answer=answer,
                    answer_processed=answer_processed,
                    paragraph_text=paragraph_text,
                    paragraph_source=paragraph_source,
                    context_missing=context_missing,
                )
                yielded += 1


def compute_scan_stats(paths: Sequence[Path], max_rows_per_file: int | None) -> ScanStats:
    overlaps: List[float] = []
    source_counts: Counter[str] = Counter()
    context_missing_counts: Counter[str] = Counter()

    total_rows = 0
    for row in iter_webis_rows(paths, max_rows_per_file=max_rows_per_file):
        total_rows += 1
        source_counts[row.source] += 1
        if row.context_missing:
            context_missing_counts[row.source] += 1

        q_tokens = tokenize(row.question)
        p_tokens = tokenize(row.paragraph_text)
        overlaps.append(jaccard(q_tokens, p_tokens))

    if total_rows == 0:
        raise ValueError("No valid rows found in input CSV files.")

    sorted_overlaps = sorted(overlaps)
    q1 = percentile(sorted_overlaps, 1.0 / 3.0)
    q2 = percentile(sorted_overlaps, 2.0 / 3.0)

    return ScanStats(
        total_rows=total_rows,
        source_counts=dict(source_counts),
        context_missing_counts=dict(context_missing_counts),
        q1=q1,
        q2=q2,
    )


def to_record(row: WebisRow, max_words_per_paragraph: int) -> dict:
    sample_id = f"{row.source}_{row.row_index}"
    aliases: List[str] = []
    if row.answer_processed and row.answer_processed.lower() != row.answer.lower():
        aliases.append(row.answer_processed)

    chunks = split_text_by_words(row.paragraph_text, max_words=max_words_per_paragraph)
    if not chunks:
        chunks = [row.paragraph_text]

    support_indices = detect_support_indices(chunks, [row.answer, row.answer_processed])
    paragraphs = []
    for idx, chunk in enumerate(chunks):
        paragraphs.append(
            {
                "idx": idx,
                "title": row.source,
                "paragraph_text": chunk,
                "is_supporting": idx in support_indices,
            }
        )

    return {
        "id": sample_id,
        "paragraphs": paragraphs,
        "question": row.question,
        "answer": row.answer,
        "answer_aliases": aliases,
        "answerable": True,
        "metadata": {
            "source_dataset": row.source,
            "category": row.source,
            "source_file": row.source_file,
            "row_index": row.row_index,
            "original_id": row.original_id,
            "context_missing": row.context_missing,
            "paragraph_source": row.paragraph_source,
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def reservoir_add(pool: List[NegativeCandidate], candidate: NegativeCandidate, seen: int, max_size: int, rng: random.Random) -> int:
    seen += 1
    if max_size <= 0:
        return seen
    if len(pool) < max_size:
        pool.append(candidate)
        return seen

    replacement_pos = rng.randrange(seen)
    if replacement_pos < max_size:
        pool[replacement_pos] = candidate
    return seen


def add_hard_negatives(
    examples: Sequence[dict],
    pool: Sequence[NegativeCandidate],
    hard_negatives_per_sample: int,
    rng: random.Random,
) -> List[dict]:
    if hard_negatives_per_sample <= 0:
        return list(examples)
    if not pool:
        return [json.loads(json.dumps(ex, ensure_ascii=False)) for ex in examples]

    augmented: List[dict] = []

    for ex in examples:
        q_tokens = tokenize(ex.get("question", ""))
        existing_texts = {p.get("paragraph_text", "") for p in ex.get("paragraphs", [])}
        scored: List[Tuple[float, int]] = []

        for i, candidate in enumerate(pool):
            if candidate.sample_id == ex.get("id"):
                continue
            if candidate.paragraph_text in existing_texts:
                continue

            inter = len(q_tokens & candidate.tokens)
            if inter == 0:
                continue
            score = inter / max(1, len(q_tokens))
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [pool[idx] for _, idx in scored[:hard_negatives_per_sample]]

        if len(chosen) < hard_negatives_per_sample:
            remaining = hard_negatives_per_sample - len(chosen)
            candidates = [
                c
                for c in pool
                if c.sample_id != ex.get("id") and c.paragraph_text not in existing_texts
            ]
            rng.shuffle(candidates)
            chosen.extend(candidates[:remaining])

        new_obj = json.loads(json.dumps(ex, ensure_ascii=False))
        paragraphs = list(new_obj.get("paragraphs", []))
        next_idx = 0
        if paragraphs:
            next_idx = max(int(p.get("idx", i)) for i, p in enumerate(paragraphs)) + 1

        for cand in chosen:
            paragraphs.append(
                {
                    "idx": next_idx,
                    "title": f"{cand.title} [hard-negative]",
                    "paragraph_text": cand.paragraph_text,
                    "is_supporting": False,
                    "hard_negative": True,
                    "hard_negative_source": cand.source,
                    "hard_negative_source_id": cand.sample_id,
                }
            )
            next_idx += 1

        new_obj["paragraphs"] = paragraphs
        augmented.append(new_obj)

    return augmented


def pass_two_process(
    *,
    paths: Sequence[Path],
    output_dir: Path,
    q1: float,
    q2: float,
    source_quotas: Dict[str, int],
    max_rows_per_file: int | None,
    seed: int,
    hard_negative_pool_size: int,
    max_words_per_paragraph: int,
    write_all: bool,
    write_source_splits: bool,
    write_overlap_splits: bool,
) -> Tuple[List[dict], Dict[str, int], Dict[str, int], List[NegativeCandidate]]:
    manifest_name = "webis_train_manifest.jsonl"
    all_name = "webis_train_all.jsonl"

    selected_by_source: Dict[str, List[dict]] = {
        source: [] for source, quota in source_quotas.items() if quota > 0
    }
    seen_by_source: Dict[str, int] = defaultdict(int)

    generated_counts: Dict[str, int] = defaultdict(int)
    overlap_counts: Dict[str, int] = defaultdict(int)

    pool: List[NegativeCandidate] = []
    pool_seen = 0

    rng = random.Random(seed)

    writers: Dict[str, object] = {}
    try:
        manifest_path = output_dir / manifest_name
        manifest_f = manifest_path.open("w", encoding="utf-8")
        writers[manifest_name] = manifest_f

        all_f = None
        if write_all:
            all_f = (output_dir / all_name).open("w", encoding="utf-8")
            writers[all_name] = all_f

        source_writers: Dict[str, object] = {}
        if write_source_splits:
            for source in sorted(source_quotas.keys()):
                name = f"webis_train_source_{source}.jsonl"
                source_writers[source] = (output_dir / name).open("w", encoding="utf-8")
                writers[name] = source_writers[source]

        overlap_writers: Dict[str, object] = {}
        if write_overlap_splits:
            for bucket in OVERLAP_BUCKETS:
                name = f"webis_train_overlap_{bucket}.jsonl"
                overlap_writers[bucket] = (output_dir / name).open("w", encoding="utf-8")
                writers[name] = overlap_writers[bucket]

        for row in iter_webis_rows(paths, max_rows_per_file=max_rows_per_file):
            record = to_record(row, max_words_per_paragraph=max_words_per_paragraph)
            sample_id = record["id"]
            q_tokens = tokenize(row.question)
            p_tokens = tokenize(row.paragraph_text)
            score = jaccard(q_tokens, p_tokens)
            bucket = overlap_bucket(score, q1, q2)
            overlap_counts[bucket] += 1

            manifest_row = {
                "id": sample_id,
                "source_dataset": row.source,
                "source_file": row.source_file,
                "row_index": row.row_index,
                "original_id": row.original_id,
                "lexical_overlap_score": round(score, 8),
                "overlap_bucket": bucket,
                "question_token_count": len(q_tokens),
                "paragraph_token_count": len(p_tokens),
                "context_missing": row.context_missing,
                "paragraph_source": row.paragraph_source,
            }
            manifest_f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
            generated_counts[manifest_name] += 1

            if all_f is not None:
                all_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                generated_counts[all_name] += 1

            if write_source_splits:
                source_writer = source_writers[row.source]
                source_name = f"webis_train_source_{row.source}.jsonl"
                source_writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                generated_counts[source_name] += 1

            if write_overlap_splits:
                overlap_name = f"webis_train_overlap_{bucket}.jsonl"
                overlap_writers[bucket].write(json.dumps(record, ensure_ascii=False) + "\n")
                generated_counts[overlap_name] += 1

            quota = source_quotas.get(row.source, 0)
            if quota > 0:
                seen_by_source[row.source] += 1
                selected = selected_by_source[row.source]
                if len(selected) < quota:
                    selected.append(record)
                else:
                    replacement_idx = rng.randrange(seen_by_source[row.source])
                    if replacement_idx < quota:
                        selected[replacement_idx] = record

            if hard_negative_pool_size > 0 and p_tokens:
                for paragraph in record.get("paragraphs", []):
                    if paragraph.get("is_supporting"):
                        continue
                    paragraph_text = normalize_text(paragraph.get("paragraph_text", ""))
                    paragraph_tokens = tokenize(paragraph_text)
                    if not paragraph_tokens:
                        continue
                    cand = NegativeCandidate(
                        sample_id=sample_id,
                        source=row.source,
                        paragraph_text=paragraph_text,
                        title=paragraph.get("title", row.source),
                        tokens=paragraph_tokens,
                    )
                    pool_seen = reservoir_add(
                        pool=pool,
                        candidate=cand,
                        seen=pool_seen,
                        max_size=hard_negative_pool_size,
                        rng=rng,
                    )

        balanced: List[dict] = []
        for source in sorted(selected_by_source.keys()):
            balanced.extend(selected_by_source[source][: source_quotas[source]])
        rng.shuffle(balanced)

        return balanced, dict(generated_counts), dict(overlap_counts), pool
    finally:
        for file_obj in writers.values():
            file_obj.close()


def write_summary(
    output_path: Path,
    *,
    scan: ScanStats,
    source_quotas: Dict[str, int],
    overlap_counts: Dict[str, int],
    generated_files: Dict[str, int],
    balanced_size: int,
    hard_negatives_per_sample: int,
    hard_negative_pool_size: int,
    max_words_per_paragraph: int,
    write_all: bool,
    write_source_splits: bool,
    write_overlap_splits: bool,
) -> None:
    lines: List[str] = []
    lines.append("# Webis Experiment Data Summary")
    lines.append("")
    lines.append(f"Total rows converted: {scan.total_rows}")
    lines.append(f"Lexical overlap tertiles: q1={scan.q1:.6f}, q2={scan.q2:.6f}")
    lines.append("")
    lines.append("## Source counts")
    for source in sorted(scan.source_counts.keys()):
        total = scan.source_counts[source]
        missing = scan.context_missing_counts.get(source, 0)
        lines.append(f"- {source}: {total} (context-missing fallback: {missing})")
    lines.append("")
    lines.append("## Balanced subset quotas")
    for source in sorted(source_quotas.keys()):
        lines.append(f"- {source}: {source_quotas[source]}")
    lines.append("")
    lines.append("## Overlap bucket counts")
    for bucket in OVERLAP_BUCKETS:
        lines.append(f"- {bucket}: {overlap_counts.get(bucket, 0)}")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- balanced_size: {balanced_size}")
    lines.append(f"- max_words_per_paragraph: {max_words_per_paragraph}")
    lines.append(f"- hard_negatives_per_sample: {hard_negatives_per_sample}")
    lines.append(f"- hard_negative_pool_size: {hard_negative_pool_size}")
    lines.append(f"- write_all: {write_all}")
    lines.append(f"- write_source_splits: {write_source_splits}")
    lines.append(f"- write_overlap_splits: {write_overlap_splits}")
    lines.append("")
    lines.append("## Generated files")
    for name in sorted(generated_files.keys()):
        lines.append(f"- {name}: {generated_files[name]} rows")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default="Webis/Webis-CausalQA-22-v-2.0/input/original_splits/*_train_*.csv",
        help="Input CSV glob for Webis train files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/webis_experiment",
        help="Output directory",
    )
    parser.add_argument(
        "--balanced-size",
        type=int,
        default=300,
        help="Target size of source-balanced subset",
    )
    parser.add_argument(
        "--max-words-per-paragraph",
        type=int,
        default=150,
        help="Maximum words per paragraph chunk while keeping sentence boundaries (default: 150)",
    )
    parser.add_argument(
        "--hard-negatives-per-sample",
        type=int,
        default=3,
        help="How many hard negatives to append to each balanced sample",
    )
    parser.add_argument(
        "--hard-negative-pool-size",
        type=int,
        default=20000,
        help="Reservoir size for hard-negative candidate contexts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--write-all",
        action="store_true",
        help="Write webis_train_all.jsonl (large output)",
    )
    parser.add_argument(
        "--write-source-splits",
        action="store_true",
        help="Write one JSONL per source dataset (large outputs)",
    )
    parser.add_argument(
        "--write-overlap-splits",
        action="store_true",
        help="Write overlap low/mid/high JSONL files (large outputs)",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional debug cap for rows read from each CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.balanced_size <= 0:
        raise ValueError("--balanced-size must be > 0")
    if args.hard_negatives_per_sample < 0:
        raise ValueError("--hard-negatives-per-sample must be >= 0")
    if args.max_words_per_paragraph <= 0:
        raise ValueError("--max-words-per-paragraph must be > 0")
    if args.hard_negative_pool_size < 0:
        raise ValueError("--hard-negative-pool-size must be >= 0")
    if args.max_rows_per_file is not None and args.max_rows_per_file <= 0:
        raise ValueError("--max-rows-per-file must be > 0")

    input_paths = [Path(p) for p in sorted(Path().glob(args.input_glob))]
    if not input_paths:
        raise SystemExit(f"No input files matched: {args.input_glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scan = compute_scan_stats(input_paths, max_rows_per_file=args.max_rows_per_file)
    target_balanced = min(args.balanced_size, scan.total_rows)
    source_quotas = equal_quota_allocation(scan.source_counts, target_balanced)

    balanced, generated_counts, overlap_counts, negative_pool = pass_two_process(
        paths=input_paths,
        output_dir=output_dir,
        q1=scan.q1,
        q2=scan.q2,
        source_quotas=source_quotas,
        max_rows_per_file=args.max_rows_per_file,
        seed=args.seed,
        hard_negative_pool_size=args.hard_negative_pool_size,
        max_words_per_paragraph=args.max_words_per_paragraph,
        write_all=args.write_all,
        write_source_splits=args.write_source_splits,
        write_overlap_splits=args.write_overlap_splits,
    )

    balanced_name = f"webis_train_balanced_{target_balanced}.jsonl"
    balanced_count = write_jsonl(output_dir / balanced_name, balanced)
    generated_counts[balanced_name] = balanced_count

    if args.hard_negatives_per_sample > 0:
        hardneg = add_hard_negatives(
            examples=balanced,
            pool=negative_pool,
            hard_negatives_per_sample=args.hard_negatives_per_sample,
            rng=random.Random(args.seed + 999),
        )
        hardneg_name = (
            f"webis_train_balanced_{target_balanced}_hardneg{args.hard_negatives_per_sample}.jsonl"
        )
        hardneg_count = write_jsonl(output_dir / hardneg_name, hardneg)
        generated_counts[hardneg_name] = hardneg_count

    write_summary(
        output_path=output_dir / "README.md",
        scan=scan,
        source_quotas=source_quotas,
        overlap_counts=overlap_counts,
        generated_files=generated_counts,
        balanced_size=target_balanced,
        hard_negatives_per_sample=args.hard_negatives_per_sample,
        hard_negative_pool_size=args.hard_negative_pool_size,
        max_words_per_paragraph=args.max_words_per_paragraph,
        write_all=args.write_all,
        write_source_splits=args.write_source_splits,
        write_overlap_splits=args.write_overlap_splits,
    )
    generated_counts["README.md"] = 1

    print(f"[prepare_webis_experiment_data] converted rows: {scan.total_rows}")
    print(
        "[prepare_webis_experiment_data] overlap tertiles: "
        f"q1={scan.q1:.6f}, q2={scan.q2:.6f}"
    )
    print(f"[prepare_webis_experiment_data] balanced rows: {balanced_count} -> {balanced_name}")
    for name in sorted(generated_counts.keys()):
        print(f"  {name}: {generated_counts[name]}")


if __name__ == "__main__":
    main()
