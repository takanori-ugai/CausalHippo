#!/usr/bin/env python3
"""Prepare MuSiQue experiment data for multi-hop retrieval evaluation.

This script reads MuSiQue answerable JSONL, computes per-example metadata,
and writes experiment-ready subsets:
- full multi-hop dev set
- hop-specific splits
- lexical-overlap splits (low/mid/high)
- balanced subset for low-cost runs
- hard-negative augmented balanced subset
- metadata manifest and summary
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


TOKEN_RE = re.compile(r"[a-z0-9]+")
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


@dataclass
class ExampleMeta:
    id: str
    hop_count: int
    support_indices: List[int]
    support_count: int
    overlap_score: float
    overlap_bucket: str


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


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def compute_metadata(examples: Sequence[dict]) -> Tuple[Dict[str, ExampleMeta], float, float]:
    raw: List[Tuple[str, int, List[int], int, float]] = []
    overlaps: List[float] = []

    for ex in examples:
        qd = ex.get("question_decomposition")
        if not isinstance(qd, list) or len(qd) < 2:
            continue
        paragraphs = ex.get("paragraphs", [])
        supports = [p for p in paragraphs if p.get("is_supporting")]
        support_indices = [int(p.get("idx", i)) for i, p in enumerate(supports)]

        q_tokens = tokenize(ex.get("question", ""))
        support_scores: List[float] = []
        for p in supports:
            p_tokens = tokenize(p.get("paragraph_text", ""))
            support_scores.append(jaccard(q_tokens, p_tokens))
        overlap_score = sum(support_scores) / len(support_scores) if support_scores else 0.0

        hop_count = len(qd)
        support_count = len(supports)
        raw.append((ex["id"], hop_count, support_indices, support_count, overlap_score))
        overlaps.append(overlap_score)

    sorted_overlaps = sorted(overlaps)
    q1 = percentile(sorted_overlaps, 1.0 / 3.0)
    q2 = percentile(sorted_overlaps, 2.0 / 3.0)

    meta: Dict[str, ExampleMeta] = {}
    for ex_id, hop_count, support_indices, support_count, overlap_score in raw:
        if overlap_score <= q1:
            bucket = "low"
        elif overlap_score <= q2:
            bucket = "mid"
        else:
            bucket = "high"
        meta[ex_id] = ExampleMeta(
            id=ex_id,
            hop_count=hop_count,
            support_indices=support_indices,
            support_count=support_count,
            overlap_score=overlap_score,
            overlap_bucket=bucket,
        )

    return meta, q1, q2


def assign_hop_overlap_files(
    examples: Sequence[dict],
    meta: Dict[str, ExampleMeta],
) -> Dict[str, List[dict]]:
    outputs: Dict[str, List[dict]] = {
        "musique_dev_multihop_all.jsonl": [],
        "musique_dev_hop2.jsonl": [],
        "musique_dev_hop3.jsonl": [],
        "musique_dev_hop4plus.jsonl": [],
        "musique_dev_overlap_low.jsonl": [],
        "musique_dev_overlap_mid.jsonl": [],
        "musique_dev_overlap_high.jsonl": [],
    }

    for ex in examples:
        m = meta.get(ex.get("id"))
        if m is None:
            continue
        outputs["musique_dev_multihop_all.jsonl"].append(ex)

        if m.hop_count == 2:
            outputs["musique_dev_hop2.jsonl"].append(ex)
        elif m.hop_count == 3:
            outputs["musique_dev_hop3.jsonl"].append(ex)
        else:
            outputs["musique_dev_hop4plus.jsonl"].append(ex)

        outputs[f"musique_dev_overlap_{m.overlap_bucket}.jsonl"].append(ex)

    return outputs


def equal_quota_allocation(group_sizes: Dict[int, int], target: int) -> Dict[int, int]:
    """Allocate target as equally as possible across hop groups with capacity limits."""
    hops = sorted(group_sizes.keys())
    quotas = {h: 0 for h in hops}
    remaining = target

    while remaining > 0:
        eligible = [h for h in hops if quotas[h] < group_sizes[h]]
        if not eligible:
            break
        per = max(1, remaining // len(eligible))
        used = 0
        for h in eligible:
            add = min(per, group_sizes[h] - quotas[h])
            quotas[h] += add
            used += add
        if used == 0:
            break
        remaining -= used

    if remaining > 0:
        for h in hops:
            while remaining > 0 and quotas[h] < group_sizes[h]:
                quotas[h] += 1
                remaining -= 1

    return quotas


def build_balanced_subset(
    examples: Sequence[dict],
    meta: Dict[str, ExampleMeta],
    target_size: int,
    rng: random.Random,
) -> List[dict]:
    by_hop: Dict[int, List[dict]] = defaultdict(list)
    for ex in examples:
        m = meta.get(ex.get("id"))
        if m is None:
            continue
        by_hop[m.hop_count].append(ex)

    for hop_examples in by_hop.values():
        rng.shuffle(hop_examples)

    group_sizes = {hop: len(items) for hop, items in by_hop.items()}
    target = min(target_size, sum(group_sizes.values()))
    quotas = equal_quota_allocation(group_sizes, target)

    selected: List[dict] = []
    for hop in sorted(by_hop.keys()):
        selected.extend(by_hop[hop][: quotas[hop]])

    rng.shuffle(selected)
    return selected


def add_hard_negatives(
    examples: Sequence[dict],
    full_examples: Sequence[dict],
    hard_negatives_per_sample: int,
) -> List[dict]:
    """Append high lexical-overlap paragraphs from other examples as hard negatives."""
    pool: List[dict] = []
    token_to_positions: Dict[str, List[int]] = defaultdict(list)

    for ex in full_examples:
        source_id = ex["id"]
        for p in ex.get("paragraphs", []):
            if p.get("is_supporting"):
                # Skip globally supporting paragraphs to reduce accidental positives.
                continue
            text = p.get("paragraph_text", "")
            tokens = tokenize(text)
            if not tokens:
                continue
            entry = {
                "source_id": source_id,
                "title": p.get("title", ""),
                "paragraph_text": text,
                "tokens": tokens,
            }
            pos = len(pool)
            pool.append(entry)
            for tok in tokens:
                token_to_positions[tok].append(pos)

    augmented: List[dict] = []

    for ex in examples:
        q_tokens = tokenize(ex.get("question", ""))
        existing_texts = {p.get("paragraph_text", "") for p in ex.get("paragraphs", [])}
        candidate_positions: Set[int] = set()
        for tok in q_tokens:
            candidate_positions.update(token_to_positions.get(tok, []))

        scored: List[Tuple[float, int]] = []
        for pos in candidate_positions:
            cand = pool[pos]
            if cand["source_id"] == ex["id"]:
                continue
            if cand["paragraph_text"] in existing_texts:
                continue
            c_tokens = cand["tokens"]
            inter = len(q_tokens & c_tokens)
            if inter == 0:
                continue
            # Overlap ratio anchored by question length to prefer lexically similar distractions.
            score = inter / max(1, len(q_tokens))
            scored.append((score, pos))

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen_positions = [pos for _, pos in scored[:hard_negatives_per_sample]]

        new_obj = json.loads(json.dumps(ex, ensure_ascii=False))
        paragraphs = list(new_obj.get("paragraphs", []))
        next_idx = 0
        if paragraphs:
            next_idx = max(int(p.get("idx", i)) for i, p in enumerate(paragraphs)) + 1

        for pos in chosen_positions:
            cand = pool[pos]
            paragraphs.append(
                {
                    "idx": next_idx,
                    "title": f"{cand['title']} [hard-negative]",
                    "paragraph_text": cand["paragraph_text"],
                    "is_supporting": False,
                    "hard_negative": True,
                    "hard_negative_source_id": cand["source_id"],
                }
            )
            next_idx += 1

        new_obj["paragraphs"] = paragraphs
        augmented.append(new_obj)

    return augmented


def write_manifest(
    output_path: Path,
    meta: Dict[str, ExampleMeta],
) -> None:
    rows = []
    for ex_id in sorted(meta.keys()):
        m = meta[ex_id]
        rows.append(
            {
                "id": m.id,
                "hop_count": m.hop_count,
                "support_indices": m.support_indices,
                "support_count": m.support_count,
                "lexical_overlap_score": round(m.overlap_score, 8),
                "overlap_bucket": m.overlap_bucket,
            }
        )
    write_jsonl(output_path, rows)


def write_summary(
    output_path: Path,
    meta: Dict[str, ExampleMeta],
    generated_files: Dict[str, int],
    q1: float,
    q2: float,
) -> None:
    hop_counts: Dict[int, int] = defaultdict(int)
    overlap_counts: Dict[str, int] = defaultdict(int)
    for m in meta.values():
        hop_counts[m.hop_count] += 1
        overlap_counts[m.overlap_bucket] += 1

    lines = []
    lines.append("# MuSiQue Experiment Data Summary")
    lines.append("")
    lines.append(f"Total multihop examples: {len(meta)}")
    lines.append(f"Lexical overlap tertiles: q1={q1:.6f}, q2={q2:.6f}")
    lines.append("")
    lines.append("## Hop counts")
    for hop in sorted(hop_counts.keys()):
        lines.append(f"- hop={hop}: {hop_counts[hop]}")
    lines.append("")
    lines.append("## Overlap bucket counts")
    for bucket in OVERLAP_BUCKETS:
        lines.append(f"- {bucket}: {overlap_counts.get(bucket, 0)}")
    lines.append("")
    lines.append("## Generated files")
    for name in sorted(generated_files.keys()):
        lines.append(f"- {name}: {generated_files[name]} examples")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MuSiQue experiment datasets.")
    parser.add_argument(
        "--input",
        default="MuSiQue/data/musique_ans_v1.0_dev.jsonl",
        help="Input MuSiQue JSONL file (default: dev set)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/musique_experiment",
        help="Output directory",
    )
    parser.add_argument(
        "--balanced-size",
        type=int,
        default=300,
        help="Target size for balanced subset",
    )
    parser.add_argument(
        "--hard-negatives-per-sample",
        type=int,
        default=3,
        help="How many hard-negative paragraphs to append to each balanced sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_examples = read_jsonl(input_path)
    examples = [ex for ex in raw_examples if ex.get("answerable", True)]

    meta, q1, q2 = compute_metadata(examples)
    multihop_examples = [ex for ex in examples if ex.get("id") in meta]

    files = assign_hop_overlap_files(multihop_examples, meta)

    rng = random.Random(args.seed)
    balanced = build_balanced_subset(multihop_examples, meta, args.balanced_size, rng)
    balanced_name = f"musique_dev_balanced_{args.balanced_size}.jsonl"
    files[balanced_name] = balanced

    if args.hard_negatives_per_sample > 0:
        hardneg = add_hard_negatives(
            examples=balanced,
            full_examples=multihop_examples,
            hard_negatives_per_sample=args.hard_negatives_per_sample,
        )
        hardneg_name = (
            f"musique_dev_balanced_{args.balanced_size}_hardneg{args.hard_negatives_per_sample}.jsonl"
        )
        files[hardneg_name] = hardneg

    generated_counts: Dict[str, int] = {}
    for name, rows in files.items():
        out = output_dir / name
        write_jsonl(out, rows)
        generated_counts[name] = len(rows)

    write_manifest(output_dir / "musique_dev_multihop_manifest.jsonl", meta)
    generated_counts["musique_dev_multihop_manifest.jsonl"] = len(meta)

    write_summary(
        output_dir / "README.md",
        meta=meta,
        generated_files=generated_counts,
        q1=q1,
        q2=q2,
    )

    print(f"Prepared MuSiQue experiment data in: {output_dir}")
    for name in sorted(generated_counts.keys()):
        print(f"  {name}: {generated_counts[name]}")


if __name__ == "__main__":
    main()
