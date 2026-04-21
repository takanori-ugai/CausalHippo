#!/usr/bin/env python3
"""Aggregate multi-condition experiment JSONL into CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


PER_QUESTION_COLUMNS = [
    "condition",
    "sample_id",
    "hop_count",
    "overlap_bucket",
    "category",
    "label_true",
    "support_count",
    "exact_match",
    "precision",
    "recall",
    "f1",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "support_recall_at1",
    "support_recall_at3",
    "support_recall_at5",
    "bridge_coverage_at5",
    "mrr_at5",
    "ndcg_at5",
    "faithfulness",
    "response_groundedness",
    "index_latency_ms",
    "query_latency_ms",
    "total_latency_ms",
    "prediction",
    "error",
]

SUMMARY_COLUMNS = [
    "condition",
    "samples",
    "errors",
    "exact_match_mean",
    "precision_mean",
    "recall_mean",
    "f1_mean",
    "bertscore_precision_mean",
    "bertscore_recall_mean",
    "bertscore_f1_mean",
    "support_recall_at1_mean",
    "support_recall_at3_mean",
    "support_recall_at5_mean",
    "bridge_coverage_at5_mean",
    "mrr_at5_mean",
    "ndcg_at5_mean",
    "faithfulness_mean",
    "response_groundedness_mean",
    "index_latency_ms_avg",
    "query_latency_ms_avg",
    "query_latency_ms_p50",
    "query_latency_ms_p95",
    "total_latency_ms_avg",
]

CATEGORY_SUMMARY_COLUMNS = [
    "condition",
    "category",
    "samples",
    "errors",
    "labeled_samples",
    "label_true_samples",
    "label_false_samples",
    "exact_match_mean",
    "precision_mean",
    "recall_mean",
    "f1_mean",
    "bertscore_precision_mean",
    "bertscore_recall_mean",
    "bertscore_f1_mean",
    "support_recall_at5_mean",
    "bridge_coverage_at5_mean",
    "mrr_at5_mean",
    "ndcg_at5_mean",
    "faithfulness_mean",
    "response_groundedness_mean",
    "unknown_precision",
    "unknown_recall",
    "unknown_f1",
]

LABEL_AWARE_SUMMARY_COLUMNS = [
    "condition",
    "labeled_samples",
    "label_true_samples",
    "label_false_samples",
    "exact_match_true_mean",
    "exact_match_false_mean",
    "precision_true_mean",
    "precision_false_mean",
    "recall_true_mean",
    "recall_false_mean",
    "f1_true_mean",
    "f1_false_mean",
    "bertscore_precision_true_mean",
    "bertscore_precision_false_mean",
    "bertscore_recall_true_mean",
    "bertscore_recall_false_mean",
    "bertscore_f1_true_mean",
    "bertscore_f1_false_mean",
    "support_recall_at5_true_mean",
    "bridge_coverage_at5_true_mean",
    "mrr_at5_true_mean",
    "ndcg_at5_true_mean",
    "faithfulness_true_mean",
    "response_groundedness_true_mean",
    "unknown_precision",
    "unknown_recall",
    "unknown_f1",
    "category_macro_exact_match_mean",
    "category_macro_precision_mean",
    "category_macro_recall_mean",
    "category_macro_f1_mean",
    "category_macro_bertscore_precision_mean",
    "category_macro_bertscore_recall_mean",
    "category_macro_bertscore_f1_mean",
    "category_macro_unknown_f1",
]


NON_ALNUM_WHITESPACE_REGEX = re.compile(r"[^a-z0-9\s]")
ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b")
MULTISPACE_REGEX = re.compile(r"\s+")
UNKNOWN_ALIASES = [
    "unknown",
    "none",
    "no direct cause",
    "not enough information",
]
UNKNOWN_ALIASES_NORMALIZED = set()
UNKNOWN_PATTERNS = [
    re.compile(r"\bunknown\b"),
    re.compile(r"\bnone\b"),
    re.compile(r"\bno\s+direct\s+cause\b"),
    re.compile(r"\bnot\s+enough\s+information\b"),
    re.compile(r"\b(?:do\s+not|dont|don\s+t)\s+know\b"),
]


for alias in UNKNOWN_ALIASES:
    norm = alias.lower().strip()
    norm = NON_ALNUM_WHITESPACE_REGEX.sub(" ", norm)
    norm = ARTICLES_REGEX.sub(" ", norm)
    norm = MULTISPACE_REGEX.sub(" ", norm).strip()
    if norm:
        UNKNOWN_ALIASES_NORMALIZED.add(norm)


def safe_mean(values: List[float]) -> float:
    finite_values = [v for v in values if math.isfinite(v)]
    if not finite_values:
        return 0.0
    return sum(finite_values) / len(finite_values)


def is_error_row(row: Dict) -> bool:
    return str(row.get("error", "")).strip() != ""


def non_error_rows(rows: List[Dict]) -> List[Dict]:
    return [row for row in rows if not is_error_row(row)]


def percentile(values: List[float], p: float) -> float:
    finite_values = [v for v in values if math.isfinite(v)]
    if not finite_values:
        return 0.0
    vals = sorted(finite_values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def to_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if not math.isfinite(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def parse_optional_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in {"true", "1", "yes", "y"}:
        return True
    if raw in {"false", "0", "no", "n"}:
        return False
    return None


def normalize_text(text: str) -> str:
    lowered = text.lower()
    no_punc = NON_ALNUM_WHITESPACE_REGEX.sub(" ", lowered)
    no_articles = ARTICLES_REGEX.sub(" ", no_punc)
    return MULTISPACE_REGEX.sub(" ", no_articles).strip()


def is_unknown_prediction(prediction: str) -> bool:
    normalized = normalize_text(prediction)
    if not normalized:
        return False
    if normalized in UNKNOWN_ALIASES_NORMALIZED:
        return True
    padded = f" {normalized} "
    for alias in UNKNOWN_ALIASES_NORMALIZED:
        if f" {alias} " in padded:
            return True
    return any(pattern.search(normalized) is not None for pattern in UNKNOWN_PATTERNS)


def read_rows(per_question_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(per_question_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                error_value = obj.get("error")
                if error_value is None:
                    error_value = ""

                prediction = str(obj.get("prediction", ""))
                row = {
                    "condition": str(obj.get("condition", "")),
                    "sample_id": str(obj.get("sampleId", "")),
                    "hop_count": int(to_float(obj.get("hopCount", 0), 0.0)),
                    "overlap_bucket": str(obj.get("overlapBucket", "unknown")),
                    "category": str(obj.get("category", "unknown") or "unknown"),
                    "label_true": parse_optional_bool(obj.get("labelTrue")),
                    "support_count": int(to_float(obj.get("supportCount", 0), 0.0)),
                    "exact_match": to_float(obj.get("exactMatch", 0.0)),
                    "precision": to_float(obj.get("precision", 0.0)),
                    "recall": to_float(obj.get("recall", 0.0)),
                    "f1": to_float(obj.get("f1", 0.0)),
                    "bertscore_precision": to_float(obj.get("bertScorePrecision", 0.0)),
                    "bertscore_recall": to_float(obj.get("bertScoreRecall", 0.0)),
                    "bertscore_f1": to_float(obj.get("bertScoreF1", 0.0)),
                    "support_recall_at1": to_float(obj.get("supportRecallAt1", 0.0)),
                    "support_recall_at3": to_float(obj.get("supportRecallAt3", 0.0)),
                    "support_recall_at5": to_float(obj.get("supportRecallAt5", 0.0)),
                    "bridge_coverage_at5": to_float(obj.get("bridgeCoverageAt5", 0.0)),
                    "mrr_at5": to_float(obj.get("mrrAt5", 0.0)),
                    "ndcg_at5": to_float(obj.get("ndcgAt5", 0.0)),
                    "faithfulness": to_float(obj.get("faithfulness", 0.0)),
                    "response_groundedness": to_float(obj.get("responseGroundedness", 0.0)),
                    "index_latency_ms": to_float(obj.get("indexLatencyMs", 0.0)),
                    "query_latency_ms": to_float(obj.get("queryLatencyMs", 0.0)),
                    "total_latency_ms": to_float(obj.get("totalLatencyMs", 0.0)),
                    "prediction": prediction,
                    "predicted_unknown": is_unknown_prediction(prediction),
                    "error": str(error_value),
                }
                rows.append(row)
    return rows


def write_per_question_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_QUESTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            output_row.pop("predicted_unknown", None)
            writer.writerow(output_row)


def write_summary_csv(path: Path, rows: List[Dict]) -> None:
    by_condition: Dict[str, List[Dict]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()

        for condition in sorted(by_condition.keys()):
            group = by_condition[condition]
            scored = non_error_rows(group)
            em = [float(r["exact_match"]) for r in scored]
            precision = [float(r["precision"]) for r in scored]
            recall = [float(r["recall"]) for r in scored]
            f1 = [float(r["f1"]) for r in scored]
            bertscore_precision = [float(r["bertscore_precision"]) for r in scored]
            bertscore_recall = [float(r["bertscore_recall"]) for r in scored]
            bertscore_f1 = [float(r["bertscore_f1"]) for r in scored]
            r1 = [float(r["support_recall_at1"]) for r in scored]
            r3 = [float(r["support_recall_at3"]) for r in scored]
            r5 = [float(r["support_recall_at5"]) for r in scored]
            b5 = [float(r["bridge_coverage_at5"]) for r in scored]
            mrr5 = [float(r["mrr_at5"]) for r in scored]
            ndcg5 = [float(r["ndcg_at5"]) for r in scored]
            faithfulness = [float(r["faithfulness"]) for r in scored]
            response_groundedness = [float(r["response_groundedness"]) for r in scored]
            idx = [float(r["index_latency_ms"]) for r in scored]
            qry = [float(r["query_latency_ms"]) for r in scored]
            ttl = [float(r["total_latency_ms"]) for r in scored]
            errors = sum(1 for r in group if is_error_row(r))

            writer.writerow(
                {
                    "condition": condition,
                    "samples": len(group),
                    "errors": errors,
                    "exact_match_mean": f"{safe_mean(em):.6f}",
                    "precision_mean": f"{safe_mean(precision):.6f}",
                    "recall_mean": f"{safe_mean(recall):.6f}",
                    "f1_mean": f"{safe_mean(f1):.6f}",
                    "bertscore_precision_mean": f"{safe_mean(bertscore_precision):.6f}",
                    "bertscore_recall_mean": f"{safe_mean(bertscore_recall):.6f}",
                    "bertscore_f1_mean": f"{safe_mean(bertscore_f1):.6f}",
                    "support_recall_at1_mean": f"{safe_mean(r1):.6f}",
                    "support_recall_at3_mean": f"{safe_mean(r3):.6f}",
                    "support_recall_at5_mean": f"{safe_mean(r5):.6f}",
                    "bridge_coverage_at5_mean": f"{safe_mean(b5):.6f}",
                    "mrr_at5_mean": f"{safe_mean(mrr5):.6f}",
                    "ndcg_at5_mean": f"{safe_mean(ndcg5):.6f}",
                    "faithfulness_mean": f"{safe_mean(faithfulness):.6f}",
                    "response_groundedness_mean": f"{safe_mean(response_groundedness):.6f}",
                    "index_latency_ms_avg": f"{safe_mean(idx):.3f}",
                    "query_latency_ms_avg": f"{safe_mean(qry):.3f}",
                    "query_latency_ms_p50": f"{percentile(qry, 0.50):.3f}",
                    "query_latency_ms_p95": f"{percentile(qry, 0.95):.3f}",
                    "total_latency_ms_avg": f"{safe_mean(ttl):.3f}",
                }
            )


def binary_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def unknown_detection_metrics(rows: List[Dict]) -> Dict[str, float]:
    labeled = [r for r in non_error_rows(rows) if r.get("label_true") is not None]
    if not labeled:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = 0
    fp = 0
    fn = 0
    for row in labeled:
        label_true = bool(row["label_true"])
        gold_unknown = not label_true
        pred_unknown = bool(row.get("predicted_unknown", False))
        if pred_unknown and gold_unknown:
            tp += 1
        elif pred_unknown and not gold_unknown:
            fp += 1
        elif (not pred_unknown) and gold_unknown:
            fn += 1

    return binary_prf(tp=tp, fp=fp, fn=fn)


def category_macro_metrics(rows: List[Dict]) -> Dict[str, float]:
    labeled = [r for r in non_error_rows(rows) if r.get("label_true") is not None]
    if not labeled:
        return {
            "exact_match": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "unknown_f1": 0.0,
        }

    by_category: Dict[str, List[Dict]] = {}
    for row in labeled:
        by_category.setdefault(str(row.get("category", "unknown") or "unknown"), []).append(row)

    exact_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    bertscore_precision_scores = []
    bertscore_recall_scores = []
    bertscore_f1_scores = []
    unknown_f1_scores = []
    for category_rows in by_category.values():
        exact_scores.append(safe_mean([float(r["exact_match"]) for r in category_rows]))
        precision_scores.append(safe_mean([float(r["precision"]) for r in category_rows]))
        recall_scores.append(safe_mean([float(r["recall"]) for r in category_rows]))
        f1_scores.append(safe_mean([float(r["f1"]) for r in category_rows]))
        bertscore_precision_scores.append(safe_mean([float(r["bertscore_precision"]) for r in category_rows]))
        bertscore_recall_scores.append(safe_mean([float(r["bertscore_recall"]) for r in category_rows]))
        bertscore_f1_scores.append(safe_mean([float(r["bertscore_f1"]) for r in category_rows]))
        unknown_f1_scores.append(unknown_detection_metrics(category_rows)["f1"])

    return {
        "exact_match": safe_mean(exact_scores),
        "precision": safe_mean(precision_scores),
        "recall": safe_mean(recall_scores),
        "f1": safe_mean(f1_scores),
        "bertscore_precision": safe_mean(bertscore_precision_scores),
        "bertscore_recall": safe_mean(bertscore_recall_scores),
        "bertscore_f1": safe_mean(bertscore_f1_scores),
        "unknown_f1": safe_mean(unknown_f1_scores),
    }


def write_category_summary_csv(path: Path, rows: List[Dict]) -> None:
    by_condition_category: Dict[tuple[str, str], List[Dict]] = {}
    for row in rows:
        key = (row["condition"], str(row.get("category", "unknown") or "unknown"))
        by_condition_category.setdefault(key, []).append(row)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CATEGORY_SUMMARY_COLUMNS)
        writer.writeheader()

        for condition, category in sorted(by_condition_category.keys()):
            group = by_condition_category[(condition, category)]
            scored = non_error_rows(group)
            labeled = [r for r in scored if r.get("label_true") is not None]
            true_rows = [r for r in labeled if bool(r["label_true"])]
            false_rows = [r for r in labeled if not bool(r["label_true"])]
            unknown_metrics = unknown_detection_metrics(scored)
            errors = sum(1 for r in group if is_error_row(r))

            writer.writerow(
                {
                    "condition": condition,
                    "category": category,
                    "samples": len(group),
                    "errors": errors,
                    "labeled_samples": len(labeled),
                    "label_true_samples": len(true_rows),
                    "label_false_samples": len(false_rows),
                    "exact_match_mean": f"{safe_mean([float(r['exact_match']) for r in scored]):.6f}",
                    "precision_mean": f"{safe_mean([float(r['precision']) for r in scored]):.6f}",
                    "recall_mean": f"{safe_mean([float(r['recall']) for r in scored]):.6f}",
                    "f1_mean": f"{safe_mean([float(r['f1']) for r in scored]):.6f}",
                    "bertscore_precision_mean": f"{safe_mean([float(r['bertscore_precision']) for r in scored]):.6f}",
                    "bertscore_recall_mean": f"{safe_mean([float(r['bertscore_recall']) for r in scored]):.6f}",
                    "bertscore_f1_mean": f"{safe_mean([float(r['bertscore_f1']) for r in scored]):.6f}",
                    "support_recall_at5_mean": f"{safe_mean([float(r['support_recall_at5']) for r in scored]):.6f}",
                    "bridge_coverage_at5_mean": f"{safe_mean([float(r['bridge_coverage_at5']) for r in scored]):.6f}",
                    "mrr_at5_mean": f"{safe_mean([float(r['mrr_at5']) for r in scored]):.6f}",
                    "ndcg_at5_mean": f"{safe_mean([float(r['ndcg_at5']) for r in scored]):.6f}",
                    "faithfulness_mean": f"{safe_mean([float(r['faithfulness']) for r in scored]):.6f}",
                    "response_groundedness_mean": f"{safe_mean([float(r['response_groundedness']) for r in scored]):.6f}",
                    "unknown_precision": f"{unknown_metrics['precision']:.6f}",
                    "unknown_recall": f"{unknown_metrics['recall']:.6f}",
                    "unknown_f1": f"{unknown_metrics['f1']:.6f}",
                }
            )


def write_label_aware_summary_csv(path: Path, rows: List[Dict]) -> None:
    by_condition: Dict[str, List[Dict]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_AWARE_SUMMARY_COLUMNS)
        writer.writeheader()

        for condition in sorted(by_condition.keys()):
            group = by_condition[condition]
            scored = non_error_rows(group)
            labeled = [r for r in scored if r.get("label_true") is not None]
            true_rows = [r for r in labeled if bool(r["label_true"])]
            false_rows = [r for r in labeled if not bool(r["label_true"])]
            unknown_metrics = unknown_detection_metrics(scored)
            category_macro = category_macro_metrics(scored)

            writer.writerow(
                {
                    "condition": condition,
                    "labeled_samples": len(labeled),
                    "label_true_samples": len(true_rows),
                    "label_false_samples": len(false_rows),
                    "exact_match_true_mean": f"{safe_mean([float(r['exact_match']) for r in true_rows]):.6f}",
                    "exact_match_false_mean": f"{safe_mean([float(r['exact_match']) for r in false_rows]):.6f}",
                    "precision_true_mean": f"{safe_mean([float(r['precision']) for r in true_rows]):.6f}",
                    "precision_false_mean": f"{safe_mean([float(r['precision']) for r in false_rows]):.6f}",
                    "recall_true_mean": f"{safe_mean([float(r['recall']) for r in true_rows]):.6f}",
                    "recall_false_mean": f"{safe_mean([float(r['recall']) for r in false_rows]):.6f}",
                    "f1_true_mean": f"{safe_mean([float(r['f1']) for r in true_rows]):.6f}",
                    "f1_false_mean": f"{safe_mean([float(r['f1']) for r in false_rows]):.6f}",
                    "bertscore_precision_true_mean": f"{safe_mean([float(r['bertscore_precision']) for r in true_rows]):.6f}",
                    "bertscore_precision_false_mean": f"{safe_mean([float(r['bertscore_precision']) for r in false_rows]):.6f}",
                    "bertscore_recall_true_mean": f"{safe_mean([float(r['bertscore_recall']) for r in true_rows]):.6f}",
                    "bertscore_recall_false_mean": f"{safe_mean([float(r['bertscore_recall']) for r in false_rows]):.6f}",
                    "bertscore_f1_true_mean": f"{safe_mean([float(r['bertscore_f1']) for r in true_rows]):.6f}",
                    "bertscore_f1_false_mean": f"{safe_mean([float(r['bertscore_f1']) for r in false_rows]):.6f}",
                    "support_recall_at5_true_mean": f"{safe_mean([float(r['support_recall_at5']) for r in true_rows]):.6f}",
                    "bridge_coverage_at5_true_mean": f"{safe_mean([float(r['bridge_coverage_at5']) for r in true_rows]):.6f}",
                    "mrr_at5_true_mean": f"{safe_mean([float(r['mrr_at5']) for r in true_rows]):.6f}",
                    "ndcg_at5_true_mean": f"{safe_mean([float(r['ndcg_at5']) for r in true_rows]):.6f}",
                    "faithfulness_true_mean": f"{safe_mean([float(r['faithfulness']) for r in true_rows]):.6f}",
                    "response_groundedness_true_mean": f"{safe_mean([float(r['response_groundedness']) for r in true_rows]):.6f}",
                    "unknown_precision": f"{unknown_metrics['precision']:.6f}",
                    "unknown_recall": f"{unknown_metrics['recall']:.6f}",
                    "unknown_f1": f"{unknown_metrics['f1']:.6f}",
                    "category_macro_exact_match_mean": f"{category_macro['exact_match']:.6f}",
                    "category_macro_precision_mean": f"{category_macro['precision']:.6f}",
                    "category_macro_recall_mean": f"{category_macro['recall']:.6f}",
                    "category_macro_f1_mean": f"{category_macro['f1']:.6f}",
                    "category_macro_bertscore_precision_mean": f"{category_macro['bertscore_precision']:.6f}",
                    "category_macro_bertscore_recall_mean": f"{category_macro['bertscore_recall']:.6f}",
                    "category_macro_bertscore_f1_mean": f"{category_macro['bertscore_f1']:.6f}",
                    "category_macro_unknown_f1": f"{category_macro['unknown_f1']:.6f}",
                }
            )


def write_schema_files(output_dir: Path) -> None:
    summary_schema = output_dir / "summary_by_condition.schema.csv"
    per_question_schema = output_dir / "per_question_metrics.schema.csv"
    category_schema = output_dir / "summary_by_condition_category.schema.csv"
    label_aware_schema = output_dir / "summary_label_aware_by_condition.schema.csv"

    summary_schema.write_text(",".join(SUMMARY_COLUMNS) + "\n", encoding="utf-8")
    per_question_schema.write_text(",".join(PER_QUESTION_COLUMNS) + "\n", encoding="utf-8")
    category_schema.write_text(",".join(CATEGORY_SUMMARY_COLUMNS) + "\n", encoding="utf-8")
    label_aware_schema.write_text(",".join(LABEL_AWARE_SUMMARY_COLUMNS) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-condition experiment outputs into CSV files.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Experiment output directory containing per_question/*.jsonl",
    )
    parser.add_argument(
        "--summary-csv",
        default="summary_by_condition.csv",
        help="Summary CSV filename (under input-dir unless absolute path)",
    )
    parser.add_argument(
        "--summary-by-category-csv",
        default="summary_by_condition_category.csv",
        help="Category summary CSV filename (under input-dir unless absolute path)",
    )
    parser.add_argument(
        "--label-aware-csv",
        default="summary_label_aware_by_condition.csv",
        help="Label-aware summary CSV filename (under input-dir unless absolute path)",
    )
    parser.add_argument(
        "--per-question-csv",
        default="per_question_metrics.csv",
        help="Per-question CSV filename (under input-dir unless absolute path)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    per_question_dir = input_dir / "per_question"
    if not per_question_dir.exists():
        raise FileNotFoundError(f"Missing per-question directory: {per_question_dir}")

    rows = read_rows(per_question_dir)
    if not rows:
        raise RuntimeError(f"No rows found under {per_question_dir}")

    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_absolute():
        summary_csv = input_dir / summary_csv

    summary_by_category_csv = Path(args.summary_by_category_csv)
    if not summary_by_category_csv.is_absolute():
        summary_by_category_csv = input_dir / summary_by_category_csv

    label_aware_csv = Path(args.label_aware_csv)
    if not label_aware_csv.is_absolute():
        label_aware_csv = input_dir / label_aware_csv

    per_question_csv = Path(args.per_question_csv)
    if not per_question_csv.is_absolute():
        per_question_csv = input_dir / per_question_csv

    write_per_question_csv(per_question_csv, rows)
    write_summary_csv(summary_csv, rows)
    write_category_summary_csv(summary_by_category_csv, rows)
    write_label_aware_summary_csv(label_aware_csv, rows)
    write_schema_files(input_dir)

    print(f"Wrote: {per_question_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_by_category_csv}")
    print(f"Wrote: {label_aware_csv}")
    print(f"Wrote: {input_dir / 'summary_by_condition.schema.csv'}")
    print(f"Wrote: {input_dir / 'per_question_metrics.schema.csv'}")
    print(f"Wrote: {input_dir / 'summary_by_condition_category.schema.csv'}")
    print(f"Wrote: {input_dir / 'summary_label_aware_by_condition.schema.csv'}")


if __name__ == "__main__":
    main()
