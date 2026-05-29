#!/usr/bin/env python3
"""Compare legacy vs unified evaluator per-question outputs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


NUMERIC_COLUMNS = [
    "exactMatch",
    "precision",
    "recall",
    "f1",
    "bertScorePrecision",
    "bertScoreRecall",
    "bertScoreF1",
    "supportRecallAt1",
    "supportRecallAt3",
    "supportRecallAt5",
    "bridgeCoverageAt5",
    "mrrAt5",
    "ndcgAt5",
    "faithfulness",
    "responseGroundedness",
    "indexLatencyMs",
    "queryLatencyMs",
    "totalLatencyMs",
]


@dataclass(frozen=True)
class SampleKey:
    condition: str
    sample_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", required=True, help="Legacy output dir or its per_question dir")
    parser.add_argument("--unified-dir", required=True, help="Unified output dir or its per_question dir")
    parser.add_argument(
        "--conditions",
        default="all",
        help="all or comma-separated condition IDs to compare (default: all)",
    )
    parser.add_argument(
        "--numeric-columns",
        default=",".join(NUMERIC_COLUMNS),
        help="Comma-separated numeric columns to compare",
    )
    parser.add_argument(
        "--max-mean-abs-diff",
        type=float,
        default=None,
        help="Fail if any condition/column mean absolute diff exceeds this value",
    )
    parser.add_argument(
        "--max-max-abs-diff",
        type=float,
        default=None,
        help="Fail if any condition/column max absolute diff exceeds this value",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=None,
        help="Fail if missing sample ratio exceeds this value for either side",
    )
    parser.add_argument(
        "--min-prediction-match",
        type=float,
        default=None,
        help="Fail if exact prediction match ratio is below this value",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write machine-readable summary JSON",
    )
    return parser.parse_args()


def resolve_per_question_dir(raw: str) -> Path:
    base = Path(raw)
    if (base / "per_question").is_dir():
        return base / "per_question"
    return base


def parse_conditions(raw: str) -> set[str] | None:
    if raw.strip().lower() == "all":
        return None
    parts = {x.strip() for x in raw.split(",") if x.strip()}
    if not parts:
        return None
    return parts


def to_float(value: object) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else 0.0
    except (TypeError, ValueError):
        return 0.0


def canonical_sample_id(row: Dict[str, object]) -> str:
    raw = row.get("sampleId")
    if raw is None:
        raw = row.get("sample_id")
    return str(raw or "")


def canonical_condition(row: Dict[str, object]) -> str:
    return str(row.get("condition") or "")


def read_jsonl_rows(per_question_dir: Path) -> Dict[SampleKey, Dict[str, object]]:
    rows: Dict[SampleKey, Dict[str, object]] = {}
    for path in sorted(per_question_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                condition = canonical_condition(row)
                sample_id = canonical_sample_id(row)
                if not condition or not sample_id:
                    continue
                rows[SampleKey(condition=condition, sample_id=sample_id)] = row
    return rows


def group_by_condition(rows: Dict[SampleKey, Dict[str, object]]) -> Dict[str, Dict[SampleKey, Dict[str, object]]]:
    grouped: Dict[str, Dict[SampleKey, Dict[str, object]]] = {}
    for key, row in rows.items():
        grouped.setdefault(key.condition, {})[key] = row
    return grouped


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def compare_condition(
    legacy_rows: Dict[SampleKey, Dict[str, object]],
    unified_rows: Dict[SampleKey, Dict[str, object]],
    numeric_columns: List[str],
) -> Dict[str, object]:
    legacy_keys = set(legacy_rows)
    unified_keys = set(unified_rows)
    common_keys = sorted(legacy_keys & unified_keys, key=lambda k: k.sample_id)

    missing_in_unified = sorted((legacy_keys - unified_keys), key=lambda k: k.sample_id)
    missing_in_legacy = sorted((unified_keys - legacy_keys), key=lambda k: k.sample_id)

    numeric_summary: Dict[str, Dict[str, float]] = {}
    for col in numeric_columns:
        diffs: List[float] = []
        for key in common_keys:
            left = to_float(legacy_rows[key].get(col))
            right = to_float(unified_rows[key].get(col))
            diffs.append(abs(left - right))
        numeric_summary[col] = {
            "meanAbsDiff": mean(diffs),
            "maxAbsDiff": max(diffs) if diffs else 0.0,
        }

    prediction_matches = 0
    for key in common_keys:
        left = str(legacy_rows[key].get("prediction", "")).strip()
        right = str(unified_rows[key].get("prediction", "")).strip()
        if left == right:
            prediction_matches += 1

    common_count = len(common_keys)
    legacy_count = len(legacy_keys)
    unified_count = len(unified_keys)

    return {
        "samples": {
            "legacy": legacy_count,
            "unified": unified_count,
            "common": common_count,
            "missingInUnified": len(missing_in_unified),
            "missingInLegacy": len(missing_in_legacy),
            "missingInUnifiedIds": [k.sample_id for k in missing_in_unified],
            "missingInLegacyIds": [k.sample_id for k in missing_in_legacy],
            "missingInUnifiedRatio": (len(missing_in_unified) / legacy_count) if legacy_count > 0 else 0.0,
            "missingInLegacyRatio": (len(missing_in_legacy) / unified_count) if unified_count > 0 else 0.0,
        },
        "predictions": {
            "exactMatchCount": prediction_matches,
            "exactMatchRatio": (prediction_matches / common_count) if common_count > 0 else 1.0,
        },
        "numeric": numeric_summary,
    }


def print_summary(summary: Dict[str, object]) -> None:
    conditions: Dict[str, Dict[str, object]] = summary["conditions"]  # type: ignore[assignment]
    print("Unified parity summary")
    print(f"Legacy dir : {summary['legacyPerQuestionDir']}")
    print(f"Unified dir: {summary['unifiedPerQuestionDir']}")
    for condition in sorted(conditions):
        info = conditions[condition]
        samples = info["samples"]
        predictions = info["predictions"]
        numeric = info["numeric"]
        top_numeric = sorted(numeric.items(), key=lambda kv: kv[1]["maxAbsDiff"], reverse=True)[:3]
        print(
            f"- {condition}: common={samples['common']}, "
            f"missing(unified/legacy)={samples['missingInUnified']}/{samples['missingInLegacy']}, "
            f"prediction_match={predictions['exactMatchRatio']:.4f}"
        )
        for column, col_stats in top_numeric:
            print(
                f"    {column}: mean_abs_diff={col_stats['meanAbsDiff']:.6f}, "
                f"max_abs_diff={col_stats['maxAbsDiff']:.6f}"
            )


def collect_failures(
    summary: Dict[str, object],
    max_mean_abs_diff: float | None,
    max_max_abs_diff: float | None,
    max_missing_ratio: float | None,
    min_prediction_match: float | None,
) -> List[str]:
    failures: List[str] = []
    conditions: Dict[str, Dict[str, object]] = summary["conditions"]  # type: ignore[assignment]
    for condition, info in conditions.items():
        samples = info["samples"]
        predictions = info["predictions"]
        numeric: Dict[str, Dict[str, float]] = info["numeric"]

        if max_missing_ratio is not None:
            if samples["missingInUnifiedRatio"] > max_missing_ratio:
                failures.append(
                    f"{condition}: missingInUnifiedRatio={samples['missingInUnifiedRatio']:.6f} > {max_missing_ratio:.6f}"
                )
            if samples["missingInLegacyRatio"] > max_missing_ratio:
                failures.append(
                    f"{condition}: missingInLegacyRatio={samples['missingInLegacyRatio']:.6f} > {max_missing_ratio:.6f}"
                )

        if min_prediction_match is not None and predictions["exactMatchRatio"] < min_prediction_match:
            failures.append(
                f"{condition}: prediction exactMatchRatio={predictions['exactMatchRatio']:.6f} < {min_prediction_match:.6f}"
            )

        for column, stats in numeric.items():
            if max_mean_abs_diff is not None and stats["meanAbsDiff"] > max_mean_abs_diff:
                failures.append(
                    f"{condition}.{column}: meanAbsDiff={stats['meanAbsDiff']:.6f} > {max_mean_abs_diff:.6f}"
                )
            if max_max_abs_diff is not None and stats["maxAbsDiff"] > max_max_abs_diff:
                failures.append(
                    f"{condition}.{column}: maxAbsDiff={stats['maxAbsDiff']:.6f} > {max_max_abs_diff:.6f}"
                )
    return failures


def main() -> int:
    args = parse_args()
    legacy_dir = resolve_per_question_dir(args.legacy_dir)
    unified_dir = resolve_per_question_dir(args.unified_dir)
    include_conditions = parse_conditions(args.conditions)
    numeric_columns = [x.strip() for x in args.numeric_columns.split(",") if x.strip()]

    legacy_rows = read_jsonl_rows(legacy_dir)
    unified_rows = read_jsonl_rows(unified_dir)

    legacy_by_condition = group_by_condition(legacy_rows)
    unified_by_condition = group_by_condition(unified_rows)
    all_conditions = sorted(set(legacy_by_condition) | set(unified_by_condition))
    if include_conditions is not None:
        all_conditions = [c for c in all_conditions if c in include_conditions]

    conditions_summary: Dict[str, Dict[str, object]] = {}
    for condition in all_conditions:
        conditions_summary[condition] = compare_condition(
            legacy_rows=legacy_by_condition.get(condition, {}),
            unified_rows=unified_by_condition.get(condition, {}),
            numeric_columns=numeric_columns,
        )

    summary: Dict[str, object] = {
        "legacyPerQuestionDir": str(legacy_dir),
        "unifiedPerQuestionDir": str(unified_dir),
        "conditions": conditions_summary,
    }

    print_summary(summary)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failures = collect_failures(
        summary=summary,
        max_mean_abs_diff=args.max_mean_abs_diff,
        max_max_abs_diff=args.max_max_abs_diff,
        max_missing_ratio=args.max_missing_ratio,
        min_prediction_match=args.min_prediction_match,
    )
    if failures:
        print("\nParity threshold check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
