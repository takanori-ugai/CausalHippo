<!-- Adapted for ragas-kotlin on 2026-04-01 -->
> [!NOTE]
> This page was adapted from `../docs/quoted_spans_metric.md` for the Kotlin port (`ragas-kotlin`).
> Python APIs/examples may not map 1:1. Use Kotlin entrypoints in package `ragas` and check [`/home/ugai/ragas/kotlin/PARITY_MATRIX.md`](/home/ugai/ragas/kotlin/PARITY_MATRIX.md) and [`/home/ugai/ragas/kotlin/MIGRATION.md`](/home/ugai/ragas/kotlin/MIGRATION.md).

## `QuotedSpansAlignment`

**What:** A metric that measures the fraction of quoted spans in a model's answer
that appear verbatim in the retrieved sources.  The score is in the range
[0, 1], where 1.0 indicates every quoted span is supported by evidence and 0.0
indicates no quoted spans are found in the sources.

**Why:** Users place extra trust in exact quotes.  When a model quotes facts
that aren't present in its evidence, it undermines reliability.  This metric
helps catch cases of citation drift where quoted phrases in the answer are
unsupported.

## Modern Collections API (Recommended)

```python
from ragas.metrics.collections import QuotedSpansAlignment

metric = QuotedSpansAlignment()

result = await metric.ascore(
    response='The study found that "machine learning improves accuracy".',
    retrieved_contexts=["Machine learning improves accuracy by 15%."]
)
print(f"Score: {result.value}")  # 1.0
print(f"Reason: {result.reason}")  # "Matched 1/1 quoted spans"
```

**Parameters:**

- `name`: The metric name (default: "quoted_spans_alignment")
- `casefold`: Whether to normalize text by lower-casing before matching (default: True)
- `min_span_words`: Minimum number of words in a quoted span (default: 3)

**Input:**

- `response: str` – the model's response containing quoted spans
- `retrieved_contexts: List[str]` – list of source passages to check against

**Output:** A `MetricResult` with:

- `value`: Score in [0, 1]
- `reason`: Description of matched/total spans

**Notes:**

- The implementation normalizes text by collapsing whitespace and lower‑casing.
- Spans shorter than three words are ignored by default; adjust `min_span_words` to change this.
- If no quoted spans are found in the response, the score is 1.0 (nothing to verify).

---

## Legacy API (Deprecated)

> **Warning:** The legacy `quoted_spans_alignment` function is deprecated.
> Please use `QuotedSpansAlignment` from `ragas.metrics.collections` instead.

**Input shape:**

- `answers: List[str]` – list of model answers (length N)
- `sources: List[List[str]]` – list (length N) of lists of source passages

**Output:** A dictionary containing:

```python
{
  "citation_alignment_quoted_spans": float,  # score in [0,1]
  "matched": float,                          # number of spans found in sources
  "total": float                            # total number of spans considered
}
```

**Notes:**

- If no quoted spans are found across all answers, the score is defined as 0.0 with
  `total = 0`.
  