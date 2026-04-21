# Metrics

## Default single-turn metrics

`ragas.defaultMetrics()` returns:

- `answer_relevancy`
- `context_precision`
- `faithfulness`
- `context_recall`

## Tiered collection helpers

Top-level helpers in package `ragas`:

- `tier1Metrics()`
- `tier2Metrics()`
- `tier3Metrics()`
- `tier4Metrics()`

## Metric types

- `SingleTurnMetric`
- `MultiTurnMetric`
- `MetricWithLlm`
- `MetricWithEmbeddings`

Metrics declare required columns and output type, and are validated against dataset features before execution.
