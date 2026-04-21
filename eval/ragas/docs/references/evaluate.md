# evaluate()

Top-level sync evaluation entrypoint in package `ragas`.

```kotlin
fun evaluate(
    dataset: EvaluationDataset<out Sample>,
    metrics: List<Metric>? = null,
    llm: BaseRagasLlm? = null,
    embeddings: BaseRagasEmbedding? = null,
    runConfig: RunConfig = RunConfig(),
    raiseExceptions: Boolean = false,
    batchSize: Int? = null,
): EvaluationResult
```

## Behavior

- Uses default single-turn metrics when `metrics == null`.
- Validates that required sample columns exist for selected metrics.
- Enforces metric/sample compatibility (`SingleTurnMetric` vs `MultiTurnMetric`).
- Returns row-wise score maps in `EvaluationResult.scores`.
