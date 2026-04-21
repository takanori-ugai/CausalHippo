# aevaluate()

Suspending async evaluation entrypoint in package `ragas`.

```kotlin
suspend fun aevaluate(
    dataset: EvaluationDataset<out Sample>,
    metrics: List<Metric>? = null,
    llm: BaseRagasLlm? = null,
    embeddings: BaseRagasEmbedding? = null,
    runConfig: RunConfig = RunConfig(),
    raiseExceptions: Boolean = false,
    batchSize: Int? = null,
): EvaluationResult
```

Semantics match `evaluate(...)`, without wrapping in `runBlocking`.
