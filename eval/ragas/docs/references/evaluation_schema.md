# Evaluation Schema

## Dataset

```kotlin
data class EvaluationDataset<T : Sample>(
    val samples: List<T>
)
```

- All samples in one dataset must be the same concrete type.
- Supported sample types: `SingleTurnSample`, `MultiTurnSample`.

## Sample models

- `SingleTurnSample` includes fields like `userInput`, `response`, `retrievedContexts`, `referenceContexts`, `reference`, `rubrics`.
- `MultiTurnSample` includes `userInput: List<ConversationMessage>`, with validation for tool-call ordering.

## Result

```kotlin
data class EvaluationResult(
    val scores: List<Map<String, Any?>>,
    val dataset: EvaluationDataset<out Sample>,
    val binaryColumns: Set<String> = emptySet(),
    val traces: Map<String, Any?> = emptyMap(),
)
```

Convenience methods:

- `metricValues(metricName: String): List<Any?>`
- `metricMean(metricName: String): Double?`
