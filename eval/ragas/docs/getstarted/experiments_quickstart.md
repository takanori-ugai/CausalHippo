# Experiment Workflow (Kotlin)

Python-style `@experiment` decorator workflows are not yet ported as a first-class Kotlin API.

You can still run repeatable experiments by combining:

- your own row iteration loop,
- backend persistence (`inmemory`, `local/csv`, `local/jsonl`), and
- `evaluate(...)`.

## Minimal pattern

```kotlin
import ragas.evaluate
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample

fun runExperiment(name: String, questions: List<String>): Unit {
    val samples =
        questions.map { q ->
            // Replace with your app pipeline
            val response = "stub response for $q"
            val contexts = listOf("stub context for $q")
            SingleTurnSample(
                userInput = q,
                response = response,
                retrievedContexts = contexts,
                referenceContexts = contexts,
                reference = contexts.first()
            )
        }

    val result = evaluate(EvaluationDataset(samples))
    println("Experiment: $name")
    println(result.scores)
}
```

## Persist datasets/experiments

Use backends from `ragas.backends`:

- `InMemoryBackend`
- `LocalCsvBackend`
- `LocalJsonlBackend`

For current API surface and status, see:

- [API_SURFACE.md](/home/ugai/ragas/kotlin/API_SURFACE.md)
- [PARITY_MATRIX.md](/home/ugai/ragas/kotlin/PARITY_MATRIX.md)
