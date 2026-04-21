# Quickstart (Kotlin)

```kotlin
import ragas.evaluate
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample

val dataset = EvaluationDataset(
    listOf(
        SingleTurnSample(
            userInput = "What is Kotlin?",
            response = "Kotlin is a statically typed language for the JVM.",
            retrievedContexts = listOf("Kotlin is a modern language targeting JVM and Android."),
            referenceContexts = listOf("Kotlin is a JVM language."),
            reference = "Kotlin is a JVM language"
        )
    )
)

val result = evaluate(dataset = dataset)
println(result.scores)
println("faithfulness mean = ${result.metricMean("faithfulness")}")
```

By default, `evaluate(...)` uses the single-turn default metric set.
