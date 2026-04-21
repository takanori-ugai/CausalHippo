# Evaluate a Simple LLM Application (Kotlin)

This guide shows the minimal Kotlin workflow:

1. Create an `EvaluationDataset`.
2. Run your app to produce responses.
3. Evaluate with `ragas.evaluate(...)`.

## 1) Define input samples

```kotlin
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample

val samples = listOf(
    SingleTurnSample(
        userInput = "What is Ragas?",
        reference = "Ragas is an evaluation toolkit for LLM applications.",
        referenceContexts = listOf("Ragas helps evaluate LLM systems.")
    ),
    SingleTurnSample(
        userInput = "What is Kotlin used for?",
        reference = "Kotlin is used for JVM, Android, and multiplatform development.",
        referenceContexts = listOf("Kotlin targets JVM, Android, JS, and native.")
    )
)
```

## 2) Fill responses from your app

```kotlin
fun queryApp(question: String): Pair<String, List<String>> {
    // Replace this stub with your real app call.
    val response = "Stub response for: $question"
    val retrieved = listOf("Stub context for: $question")
    return response to retrieved
}

val evaluatedSamples =
    samples.map { sample ->
        val (response, contexts) = queryApp(sample.userInput.orEmpty())
        sample.copy(response = response, retrievedContexts = contexts)
    }

val dataset = EvaluationDataset(evaluatedSamples)
```

## 3) Run evaluation

```kotlin
import ragas.evaluate

val result = evaluate(dataset = dataset)
println(result.scores)
println("answer_relevancy mean = ${result.metricMean("answer_relevancy")}")
println("faithfulness mean = ${result.metricMean("faithfulness")}")
```

## Notes

- If you pass `metrics = null`, default single-turn metrics are used.
- For custom metrics, pass `metrics = listOf(...)`.
- For async flows, use `aevaluate(...)`.
