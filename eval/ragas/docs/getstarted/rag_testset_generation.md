# Testset Generation for RAG (Kotlin)

The Kotlin port currently provides a scaffold generator API.

## 1) Provide source documents

```kotlin
val docs = listOf(
    "Kotlin is a modern language for JVM and Android.",
    "Ragas provides LLM and RAG evaluation metrics."
)
```

## 2) Generate testset

```kotlin
import ragas.testset.synthesizers.TestsetGenerator

val generator = TestsetGenerator()
val testset = generator.generateFromDocuments(
    documents = docs,
    testsetSize = 5,
    transforms = null
)
```

## 3) Convert to evaluation dataset and run eval

```kotlin
import ragas.evaluate

val dataset = testset.toEvaluationDataset()
val result = evaluate(dataset = dataset)
println(result.scores)
```

## Scope note

This is scaffold-level generation, not full parity with the Python synthesizer pipeline yet.
See [PARITY_MATRIX.md](/home/ugai/ragas/kotlin/PARITY_MATRIX.md).
