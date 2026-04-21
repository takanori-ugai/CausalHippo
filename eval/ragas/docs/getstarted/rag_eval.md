# Evaluate a Simple RAG System (Kotlin)

This example demonstrates a basic RAG evaluation loop in Kotlin.

## 1) Build a small retriever + generator stub

```kotlin
class SimpleRag(private val documents: List<String>) {
    fun retrieve(query: String): List<String> {
        // Naive retrieval: top-1 by token overlap (placeholder).
        return documents
            .sortedByDescending { doc ->
                query.lowercase().split(" ").count { token -> token.isNotBlank() && doc.lowercase().contains(token) }
            }
            .take(1)
    }

    fun answer(query: String, contexts: List<String>): String {
        // Replace with your LLM call.
        return "Answer based on: ${contexts.firstOrNull().orEmpty()}"
    }
}
```

## 2) Build evaluated samples

```kotlin
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample

val rag = SimpleRag(
    documents = listOf(
        "Kotlin is a modern JVM language.",
        "Ragas evaluates LLM applications and RAG pipelines."
    )
)

val questions = listOf(
    "What is Kotlin?" to "Kotlin is a modern JVM language.",
    "What does Ragas do?" to "Ragas evaluates LLM applications and RAG pipelines."
)

val dataset = EvaluationDataset(
    questions.map { (question, reference) ->
        val contexts = rag.retrieve(question)
        val response = rag.answer(question, contexts)
        SingleTurnSample(
            userInput = question,
            response = response,
            retrievedContexts = contexts,
            referenceContexts = listOf(reference),
            reference = reference
        )
    }
)
```

## 3) Evaluate

```kotlin
import ragas.evaluate

val result = evaluate(dataset = dataset)
println(result.scores)
println("context_precision mean = ${result.metricMean("context_precision")}")
println("context_recall mean = ${result.metricMean("context_recall")}")
```

## Optional model adapters

If you use LangChain4j models, use:

- `ragas.llms.LangChain4jLlm`
- `ragas.embeddings.LangChain4jEmbedding`
