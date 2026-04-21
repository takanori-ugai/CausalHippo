package ragas.examples

import ragas.evaluate
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample
import ragas.tier1Metrics

/**
 * Minimal runnable example for the ragas-kotlin evaluation flow.
 *
 * Run by targeting `ragas.examples.RagasQuickstartExampleKt` from your IDE
 * or with your preferred Gradle/Java exec task.
 */
fun main() {
    val dataset =
        EvaluationDataset(
            listOf(
                SingleTurnSample(
                    userInput = "What language is Kotlin primarily used with?",
                    retrievedContexts =
                        listOf(
                            "Kotlin is a statically typed programming language for the JVM, Android, and beyond.",
                            "It interoperates with Java and is commonly used in Android development.",
                        ),
                    response = "Kotlin is mainly used with the JVM and Android, with strong Java interoperability.",
                    reference = "Kotlin is a statically typed language for JVM and Android ecosystems.",
                    retrievedContextIds = listOf("ctx-1", "ctx-2"),
                    referenceContextIds = listOf("ctx-1"),
                ),
                SingleTurnSample(
                    userInput = "Who developed Kotlin?",
                    retrievedContexts =
                        listOf(
                            "Kotlin was created by JetBrains and announced in 2011.",
                            "It became an officially supported Android language in 2017.",
                        ),
                    response = "Kotlin was developed by JetBrains.",
                    reference = "JetBrains developed Kotlin.",
                    retrievedContextIds = listOf("ctx-3", "ctx-4"),
                    referenceContextIds = listOf("ctx-3"),
                ),
            ),
        )

    val result = evaluate(dataset = dataset, metrics = tier1Metrics())

    println("Per-row scores:")
    result.scores.forEachIndexed { index, row ->
        println("Row ${index + 1}: $row")
    }

    val metricNames =
        result.scores
            .firstOrNull()
            ?.keys
            .orEmpty()
    println("\nMetric means:")
    metricNames.forEach { metric ->
        val mean = result.metricMean(metric) ?: Double.NaN
        println("$metric = %.4f".format(mean))
    }
}
