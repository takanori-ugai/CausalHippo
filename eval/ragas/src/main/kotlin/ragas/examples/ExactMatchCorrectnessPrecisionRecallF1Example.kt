package ragas.examples

import dev.langchain4j.model.openai.OpenAiChatModel
import ragas.evaluate
import ragas.llms.LangChain4jLlm
import ragas.metrics.collections.AnswerCorrectnessMetric
import ragas.metrics.collections.FactualCorrectnessMetric
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig

/**
 * Example: print ExactMatch, Correctness, Precision, Recall, and F1 for one sample using an LLM.
 *
 * Required environment:
 * - OPENAI_API_KEY
 *
 * Run:
 *   ./gradlew run --args="" -PmainClass=ragas.examples.ExactMatchCorrectnessPrecisionRecallF1ExampleKt
 */
fun main() {
    val apiKey =
        System.getenv("OPENAI_API_KEY")
            ?: error("OPENAI_API_KEY is required to run this example.")
    val modelName = "gpt-5.4-mini"
    val chatModel =
        OpenAiChatModel
            .builder()
            .apiKey(apiKey)
            .modelName(modelName)
            .temperature(0.0)
            .build()
    val ragasLlm =
        LangChain4jLlm(
            model = chatModel,
            runConfig = RunConfig(timeoutSeconds = 90),
        )

    val sample =
        SingleTurnSample(
            userInput = "Who developed Kotlin and when was it announced?",
            response = "Kotlin was developed by JetBrains and announced in 2011.",
            reference = "JetBrains developed Kotlin and announced it in 2011.",
        )

    val exactMatch = exactMatch(sample.response, sample.reference)

    val correctnessMetric = AnswerCorrectnessMetric(name = "correctness", weights = listOf(1.0, 0.0))
    val precisionMetric =
        FactualCorrectnessMetric(
            name = "precision",
            mode = FactualCorrectnessMetric.Mode.PRECISION,
        )
    val recallMetric =
        FactualCorrectnessMetric(
            name = "recall",
            mode = FactualCorrectnessMetric.Mode.RECALL,
        )
    val f1Metric =
        FactualCorrectnessMetric(
            name = "f1",
            mode = FactualCorrectnessMetric.Mode.F1,
        )

    val result =
        evaluate(
            dataset = EvaluationDataset(listOf(sample)),
            metrics = listOf(correctnessMetric, precisionMetric, recallMetric, f1Metric),
            llm = ragasLlm,
        )

    val row = result.scores.firstOrNull().orEmpty()
    val correctness = row["correctness"] as? Number ?: Double.NaN
    val precision = row["precision"] as? Number ?: Double.NaN
    val recall = row["recall"] as? Number ?: Double.NaN
    val f1 = row["f1"] as? Number ?: Double.NaN

    println("ExactMatch: ${if (exactMatch) 1 else 0}")
    println("Correctness: %.4f".format(correctness.toDouble()))
    println("Precision: %.4f".format(precision.toDouble()))
    println("Recall: %.4f".format(recall.toDouble()))
    println("F1: %.4f".format(f1.toDouble()))
    println("Model: $modelName")
}

private fun exactMatch(
    response: String?,
    reference: String?,
): Boolean = normalize(response) == normalize(reference)

private fun normalize(value: String?): String =
    value
        .orEmpty()
        .trim()
        .lowercase()
        .replace(Regex("\\s+"), " ")
