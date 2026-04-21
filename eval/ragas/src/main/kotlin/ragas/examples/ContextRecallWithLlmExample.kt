package ragas.examples

import dev.langchain4j.model.openai.OpenAiChatModel
import ragas.evaluate
import ragas.llms.LangChain4jLlm
import ragas.metrics.defaults.ContextRecallMetric
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig

/**
 * Example: run ContextRecallMetric with an LLM-backed judge.
 *
 * Required environment:
 * - OPENAI_API_KEY
 *
 * Run:
 *   ./gradlew run --args="" -PmainClass=ragas.examples.ContextRecallWithLlmExampleKt
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

    val metric = ContextRecallMetric()
    val dataset =
        EvaluationDataset(
            listOf(
                SingleTurnSample(
                    userInput = "When was Kotlin announced and by whom?",
                    retrievedContexts =
                        listOf(
                            "Kotlin was created by JetBrains and publicly announced in 2011.",
                            "Kotlin became an officially supported Android language in 2017.",
                        ),
                    reference = "Kotlin was announced in 2011 by JetBrains.",
                ),
            ),
        )

    val result =
        evaluate(
            dataset = dataset,
            metrics = listOf(metric),
            llm = ragasLlm,
        )

    val score = result.scores.firstOrNull()?.get(metric.name)
    println("Model: $modelName")
    println("${metric.name} = $score")
}
