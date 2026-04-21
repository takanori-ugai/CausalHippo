package ragas.metrics.primitives

import kotlinx.coroutines.runBlocking
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals

class DiscreteMetricRegexTest {
    @Test
    fun matchesLabelsWithPunctuation() =
        runBlocking {
            val llm = FakeLlm(outputs = listOf("Result: A+."))
            val sample = SingleTurnSample(userInput = "u", response = "r")
            val discrete =
                DiscreteMetric(
                    name = "discrete_punctuation",
                    prompt = "Decide: {response}",
                    llm = llm,
                    allowedValues = listOf("A", "A+"),
                )

            assertEquals("A+", discrete.singleTurnAscore(sample), "Should match A+ when both A and A+ are allowed")
        }

    @Test
    fun matchesLabelsWithPunctuationAndWordBoundaries() =
        runBlocking {
            val llm = FakeLlm(outputs = listOf("The answer is yes/no."))
            val sample = SingleTurnSample(userInput = "u", response = "r")
            val discrete =
                DiscreteMetric(
                    name = "discrete_punctuation_2",
                    prompt = "Decide: {response}",
                    llm = llm,
                    allowedValues = listOf("yes", "no", "yes/no"),
                )

            assertEquals("yes/no", discrete.singleTurnAscore(sample))
        }
}

private class FakeLlm(
    private val outputs: List<String>,
) : BaseRagasLlm {
    private var cursor: Int = 0
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        val next = outputs[cursor]
        cursor += 1
        return LlmResult(generations = listOf(LlmGeneration(text = next)))
    }
}
