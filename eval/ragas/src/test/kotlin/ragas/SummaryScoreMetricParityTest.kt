package ragas

import kotlinx.coroutines.runBlocking
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.metrics.collections.SummaryScoreMetric
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class SummaryScoreMetricParityTest {
    @Test
    fun llmPathComputesQaAndConcisenessLikePython() =
        runBlocking {
            val metric =
                SummaryScoreMetric(lengthPenalty = true, coeff = 0.5).also { summary ->
                    summary.llm =
                        ScriptedSummaryLlm(
                            outputs =
                                listOf(
                                    """{"keyphrases":["Apple","2023"]}""",
                                    """{"questions":["Is Apple mentioned?","Is 2023 mentioned?"]}""",
                                    """{"answers":["1","0"]}""",
                                ),
                        )
                }
            val sample =
                SingleTurnSample(
                    referenceContexts =
                        listOf(
                            "Apple released products in 2023 with significant market attention.",
                        ),
                    response = "Apple released products recently.",
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            val qa = 0.5
            val text = sample.referenceContexts!!.joinToString("\n")
            val conciseness = 1.0 - (minOf(sample.response!!.length, text.length).toDouble() / (text.length.toDouble() + 1e-10))
            val expected = qa * 0.5 + conciseness * 0.5
            assertEquals(expected, score, 1e-9)
        }

    @Test
    fun llmPathValidatesRequiredInputsLikePython() =
        runBlocking {
            val metric = SummaryScoreMetric().also { it.llm = ScriptedSummaryLlm(outputs = emptyList()) }

            val missingContexts =
                assertFailsWith<IllegalArgumentException> {
                    metric.singleTurnAscore(
                        SingleTurnSample(
                            referenceContexts = emptyList(),
                            response = "summary",
                        ),
                    )
                }
            assertEquals("reference_contexts cannot be empty or contain only whitespace", missingContexts.message)

            val missingResponse =
                assertFailsWith<IllegalArgumentException> {
                    metric.singleTurnAscore(
                        SingleTurnSample(
                            referenceContexts = listOf("context"),
                            response = " ",
                        ),
                    )
                }
            assertEquals("response cannot be empty or whitespace only", missingResponse.message)
        }

    @Test
    fun llmPathUsesQaOnlyWhenLengthPenaltyDisabled() =
        runBlocking {
            val metric =
                SummaryScoreMetric(lengthPenalty = false).also { summary ->
                    summary.llm =
                        ScriptedSummaryLlm(
                            outputs =
                                listOf(
                                    """{"keyphrases":["A"]}""",
                                    """{"questions":["Q1","Q2","Q3"]}""",
                                    """{"answers":["1","1","0"]}""",
                                ),
                        )
                }
            val sample =
                SingleTurnSample(
                    referenceContexts = listOf("A context"),
                    response = "A summary",
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(2.0 / 3.0, score, 1e-9)
        }
}

private class ScriptedSummaryLlm(
    private val outputs: List<String>,
) : BaseRagasLlm {
    private var cursor = 0
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        val value = outputs.getOrElse(cursor) { outputs.lastOrNull().orEmpty() }
        cursor += 1
        return LlmResult(generations = listOf(LlmGeneration(value)))
    }
}
