package ragas

import kotlinx.coroutines.runBlocking
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.metrics.collections.NoiseSensitivityMetric
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class NoiseSensitivityMetricParityTest {
    @Test
    fun llmPathMatchesRelevantAndIrrelevantPythonSemantics() =
        runBlocking {
            val outputs =
                listOf(
                    """{"statements":["gt"]}""",
                    """{"statements":["ans"]}""",
                    """{"statements":[{"statement":"gt","reason":"ok","verdict":1}]}""",
                    """{"statements":[{"statement":"gt","reason":"bad","verdict":0}]}""",
                    """{"statements":[{"statement":"ans","reason":"ok","verdict":1}]}""",
                    """{"statements":[{"statement":"ans","reason":"ok","verdict":1}]}""",
                    """{"statements":[{"statement":"ans","reason":"bad","verdict":0}]}""",
                )
            val sample =
                SingleTurnSample(
                    userInput = "Q",
                    response = "A",
                    reference = "R",
                    retrievedContexts = listOf("ctx1", "ctx2"),
                )

            val relevant =
                NoiseSensitivityMetric(mode = NoiseSensitivityMetric.Mode.RELEVANT).also { metric ->
                    metric.llm = ScriptedNoiseSensitivityLlm(outputs = outputs)
                }
            val irrelevant =
                NoiseSensitivityMetric(mode = NoiseSensitivityMetric.Mode.IRRELEVANT).also { metric ->
                    metric.llm = ScriptedNoiseSensitivityLlm(outputs = outputs)
                }

            assertEquals(1.0, (relevant.singleTurnAscore(sample) as Number).toDouble(), 1e-9)
            assertEquals(0.0, (irrelevant.singleTurnAscore(sample) as Number).toDouble(), 1e-9)
        }

    @Test
    fun llmPathReturnsNanWhenNoAnswerStatementsAreProduced() =
        runBlocking {
            val metric =
                NoiseSensitivityMetric().also { noise ->
                    noise.llm =
                        ScriptedNoiseSensitivityLlm(
                            outputs =
                                listOf(
                                    """{"statements":["gt"]}""",
                                    """{"statements":[]}""",
                                ),
                        )
                }
            val sample =
                SingleTurnSample(
                    userInput = "Q",
                    response = "A",
                    reference = "R",
                    retrievedContexts = listOf("ctx"),
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertTrue(score.isNaN())
        }

    @Test
    fun llmPathValidatesRequiredInputs() =
        runBlocking {
            val metric = NoiseSensitivityMetric().also { it.llm = ScriptedNoiseSensitivityLlm(outputs = emptyList()) }

            val missingReference =
                assertFailsWith<IllegalArgumentException> {
                    metric.singleTurnAscore(
                        SingleTurnSample(
                            userInput = "Q",
                            response = "A",
                            reference = "",
                            retrievedContexts = listOf("ctx"),
                        ),
                    )
                }
            assertEquals("reference is missing. Please add reference to the test sample.", missingReference.message)
        }
}

private class ScriptedNoiseSensitivityLlm(
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
