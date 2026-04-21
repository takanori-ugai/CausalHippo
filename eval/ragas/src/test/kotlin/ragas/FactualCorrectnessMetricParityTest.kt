package ragas

import kotlinx.coroutines.runBlocking
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.metrics.collections.FactualCorrectnessMetric
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class FactualCorrectnessMetricParityTest {
    @Test
    fun llmPathComputesPrecisionRecallAndF1Modes() =
        runBlocking {
            val sample = SingleTurnSample(response = "R", reference = "G")

            val precisionMetric =
                FactualCorrectnessMetric(mode = FactualCorrectnessMetric.Mode.PRECISION).also { metric ->
                    metric.llm =
                        ScriptedFactualCorrectnessLlm(
                            outputs =
                                listOf(
                                    """{"claims":["c1","c2"]}""",
                                    """{"statements":[""" +
                                        """{"statement":"c1","reason":"ok","verdict":1},""" +
                                        """{"statement":"c2","reason":"bad","verdict":0}]}""",
                                ),
                        )
                }
            assertEquals(0.5, (precisionMetric.singleTurnAscore(sample) as Number).toDouble(), 1e-9)

            val recallMetric =
                FactualCorrectnessMetric(mode = FactualCorrectnessMetric.Mode.RECALL).also { metric ->
                    metric.llm =
                        ScriptedFactualCorrectnessLlm(
                            outputs =
                                listOf(
                                    """{"claims":["c1","c2"]}""",
                                    """{"statements":[""" +
                                        """{"statement":"c1","reason":"ok","verdict":1},""" +
                                        """{"statement":"c2","reason":"bad","verdict":0}]}""",
                                    """{"claims":["g1","g2"]}""",
                                    """{"statements":[""" +
                                        """{"statement":"g1","reason":"bad","verdict":0},""" +
                                        """{"statement":"g2","reason":"ok","verdict":1}]}""",
                                ),
                        )
                }
            assertEquals(0.5, (recallMetric.singleTurnAscore(sample) as Number).toDouble(), 1e-9)

            val f1Metric =
                FactualCorrectnessMetric(mode = FactualCorrectnessMetric.Mode.F1).also { metric ->
                    metric.llm =
                        ScriptedFactualCorrectnessLlm(
                            outputs =
                                listOf(
                                    """{"claims":["c1","c2"]}""",
                                    """{"statements":[""" +
                                        """{"statement":"c1","reason":"ok","verdict":1},""" +
                                        """{"statement":"c2","reason":"bad","verdict":0}]}""",
                                    """{"claims":["g1","g2"]}""",
                                    """{"statements":[""" +
                                        """{"statement":"g1","reason":"bad","verdict":0},""" +
                                        """{"statement":"g2","reason":"ok","verdict":1}]}""",
                                ),
                        )
                }
            assertEquals(0.5, (f1Metric.singleTurnAscore(sample) as Number).toDouble(), 1e-9)
        }

    @Test
    fun llmPathRoundsScoreToTwoDecimalsLikePython() =
        runBlocking {
            val metric =
                FactualCorrectnessMetric(mode = FactualCorrectnessMetric.Mode.PRECISION).also { factual ->
                    factual.llm =
                        ScriptedFactualCorrectnessLlm(
                            outputs =
                                listOf(
                                    """{"claims":["c1","c2","c3"]}""",
                                    """{"statements":[""" +
                                        """{"statement":"c1","reason":"ok","verdict":1},""" +
                                        """{"statement":"c2","reason":"ok","verdict":1},""" +
                                        """{"statement":"c3","reason":"bad","verdict":0}]}""",
                                ),
                        )
                }
            val sample = SingleTurnSample(response = "R", reference = "G")

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(0.67, score, 1e-9)
        }

    @Test
    fun llmPathValidatesMissingRequiredInputs() =
        runBlocking {
            val metric =
                FactualCorrectnessMetric().also { it.llm = ScriptedFactualCorrectnessLlm(outputs = emptyList()) }

            val responseError =
                assertFailsWith<IllegalArgumentException> {
                    metric.singleTurnAscore(SingleTurnSample(response = "", reference = "x"))
                }
            assertEquals("response is missing. Please add response to the test sample.", responseError.message)

            val referenceError =
                assertFailsWith<IllegalArgumentException> {
                    metric.singleTurnAscore(SingleTurnSample(response = "x", reference = ""))
                }
            assertEquals("reference is missing. Please add reference to the test sample.", referenceError.message)
        }
}

private class ScriptedFactualCorrectnessLlm(
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
        val value =
            checkNotNull(outputs.getOrNull(cursor)) {
                "Unexpected generateText call ${cursor + 1}; only ${outputs.size} scripted responses were provided."
            }
        cursor += 1
        return LlmResult(generations = listOf(LlmGeneration(value)))
    }
}
