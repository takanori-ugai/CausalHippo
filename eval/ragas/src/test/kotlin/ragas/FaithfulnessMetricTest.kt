package ragas

import kotlinx.coroutines.runBlocking
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.metrics.defaults.FaithfulnessMetric
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FaithfulnessMetricTest {
    @Test
    fun requiredColumnsDependOnHeuristicFallbackMode() {
        assertEquals(
            setOf("response", "retrieved_contexts"),
            FaithfulnessMetric(allowHeuristicFallback = true).requiredColumns[ragas.metrics.MetricType.SINGLE_TURN],
        )
        assertEquals(
            setOf("user_input", "response", "retrieved_contexts"),
            FaithfulnessMetric().requiredColumns[ragas.metrics.MetricType.SINGLE_TURN],
        )
    }

    @Test
    fun crossContextSplicedSentenceIsNotMarkedSupported() =
        runBlocking {
            val metric = FaithfulnessMetric(allowHeuristicFallback = true)
            val sample =
                SingleTurnSample(
                    response = "Kotlin runs on JVM and blue whales migrate oceans.",
                    retrievedContexts =
                        listOf(
                            "Kotlin runs on JVM.",
                            "Blue whales migrate oceans.",
                        ),
                )

            assertEquals(0.0, metric.singleTurnAscore(sample))
        }

    @Test
    fun sentenceSupportedBySingleContextScoresAsSupported() =
        runBlocking {
            val metric = FaithfulnessMetric(allowHeuristicFallback = true)
            val sample =
                SingleTurnSample(
                    response = "Kotlin runs on JVM.",
                    retrievedContexts =
                        listOf(
                            "Kotlin runs on JVM with strong tooling.",
                            "Python uses indentation.",
                        ),
                )

            assertEquals(1.0, metric.singleTurnAscore(sample))
        }

    @Test
    fun defaultPathRequiresLlmForParitySemantics() {
        runBlocking {
            val metric = FaithfulnessMetric()
            val sample =
                SingleTurnSample(
                    userInput = "What is Kotlin?",
                    response = "Kotlin runs on JVM.",
                    retrievedContexts = listOf("Kotlin runs on JVM."),
                )
            val error = assertFailsWith<IllegalStateException> { metric.singleTurnAscore(sample) }
            assertEquals(
                "FaithfulnessMetric requires an LLM for parity semantics. " +
                    "Set llm on the metric or use FaithfulnessMetric(allowHeuristicFallback = true).",
                error.message,
            )
        }
    }

    @Test
    fun llmPathUsesStatementAndVerdictPipeline() =
        runBlocking {
            val metric =
                FaithfulnessMetric().also { faithfulness ->
                    faithfulness.llm =
                        ScriptedLlm(
                            outputs =
                                listOf(
                                    """{"statements":["Kotlin runs on JVM.","Kotlin was created in 2010."]}""",
                                    """{"statements":[""" +
                                        """{"statement":"Kotlin runs on JVM.","reason":"In context","verdict":1},""" +
                                        """{"statement":"Kotlin was created in 2010.","reason":"Not in context","verdict":0}]}""",
                                ),
                        )
                }
            val sample =
                SingleTurnSample(
                    userInput = "What is Kotlin?",
                    response = "Kotlin runs on JVM. Kotlin was created in 2010.",
                    retrievedContexts = listOf("Kotlin runs on JVM and Android."),
                )

            assertEquals(0.5, metric.singleTurnAscore(sample))
        }

    @Test
    fun llmVerdictPromptEscapesQuotedStatementsAsJson() =
        runBlocking {
            val llm =
                CapturingScriptedLlm(
                    outputs =
                        listOf(
                            """{"statements":["Kotlin is called \"JetBrains language\"."]}""",
                            """{"statements":[""" +
                                """{"statement":"Kotlin is called \"JetBrains language\".","reason":"In context","verdict":1}]}""",
                        ),
                )
            val metric = FaithfulnessMetric().also { it.llm = llm }
            val sample =
                SingleTurnSample(
                    userInput = "What is Kotlin called?",
                    response = """Kotlin is called "JetBrains language".""",
                    retrievedContexts = listOf("""Kotlin is called "JetBrains language" by some developers."""),
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(1.0, score, 1e-9)
            assertTrue(llm.prompts.size >= 2)
            assertTrue(llm.prompts[1].contains("""["Kotlin is called \"JetBrains language\"."]"""))
        }

    @Test
    fun llmPathReturnsNanWhenAnyVerdictIsMissing() =
        runBlocking {
            val metric =
                FaithfulnessMetric().also { faithfulness ->
                    faithfulness.llm =
                        ScriptedLlm(
                            outputs =
                                listOf(
                                    """{"statements":["S1","S2"]}""",
                                    """{"statements":[""" +
                                        """{"statement":"S1","reason":"ok","verdict":1},""" +
                                        """{"statement":"S2","reason":"missing"}]}""",
                                ),
                        )
                }
            val sample =
                SingleTurnSample(
                    userInput = "Q",
                    response = "R",
                    retrievedContexts = listOf("C"),
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertTrue(score.isNaN())
        }

    @Test
    fun llmPathReturnsNanWhenAnyVerdictIsNonBinary() =
        runBlocking {
            val metric =
                FaithfulnessMetric().also { faithfulness ->
                    faithfulness.llm =
                        ScriptedLlm(
                            outputs =
                                listOf(
                                    """{"statements":["S1","S2"]}""",
                                    """{"statements":[""" +
                                        """{"statement":"S1","reason":"ok","verdict":1},""" +
                                        """{"statement":"S2","reason":"bad","verdict":2}]}""",
                                ),
                        )
                }
            val sample =
                SingleTurnSample(
                    userInput = "Q",
                    response = "R",
                    retrievedContexts = listOf("C"),
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertTrue(score.isNaN())
        }
}

private class ScriptedLlm(
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
        check(cursor < outputs.size) {
            "Unexpected generateText call #${cursor + 1}; scripted outputs exhausted."
        }
        val value = outputs[cursor]
        cursor += 1
        return LlmResult(generations = listOf(LlmGeneration(value)))
    }
}

private class CapturingScriptedLlm(
    private val outputs: List<String>,
) : BaseRagasLlm {
    private var cursor: Int = 0
    val prompts = mutableListOf<String>()
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        prompts += prompt
        check(cursor < outputs.size) {
            "Unexpected generateText call #${cursor + 1}; scripted outputs exhausted."
        }
        val value = outputs[cursor]
        cursor += 1
        return LlmResult(generations = listOf(LlmGeneration(value)))
    }
}
