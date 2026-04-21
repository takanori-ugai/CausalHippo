package ragas

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.metrics.collections.AnswerAccuracyMetric
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class AnswerAccuracyMetricParityTest {
    @Test
    fun llmPathUsesDualJudgeAverageAndSwappedSecondJudgeInputs() =
        runBlocking {
            val llm =
                ScriptedAnswerAccuracyLlm(
                    outputs =
                        listOf(
                            """{"rating":4}""",
                            """{"rating":2}""",
                        ),
                )
            val metric = AnswerAccuracyMetric().also { it.llm = llm }
            val sample =
                SingleTurnSample(
                    userInput = "When was Einstein born?",
                    response = "Einstein was born in 1879.",
                    reference = "Albert Einstein was born on March 14, 1879.",
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(0.75, score, 1e-9)
            assertEquals(2, llm.prompts.size)
            assertTrue(llm.prompts[1].contains("""Reference Answer: "Einstein was born in 1879.""""))
            assertTrue(llm.prompts[1].contains("""User Answer: "Albert Einstein was born on March 14, 1879.""""))
        }

    @Test
    fun llmPathRetriesInvalidJudgeRatings() =
        runBlocking {
            val llm =
                ScriptedAnswerAccuracyLlm(
                    outputs =
                        listOf(
                            """{"rating":3}""",
                            """{"rating":2}""",
                            """{"rating":4}""",
                        ),
                )
            val metric = AnswerAccuracyMetric(maxRetries = 2).also { it.llm = llm }
            val sample =
                SingleTurnSample(
                    userInput = "When was Einstein born?",
                    response = "Einstein was born in 1879.",
                    reference = "Albert Einstein was born on March 14, 1879.",
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(0.75, score, 1e-9)
            assertEquals(3, llm.prompts.size)
        }

    @Test
    fun llmPathReturnsNanWhenBothJudgesFailAfterRetries() =
        runBlocking {
            val llm =
                ScriptedAnswerAccuracyLlm(
                    outputs =
                        listOf(
                            """{"rating":3}""",
                            """{"rating":5}""",
                            """{"rating":1}""",
                            """{"rating":-1}""",
                        ),
                )
            val metric = AnswerAccuracyMetric(maxRetries = 2).also { it.llm = llm }
            val sample =
                SingleTurnSample(
                    userInput = "When was Einstein born?",
                    response = "Einstein was born in 1879.",
                    reference = "Albert Einstein was born on March 14, 1879.",
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertTrue(score.isNaN())
        }

    @Test
    fun llmPathReturnsNanWhenOnlyOneJudgeSucceeds() =
        runBlocking {
            val llm =
                ScriptedAnswerAccuracyLlm(
                    outputs =
                        listOf(
                            """{"rating":4}""",
                            """{"rating":3}""",
                            """{"rating":5}""",
                        ),
                )
            val metric = AnswerAccuracyMetric(maxRetries = 2).also { it.llm = llm }
            val sample =
                SingleTurnSample(
                    userInput = "When was Einstein born?",
                    response = "Einstein was born in 1879.",
                    reference = "Albert Einstein was born on March 14, 1879.",
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertTrue(score.isNaN())
            assertEquals(3, llm.prompts.size)
        }

    @Test
    fun llmPathPropagatesCancellationExceptionFromJudgeCalls() {
        runBlocking {
            val metric =
                AnswerAccuracyMetric(maxRetries = 2).also {
                    it.llm =
                        object : BaseRagasLlm {
                            override var runConfig: RunConfig = RunConfig()

                            override suspend fun generateText(
                                prompt: String,
                                n: Int,
                                temperature: Double?,
                                stop: List<String>?,
                            ): LlmResult = throw CancellationException("cancelled")
                        }
                }
            val sample =
                SingleTurnSample(
                    userInput = "When was Einstein born?",
                    response = "Einstein was born in 1879.",
                    reference = "Albert Einstein was born on March 14, 1879.",
                )

            assertFailsWith<CancellationException> { metric.singleTurnAscore(sample) }
        }
    }

    @Test
    fun llmPathValidatesRequiredFieldsLikePythonImplementation() =
        runBlocking {
            val metric =
                AnswerAccuracyMetric().also {
                    it.llm = ScriptedAnswerAccuracyLlm(outputs = listOf("""{"rating":4}"""))
                }
            val sample =
                SingleTurnSample(
                    userInput = "",
                    response = "Einstein was born in 1879.",
                    reference = "Albert Einstein was born on March 14, 1879.",
                )

            val error = assertFailsWith<IllegalArgumentException> { metric.singleTurnAscore(sample) }
            assertEquals(
                "user_input is missing. Please add user_input to the test sample.",
                error.message,
            )
        }
}

private class ScriptedAnswerAccuracyLlm(
    private val outputs: List<String>,
) : BaseRagasLlm {
    private var cursor = 0
    val prompts = mutableListOf<String>()
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        prompts += prompt
        val value =
            checkNotNull(outputs.getOrNull(cursor)) {
                "Unexpected generateText call ${cursor + 1}; only ${outputs.size} scripted responses were provided."
            }
        cursor += 1
        return LlmResult(generations = listOf(LlmGeneration(value)))
    }
}
