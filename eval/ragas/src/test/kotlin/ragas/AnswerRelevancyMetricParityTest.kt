package ragas

import kotlinx.coroutines.runBlocking
import ragas.embeddings.BaseRagasEmbedding
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.metrics.defaults.AnswerRelevancyMetric
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class AnswerRelevancyMetricParityTest {
    @Test
    fun defaultPathRequiresLlmAndEmbeddingsForParitySemantics() {
        runBlocking {
            val metric = AnswerRelevancyMetric()
            val sample =
                SingleTurnSample(
                    userInput = "What is the capital of France?",
                    response = "Paris is the capital of France.",
                )

            val error = assertFailsWith<IllegalStateException> { metric.singleTurnAscore(sample) }
            assertEquals(
                "AnswerRelevancyMetric requires both LLM and embeddings for parity semantics. " +
                    "Set llm+embeddings on the metric or use AnswerRelevancyMetric(allowHeuristicFallback = true).",
                error.message,
            )
        }
    }

    @Test
    fun llmPathUsesGeneratedQuestionsAndEmbeddingSimilarity() =
        runBlocking {
            val metric =
                AnswerRelevancyMetric().also {
                    it.strictness = 2
                    it.llm =
                        ScriptedAnswerRelevancyLlm(
                            outputs =
                                listOf(
                                    """{"question":"What is the capital of France?","noncommittal":0}""",
                                    """{"question":"What is 2 + 2?","noncommittal":0}""",
                                ),
                        )
                    it.embeddings =
                        ScriptedAnswerRelevancyEmbedding(
                            vectors =
                                mapOf(
                                    "What is the capital of France?" to listOf(1f, 0f),
                                    "What is 2 + 2?" to listOf(0f, 1f),
                                ),
                        )
                }
            val sample =
                SingleTurnSample(
                    userInput = "What is the capital of France?",
                    response = "Paris is the capital of France.",
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(0.5, score, 1e-9)
        }

    @Test
    fun llmPathReturnsZeroWhenAllGeneratedQuestionsAreNoncommittal() =
        runBlocking {
            val metric =
                AnswerRelevancyMetric().also {
                    it.strictness = 2
                    it.llm =
                        ScriptedAnswerRelevancyLlm(
                            outputs =
                                listOf(
                                    """{"question":"What is the capital of France?","noncommittal":1}""",
                                    """{"question":"What city is France's capital?","noncommittal":1}""",
                                ),
                        )
                    it.embeddings =
                        ScriptedAnswerRelevancyEmbedding(
                            vectors =
                                mapOf(
                                    "What is the capital of France?" to listOf(1f, 0f),
                                    "What city is France's capital?" to listOf(1f, 0f),
                                ),
                        )
                }
            val sample =
                SingleTurnSample(
                    userInput = "What is the capital of France?",
                    response = "I am not sure.",
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(0.0, score, 1e-9)
        }

    @Test
    fun llmPathValidatesRequiredFieldsLikePythonImplementation() =
        runBlocking {
            val metric =
                AnswerRelevancyMetric().also {
                    it.llm = ScriptedAnswerRelevancyLlm(outputs = listOf("""{"question":"x","noncommittal":0}"""))
                    it.embeddings = ScriptedAnswerRelevancyEmbedding(vectors = emptyMap())
                }
            val sample = SingleTurnSample(userInput = " ", response = "answer")

            val error = assertFailsWith<IllegalArgumentException> { metric.singleTurnAscore(sample) }
            assertEquals("user_input cannot be empty", error.message)
        }
}

private class ScriptedAnswerRelevancyLlm(
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

private class ScriptedAnswerRelevancyEmbedding(
    private val vectors: Map<String, List<Float>>,
) : BaseRagasEmbedding {
    override suspend fun embedText(text: String): List<Float> = vectors[text] ?: listOf(0f, 0f)
}
