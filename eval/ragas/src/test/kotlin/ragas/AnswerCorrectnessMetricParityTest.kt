package ragas

import kotlinx.coroutines.runBlocking
import ragas.embeddings.BaseRagasEmbedding
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.metrics.collections.AnswerCorrectnessMetric
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class AnswerCorrectnessMetricParityTest {
    @Test
    fun llmPathUsesStatementPipelineAndWeightedSimilarity() =
        runBlocking {
            val llm =
                ScriptedAnswerCorrectnessLlm(
                    outputs =
                        listOf(
                            """{"statements":["Paris is the capital of France.","Paris has museums."]}""",
                            """{"statements":["Paris is the capital of France.","Paris has monuments."]}""",
                            """{"TP":[""" +
                                """{"statement":"Paris is the capital of France.","reason":"supported"}],""" +
                                """"FP":[{"statement":"Paris has museums.","reason":"not in reference"}],""" +
                                """"FN":[{"statement":"Paris has monuments.","reason":"missing"}]}""",
                        ),
                )
            val embeddings =
                ScriptedAnswerCorrectnessEmbedding(
                    vectors =
                        mapOf(
                            "Paris is the capital of France." to listOf(1f, 0f),
                            "Paris is the capital of France and has museums." to listOf(1f, 0f),
                        ),
                )
            val metric =
                AnswerCorrectnessMetric().also {
                    it.llm = llm
                    it.embeddings = embeddings
                }
            val sample =
                SingleTurnSample(
                    userInput = "What is Paris known for?",
                    response = "Paris is the capital of France and has museums.",
                    reference = "Paris is the capital of France.",
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(0.625, score, 1e-9)
        }

    @Test
    fun llmPathAllowsFactualityOnlyWithoutEmbeddings() =
        runBlocking {
            val llm =
                ScriptedAnswerCorrectnessLlm(
                    outputs =
                        listOf(
                            """{"statements":["A"]}""",
                            """{"statements":["A"]}""",
                            """{"TP":[{"statement":"A","reason":"same"}],"FP":[],"FN":[]}""",
                        ),
                )
            val metric = AnswerCorrectnessMetric(weights = listOf(1.0, 0.0)).also { it.llm = llm }
            val sample = SingleTurnSample(userInput = "Q", response = "R", reference = "G")

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(1.0, score, 1e-9)
        }

    @Test
    fun llmPathRequiresEmbeddingsWhenSimilarityWeightIsEnabled() =
        runBlocking {
            val llm =
                ScriptedAnswerCorrectnessLlm(
                    outputs =
                        listOf(
                            """{"statements":["A"]}""",
                            """{"statements":["A"]}""",
                            """{"TP":[{"statement":"A","reason":"same"}],"FP":[],"FN":[]}""",
                        ),
                )
            val metric = AnswerCorrectnessMetric(weights = listOf(0.75, 0.25)).also { it.llm = llm }
            val sample = SingleTurnSample(userInput = "Q", response = "R", reference = "G")

            val error = assertFailsWith<IllegalArgumentException> { metric.singleTurnAscore(sample) }
            assertEquals(
                "Embeddings are required for semantic similarity scoring. Either provide embeddings " +
                    "or set similarity weight to 0 (weights=[1.0, 0.0]) for pure factuality-only evaluation.",
                error.message,
            )
        }

    @Test
    fun llmPathReturnsNanWhenStatementExtractionFails() =
        runBlocking {
            val llm =
                ScriptedAnswerCorrectnessLlm(
                    outputs =
                        listOf(
                            "not-json",
                            """{"statements":["A"]}""",
                        ),
                )
            val metric = AnswerCorrectnessMetric(weights = listOf(1.0, 0.0)).also { it.llm = llm }
            val sample = SingleTurnSample(userInput = "Q", response = "R", reference = "G")

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertTrue(score.isNaN())
        }

    @Test
    fun llmPathReturnsNanWhenClassificationParsingFails() =
        runBlocking {
            val llm =
                ScriptedAnswerCorrectnessLlm(
                    outputs =
                        listOf(
                            """{"statements":["A"]}""",
                            """{"statements":["A"]}""",
                            "not-json",
                        ),
                )
            val metric = AnswerCorrectnessMetric(weights = listOf(1.0, 0.0)).also { it.llm = llm }
            val sample = SingleTurnSample(userInput = "Q", response = "R", reference = "G")

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertTrue(score.isNaN())
        }

    @Test
    fun llmPathFailsOnEmbeddingDimensionMismatch() =
        runBlocking {
            val llm =
                ScriptedAnswerCorrectnessLlm(
                    outputs =
                        listOf(
                            """{"statements":["A"]}""",
                            """{"statements":["A"]}""",
                            """{"TP":[{"statement":"A","reason":"same"}],"FP":[],"FN":[]}""",
                        ),
                )
            val embeddings =
                ScriptedAnswerCorrectnessEmbedding(
                    vectors =
                        mapOf(
                            "R" to listOf(1f, 0f),
                            "G" to listOf(1f, 0f, 0f),
                        ),
                )
            val metric =
                AnswerCorrectnessMetric(weights = listOf(0.75, 0.25)).also {
                    it.llm = llm
                    it.embeddings = embeddings
                }
            val sample = SingleTurnSample(userInput = "Q", response = "R", reference = "G")

            val error = assertFailsWith<IllegalArgumentException> { metric.singleTurnAscore(sample) }
            assertEquals("Embedding dimension mismatch: left=2, right=3", error.message)
        }
}

private class ScriptedAnswerCorrectnessLlm(
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

private class ScriptedAnswerCorrectnessEmbedding(
    private val vectors: Map<String, List<Float>>,
) : BaseRagasEmbedding {
    override suspend fun embedText(text: String): List<Float> = vectors[text] ?: listOf(0f, 0f)
}
