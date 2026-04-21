package ragas

import kotlinx.coroutines.runBlocking
import ragas.cache.InMemoryCacheBackend
import ragas.embeddings.BaseRagasEmbedding
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample
import ragas.optimizers.DspyOptimizer
import ragas.optimizers.GeneticOptimizer
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PublicApiTest {
    @Test
    fun topLevelEvaluateAndDefaultMetricsWork() {
        val dataset =
            EvaluationDataset(
                listOf(
                    SingleTurnSample(
                        userInput = "What is Kotlin language",
                        response = "Kotlin is language.",
                        retrievedContexts = listOf("Kotlin language on JVM", "Python language"),
                        referenceContexts = listOf("Kotlin language", "JVM runtime"),
                        reference = "Kotlin language",
                    ),
                ),
            )

        val result = evaluate(dataset = dataset)
        assertTrue(result.scores.first().containsKey("answer_relevancy"))
        assertEquals(4, defaultMetrics().size)
        val tier1Names = tier1Metrics().map { metric -> metric.name }.toSet()
        assertEquals(6, tier1Names.size)
        assertTrue("context_relevance" in tier1Names)
        assertTrue("response_groundedness" in tier1Names)
        assertTrue("context_precision_with_reference" in tier1Names)
        assertTrue("context_precision_without_reference" in tier1Names)
        assertTrue("id_based_context_precision" in tier1Names)
        assertTrue("context_entity_recall" in tier1Names)
        val tier2Names = tier2Metrics().map { metric -> metric.name }.toSet()
        assertEquals(5, tier2Names.size)
        assertTrue("tool_call_accuracy" in tier2Names)
        assertTrue("tool_call_f1" in tier2Names)
        assertTrue("agent_goal_accuracy_with_reference" in tier2Names)
        assertTrue("agent_goal_accuracy_without_reference" in tier2Names)
        assertTrue("agent_workflow_completion" in tier2Names)
        val tier3Names = tier3Metrics().map { metric -> metric.name }.toSet()
        assertEquals(11, tier3Names.size)
        assertTrue("answer_accuracy" in tier3Names)
        assertTrue("answer_correctness" in tier3Names)
        assertTrue("factual_correctness" in tier3Names)
        assertTrue("topic_adherence" in tier3Names)
        assertTrue("noise_sensitivity" in tier3Names)
        assertTrue("summary_score" in tier3Names)
        assertTrue("quoted_spans_alignment" in tier3Names)
        assertTrue("chrf_score" in tier3Names)
        assertTrue("bleu_score" in tier3Names)
        assertTrue("rouge_score" in tier3Names)
        assertTrue("semantic_similarity" in tier3Names)
        val tier4Names = tier4Metrics().map { metric -> metric.name }.toSet()
        assertEquals(8, tier4Names.size)
        assertTrue("domain_specific_rubrics" in tier4Names)
        assertTrue("rubrics_score_without_reference" in tier4Names)
        assertTrue("rubrics_score_with_reference" in tier4Names)
        assertTrue("instance_specific_rubrics" in tier4Names)
        assertTrue("sql_semantic_equivalence" in tier4Names)
        assertTrue("data_compare_score" in tier4Names)
        assertTrue("multi_modal_relevance" in tier4Names)
        assertTrue("multi_modal_faithfulness" in tier4Names)
        assertEquals("0.0.1", VERSION)
    }

    @Test
    fun topLevelAevaluateWorks() =
        runBlocking {
            val dataset =
                EvaluationDataset(
                    listOf(
                        SingleTurnSample(
                            userInput = "What is Kotlin language",
                            response = "Kotlin is language.",
                            retrievedContexts = listOf("Kotlin language on JVM", "Python language"),
                            referenceContexts = listOf("Kotlin language", "JVM runtime"),
                            reference = "Kotlin language",
                        ),
                    ),
                )

            val result = aevaluate(dataset = dataset)
            assertTrue(result.scores.isNotEmpty())
        }

    @Test
    fun topLevelWithCacheWrapsProviders() =
        runBlocking {
            val llm = ApiFakeLlm("0.5")
            val embedding = ApiFakeEmbedding(listOf(0.1f, 0.2f))
            val cache = InMemoryCacheBackend()

            val cachedLlm = withCache(llm, cache)
            val cachedEmbedding = withCache(embedding, cache)

            cachedLlm.generateText("prompt")
            cachedLlm.generateText("prompt")
            cachedEmbedding.embedText("text")
            cachedEmbedding.embedText("text")

            assertEquals(1, llm.calls)
            assertEquals(1, embedding.calls)
        }

    @Test
    fun backendRegistryFacadeIsAvailable() {
        val names = backendRegistry().availableNames()
        assertTrue("inmemory" in names)
    }

    @Test
    fun optimizerFacadesAreAvailable() {
        val genetic = geneticOptimizer()
        val dspy = dspyOptimizer()
        val dspyWithCache = dspyOptimizer(InMemoryCacheBackend())

        assertTrue(genetic is GeneticOptimizer)
        assertTrue(dspy is DspyOptimizer)
        assertTrue(dspyWithCache is DspyOptimizer)
    }
}

private class ApiFakeLlm(
    private val output: String,
) : BaseRagasLlm {
    var calls: Int = 0
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        calls += 1
        return LlmResult(generations = listOf(LlmGeneration(output)))
    }
}

private class ApiFakeEmbedding(
    private val vector: List<Float>,
) : BaseRagasEmbedding {
    var calls: Int = 0

    override suspend fun embedText(text: String): List<Float> {
        calls += 1
        return vector
    }
}
