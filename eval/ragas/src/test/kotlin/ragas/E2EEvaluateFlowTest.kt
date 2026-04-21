package ragas

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import ragas.embeddings.BaseRagasEmbedding
import ragas.evaluation.aevaluate
import ragas.evaluation.evaluate
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.MetricWithEmbeddings
import ragas.metrics.MetricWithLlm
import ragas.metrics.SingleTurnMetric
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class E2EEvaluateFlowTest {
    @Test
    fun evaluateInjectsLlmAndEmbeddingsAndAggregatesScores() {
        val dataset =
            EvaluationDataset(
                listOf(
                    SingleTurnSample(userInput = "u1", response = "r1", reference = "ref1"),
                    SingleTurnSample(userInput = "u2", response = "r2", reference = "ref2"),
                ),
            )

        val llm = FakeE2ELlm("0.7")
        val embeddings = FakeE2EEmbedding(listOf(0.1f, 0.2f, 0.3f))
        val metric = InjectedDependenciesMetric()

        val result =
            evaluate(
                dataset = dataset,
                metrics = listOf(metric),
                llm = llm,
                embeddings = embeddings,
            )

        assertEquals(2, result.scores.size)
        assertTrue((result.scores[0]["injected_metric"] as Double) > 0.0)
        assertTrue((result.scores[1]["injected_metric"] as Double) > 0.0)
        assertEquals(null, metric.llm)
        assertEquals(null, metric.embeddings)
        assertEquals(2, llm.calls.get())
        assertEquals(2, embeddings.calls)
    }

    @Test
    fun aevaluateDoesNotLeakInjectedLlmAcrossConcurrentCalls() =
        runBlocking {
            val dataset = EvaluationDataset(listOf(SingleTurnSample(response = "r1")))
            val metric = DelayedLlmMetric()
            val firstLlm = TaggedLlm("first")
            val secondLlm = TaggedLlm("second")

            val firstResultDeferred =
                async {
                    aevaluate(
                        dataset = dataset,
                        metrics = listOf(metric),
                        llm = firstLlm,
                    )
                }

            assertTrue(metric.awaitInit())

            val secondResultDeferred =
                async {
                    aevaluate(
                        dataset = dataset,
                        metrics = listOf(metric),
                        llm = secondLlm,
                    )
                }

            val firstScore = firstResultDeferred.await().scores.single()["delayed_llm_metric"]
            val secondScore = secondResultDeferred.await().scores.single()["delayed_llm_metric"]

            assertEquals("first", firstScore)
            assertEquals("second", secondScore)
        }

    @Test
    fun aevaluateResetsInjectedDependenciesWhenMetricInitFails() {
        val dataset = EvaluationDataset(listOf(SingleTurnSample(response = "r1")))
        val metric = InitFailingMetric()
        val llm = TaggedLlm("tag")
        val embeddings = FakeE2EEmbedding(listOf(0.1f))

        assertFailsWith<IllegalStateException> {
            evaluate(
                dataset = dataset,
                metrics = listOf(metric),
                llm = llm,
                embeddings = embeddings,
            )
        }

        assertEquals(null, metric.llm)
        assertEquals(null, metric.embeddings)
    }
}

private class InjectedDependenciesMetric :
    BaseMetric(
        name = "injected_metric",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric,
    MetricWithLlm,
    MetricWithEmbeddings {
    override var llm: BaseRagasLlm? = null
    override var embeddings: BaseRagasEmbedding? = null

    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val llmInstance = checkNotNull(llm)
        val embeddingInstance = checkNotNull(embeddings)
        val llmScore =
            llmInstance
                .generateText(sample.response.orEmpty())
                .generations
                .first()
                .text
                .toDouble()
        val embeddingScore = embeddingInstance.embedText(sample.response.orEmpty()).sum().toDouble()
        return llmScore + embeddingScore
    }
}

private class FakeE2ELlm(
    private val output: String,
) : BaseRagasLlm {
    val calls = AtomicInteger(0)
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        calls.incrementAndGet()
        return LlmResult(generations = listOf(LlmGeneration(output)))
    }
}

private class FakeE2EEmbedding(
    private val vector: List<Float>,
) : BaseRagasEmbedding {
    var calls: Int = 0

    override suspend fun embedText(text: String): List<Float> {
        calls += 1
        return vector
    }
}

private class DelayedLlmMetric :
    BaseMetric(
        name = "delayed_llm_metric",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response")),
        outputType = MetricOutputType.DISCRETE,
    ),
    SingleTurnMetric,
    MetricWithLlm {
    override var llm: BaseRagasLlm? = null

    private val initSignal = CompletableDeferred<Unit>()

    override suspend fun init(runConfig: RunConfig) {
        validateRequiredColumns()
        initSignal.complete(Unit)
    }

    suspend fun awaitInit(timeoutMillis: Long = 2_000): Boolean =
        try {
            withTimeout(timeoutMillis) {
                initSignal.await()
            }
            true
        } catch (_: kotlinx.coroutines.TimeoutCancellationException) {
            false
        }

    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        delay(150)
        return checkNotNull(llm)
            .generateText("any")
            .generations
            .first()
            .text
    }
}

private class TaggedLlm(
    private val tag: String,
) : BaseRagasLlm {
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult = LlmResult(generations = listOf(LlmGeneration(tag)))
}

private class InitFailingMetric :
    BaseMetric(
        name = "init_failing_metric",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric,
    MetricWithLlm,
    MetricWithEmbeddings {
    override var llm: BaseRagasLlm? = null
    override var embeddings: BaseRagasEmbedding? = null

    override suspend fun init(runConfig: RunConfig) {
        validateRequiredColumns()
        error("init failed")
    }

    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any = 0.0
}
