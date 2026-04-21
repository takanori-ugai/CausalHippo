package ragas

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.runBlocking
import ragas.evaluation.CostEstimate
import ragas.evaluation.EvaluationCallback
import ragas.evaluation.EvaluationEvent
import ragas.evaluation.TokenUsage
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.llms.StructuredOutputRagasLlm
import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.MetricWithLlm
import ragas.metrics.SingleTurnMetric
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample
import ragas.runtime.Executor
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class EvaluationParityHooksTest {
    @Test
    fun evaluateSupportsColumnRemapAliases() {
        val dataset =
            EvaluationDataset(
                listOf(
                    SingleTurnSample(
                        userInput = null,
                        reference = "remapped-question",
                        response = "answer",
                    ),
                ),
            )

        val result =
            evaluate(
                dataset = dataset,
                metrics = listOf(UserInputEchoMetric()),
                columnMap = mapOf("user_input" to "reference"),
            )

        assertEquals("remapped-question", result.scores.single()["user_input_echo"])
    }

    @Test
    fun evaluateIgnoresIncompatibleColumnRemapTypeForSingleTurn() {
        val dataset =
            EvaluationDataset(
                listOf(
                    SingleTurnSample(
                        userInput = "",
                        retrievedContexts = listOf("ctx-1"),
                        response = "answer",
                    ),
                ),
            )

        val result =
            evaluate(
                dataset = dataset,
                metrics = listOf(UserInputEchoMetric()),
                columnMap = mapOf("user_input" to "retrieved_contexts"),
            )

        assertEquals("", result.scores.single()["user_input_echo"])
    }

    @Test
    fun evaluateRejectsUnknownColumnMapSourceForSingleTurn() {
        val dataset =
            EvaluationDataset(
                listOf(
                    SingleTurnSample(
                        userInput = "",
                        response = "answer",
                    ),
                ),
            )

        val error =
            assertFailsWith<IllegalArgumentException> {
                evaluate(
                    dataset = dataset,
                    metrics = listOf(UserInputEchoMetric()),
                    columnMap = mapOf("user_input" to "custom_question"),
                )
            }

        assertTrue(error.message.orEmpty().contains("Unsupported columnMap source 'custom_question'"))
    }

    @Test
    fun evaluateRejectsDuplicateColumnMapTargetsAfterNormalization() {
        val dataset =
            EvaluationDataset(
                listOf(
                    SingleTurnSample(
                        userInput = "",
                        reference = "remapped-question",
                        response = "answer",
                    ),
                ),
            )

        val error =
            assertFailsWith<IllegalArgumentException> {
                evaluate(
                    dataset = dataset,
                    metrics = listOf(UserInputEchoMetric()),
                    columnMap = mapOf("question" to "reference", "user_input" to "response"),
                )
            }

        assertTrue(error.message.orEmpty().contains("Duplicate columnMap targets after normalization: user_input"))
    }

    @Test
    fun evaluateEmitsCallbacksAndTokenCostHooks() {
        val events = mutableListOf<EvaluationEvent>()
        val callback = EvaluationCallback { event -> events += event }
        val dataset = EvaluationDataset(listOf(SingleTurnSample(userInput = "q", response = "r")))
        val metric = LlmEchoMetric()

        val result =
            evaluate(
                dataset = dataset,
                metrics = listOf(metric),
                llm = EchoLlm("generated answer"),
                callbacks = listOf(callback),
                tokenUsageParser = { _, _ -> TokenUsage(promptTokens = 3, completionTokens = 2) },
                costParser = { usage -> CostEstimate(amount = usage.totalTokens * 0.001) },
            )

        assertEquals("generated answer", result.scores.single()["llm_echo"])
        assertTrue(events.any { event -> event is EvaluationEvent.RunStarted })
        assertTrue(events.any { event -> event is EvaluationEvent.ExecutorReady })
        assertTrue(events.any { event -> event is EvaluationEvent.MetricComputed })
        assertTrue(events.any { event -> event is EvaluationEvent.RowCompleted })
        assertTrue(events.any { event -> event is EvaluationEvent.TokenUsageComputed })
        assertTrue(events.any { event -> event is EvaluationEvent.CostComputed })
        assertTrue(events.any { event -> event is EvaluationEvent.RunCompleted })
    }

    @Test
    fun evaluateEmitsSanitizedRunFailedEvent() {
        val events = mutableListOf<EvaluationEvent>()
        val callback = EvaluationCallback { event -> events += event }
        val dataset = EvaluationDataset(listOf(SingleTurnSample(userInput = "q", response = "r")))

        assertFailsWith<IllegalStateException> {
            evaluate(
                dataset = dataset,
                metrics = listOf(ThrowingMetric()),
                callbacks = listOf(callback),
                raiseExceptions = true,
            )
        }
        val failed = events.filterIsInstance<EvaluationEvent.RunFailed>().single()
        assertTrue(failed.error.type.isNotBlank())
        assertEquals("Error details omitted for safety.", failed.error.message)
    }

    @Test
    fun trackingWrapperDoesNotExposeStructuredCapabilityForNonStructuredDelegate() {
        val dataset = EvaluationDataset(listOf(SingleTurnSample(response = "r")))
        val result =
            evaluate(
                dataset = dataset,
                metrics = listOf(StructuredCapabilityMetric()),
                llm = EchoLlm("plain"),
                tokenUsageParser = { _, _ -> TokenUsage(promptTokens = 1, completionTokens = 1) },
            )

        assertEquals(false, result.scores.single()["structured_supported"])
    }

    @Test
    fun trackingWrapperAccountsUsageForStructuredCalls() {
        var parserCalls = 0
        val events = mutableListOf<EvaluationEvent>()
        val dataset = EvaluationDataset(listOf(SingleTurnSample(response = "r")))

        val result =
            evaluate(
                dataset = dataset,
                metrics = listOf(StructuredDiscreteMetric()),
                llm = StructuredEchoLlm(),
                callbacks = listOf(EvaluationCallback { event -> events += event }),
                tokenUsageParser = { _, _ ->
                    parserCalls += 1
                    TokenUsage(promptTokens = 2, completionTokens = 3)
                },
            )

        assertEquals("ok", result.scores.single()["structured_discrete"])
        assertEquals(1, parserCalls)
        val usageEvents = events.filterIsInstance<EvaluationEvent.TokenUsageComputed>()
        assertEquals(1, usageEvents.size)
        assertEquals(5, usageEvents.single().usage.totalTokens)
    }

    @Test
    fun executorObserverCanCancelEvaluation() =
        runBlocking {
            val dataset = EvaluationDataset(listOf(SingleTurnSample(response = "r1"), SingleTurnSample(response = "r2")))
            val gate = CompletableDeferred<Unit>()
            val metric = BlockingMetric(gate)
            lateinit var captured: Executor

            val result =
                aevaluate(
                    dataset = dataset,
                    metrics = listOf(metric),
                    executorObserver = { executor ->
                        // Cancellation is expected to null out in-flight and pending metric tasks.
                        captured = executor
                        executor.cancel()
                    },
                )

            assertTrue(captured.isCancelled())
            assertEquals(null, result.scores[0]["slow_metric"])
            assertEquals(null, result.scores[1]["slow_metric"])
        }
}

private class UserInputEchoMetric :
    BaseMetric(
        name = "user_input_echo",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("user_input")),
        outputType = MetricOutputType.DISCRETE,
    ),
    SingleTurnMetric {
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any = sample.userInput.orEmpty()
}

private class LlmEchoMetric :
    BaseMetric(
        name = "llm_echo",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response")),
        outputType = MetricOutputType.DISCRETE,
    ),
    SingleTurnMetric,
    MetricWithLlm {
    override var llm: BaseRagasLlm? = null

    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any =
        checkNotNull(llm)
            .generateText("prompt: ${sample.response.orEmpty()}")
            .generations
            .firstOrNull()
            ?.text
            .orEmpty()
}

private class EchoLlm(
    private val output: String,
) : BaseRagasLlm {
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult = LlmResult(generations = listOf(LlmGeneration(output)))
}

private class BlockingMetric(
    private val gate: CompletableDeferred<Unit>,
) : BaseMetric(
        name = "slow_metric",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric {
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        gate.await()
        return 1.0
    }
}

private class StructuredCapabilityMetric :
    BaseMetric(
        name = "structured_supported",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response")),
        outputType = MetricOutputType.BINARY,
    ),
    SingleTurnMetric,
    MetricWithLlm {
    override var llm: BaseRagasLlm? = null

    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any = checkNotNull(llm) is StructuredOutputRagasLlm
}

private class ThrowingMetric :
    BaseMetric(
        name = "throwing_metric",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response")),
        outputType = MetricOutputType.DISCRETE,
    ),
    SingleTurnMetric {
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any = error("boom")
}

private class StructuredDiscreteMetric :
    BaseMetric(
        name = "structured_discrete",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response")),
        outputType = MetricOutputType.DISCRETE,
    ),
    SingleTurnMetric,
    MetricWithLlm {
    override var llm: BaseRagasLlm? = null

    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val llmInstance = checkNotNull(llm)
        val structured = llmInstance as? StructuredOutputRagasLlm ?: return ""
        return structured.generateDiscreteValue("prompt: ${sample.response.orEmpty()}").orEmpty()
    }
}

private class StructuredEchoLlm :
    BaseRagasLlm,
    StructuredOutputRagasLlm {
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult = LlmResult(generations = listOf(LlmGeneration("fallback")))

    override suspend fun generateNumericValue(prompt: String): Double? = 1.0

    override suspend fun generateDiscreteValue(prompt: String): String? = "ok"

    override suspend fun generateRankingItems(prompt: String): List<String>? = listOf("a", "b")
}
