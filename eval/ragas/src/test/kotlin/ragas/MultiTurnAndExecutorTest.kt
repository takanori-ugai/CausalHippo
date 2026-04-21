package ragas

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import ragas.evaluation.evaluate
import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.MultiTurnMetric
import ragas.model.AiMessage
import ragas.model.EvaluationDataset
import ragas.model.HumanMessage
import ragas.model.MultiTurnSample
import ragas.runtime.Executor
import ragas.runtime.RunConfig
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class MultiTurnAndExecutorTest {
    @Test
    fun evaluateRunsMultiTurnMetric() {
        val sample =
            MultiTurnSample(
                userInput =
                    listOf(
                        HumanMessage("Hi"),
                        AiMessage("Hello"),
                    ),
                reference = "Greeting",
            )

        val dataset = EvaluationDataset(listOf(sample))
        val result = evaluate(dataset = dataset, metrics = listOf(ConversationLengthMetric()))

        assertEquals(2.0, result.scores.first()["conversation_length"])
    }

    @Test
    fun evaluateIgnoresIncompatibleColumnRemapTypeForMultiTurn() {
        val sample = MultiTurnSample(userInput = emptyList(), reference = "Greeting")
        val dataset = EvaluationDataset(listOf(sample))

        val result =
            evaluate(
                dataset = dataset,
                metrics = listOf(ConversationLengthMetric()),
                columnMap = mapOf("user_input" to "reference"),
            )

        assertEquals(0.0, result.scores.first()["conversation_length"])
    }

    @Test
    fun evaluateRejectsUnknownColumnMapSourceForMultiTurn() {
        val sample = MultiTurnSample(userInput = emptyList(), reference = "Greeting")
        val dataset = EvaluationDataset(listOf(sample))

        val error =
            assertFailsWith<IllegalArgumentException> {
                evaluate(
                    dataset = dataset,
                    metrics = listOf(ConversationLengthMetric()),
                    columnMap = mapOf("reference" to "custom_reference"),
                )
            }

        assertTrue(error.message.orEmpty().contains("Unsupported columnMap source 'custom_reference'"))
    }

    @Test
    fun executorReturnsNullWhenRaiseExceptionsIsFalse() =
        runBlocking {
            val executor =
                Executor(
                    raiseExceptions = false,
                    runConfig = RunConfig(maxRetries = 1, maxWaitSeconds = 0, timeoutSeconds = 1),
                )
            executor.submit(name = "ok") { 1 }
            executor.submit(name = "boom") { error("fail") }

            val results = executor.aresults()
            assertEquals(2, results.size)
            assertEquals(1, results[0])
            assertEquals(null, results[1])
        }

    @Test
    fun executorTimeoutReturnsNullWhenRaiseExceptionsIsFalse() =
        runBlocking {
            val executor =
                Executor(
                    raiseExceptions = false,
                    runConfig = RunConfig(maxRetries = 1, maxWaitSeconds = 0, timeoutSeconds = 1, maxWorkers = 2),
                )
            executor.submit(name = "timeout") {
                delay(1_500)
                1
            }
            executor.submit(name = "ok") { 2 }

            val results = executor.aresults()
            assertEquals(2, results.size)
            assertEquals(null, results[0])
            assertEquals(2, results[1])
        }

    @Test
    fun executorPropagatesWhenRaiseExceptionsIsTrue() {
        runBlocking {
            val executor =
                Executor(
                    raiseExceptions = true,
                    runConfig = RunConfig(maxRetries = 1, maxWaitSeconds = 0, timeoutSeconds = 1),
                )
            executor.submit { 1 }
            executor.submit { error("hard fail") }

            assertFailsWith<IllegalStateException> {
                executor.aresults()
            }
        }
    }

    @Test
    fun executorCanBeCancelledBeforeRun() =
        runBlocking {
            val executor = Executor(runConfig = RunConfig(timeoutSeconds = 1, maxWorkers = 2))
            repeat(3) {
                executor.submit {
                    delay(50)
                    it
                }
            }

            executor.cancel()
            val results = executor.aresults()
            assertEquals(3, results.size)
            assertEquals(listOf(null, null, null), results)
        }

    @Test
    fun executorCancelStopsInFlightAndPreservesResultAlignment() =
        runBlocking {
            val executor = Executor(runConfig = RunConfig(timeoutSeconds = 5, maxWorkers = 4))
            val completed = AtomicInteger(0)
            repeat(6) { i ->
                executor.submit {
                    delay(2_000)
                    completed.incrementAndGet()
                    i
                }
            }

            val results =
                coroutineScope {
                    val resultsDeferred = async { executor.aresults() }
                    delay(100)
                    executor.cancel()
                    resultsDeferred.await()
                }

            assertEquals(6, results.size)
            assertTrue(results.all { it == null })
            assertEquals(0, completed.get())
        }

    @Test
    fun executorSupportsConcurrentSubmitCalls() =
        runBlocking {
            val executor = Executor(runConfig = RunConfig(timeoutSeconds = 1, maxWorkers = 4))
            val expected = 200

            coroutineScope {
                repeat(expected) { i ->
                    launch(Dispatchers.Default) {
                        executor.submit { i }
                    }
                }
            }

            val results = executor.aresults()
            assertEquals(expected, results.size)
            assertEquals((0 until expected).toSet(), results.filterIsInstance<Int>().toSet())
        }
}

private class ConversationLengthMetric :
    BaseMetric(
        name = "conversation_length",
        requiredColumns = mapOf(MetricType.MULTI_TURN to setOf("user_input")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    MultiTurnMetric {
    override suspend fun multiTurnAscore(sample: MultiTurnSample): Any = sample.userInput.size.toDouble()
}
