package ragas

import ragas.integrations.LangChainIntegration
import ragas.integrations.LangChainRecord
import ragas.integrations.LlamaIndexIntegration
import ragas.integrations.LlamaIndexRecord
import ragas.integrations.traceEvaluation
import ragas.integrations.tracing.InMemoryTraceObserver
import ragas.integrations.tracing.LangfuseStyleObserver
import ragas.integrations.tracing.MetricRowLogged
import ragas.integrations.tracing.MlflowStyleObserver
import ragas.integrations.tracing.RunCompleted
import ragas.integrations.tracing.RunFailed
import ragas.integrations.tracing.RunStarted
import ragas.integrations.tracing.TraceObserver
import ragas.model.EvaluationResult
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TracingIntegrationTest {
    @Test
    fun langchainAdapterEmitsTraceLifecycleAndRows() {
        val records =
            listOf(
                LangChainRecord(
                    question = "What is Kotlin language",
                    answer = "Kotlin is language.",
                    retrievedContexts = listOf("Kotlin language on JVM", "Python language"),
                    referenceContexts = listOf("Kotlin language", "JVM runtime"),
                    reference = "Kotlin language",
                ),
            )

        val memory = InMemoryTraceObserver()
        val langfuse = LangfuseStyleObserver()
        val mlflow = MlflowStyleObserver()

        LangChainIntegration.evaluateRecords(
            records = records,
            observers = listOf(memory, langfuse, mlflow),
            runName = "lc-run",
            tags = mapOf("env" to "test"),
            metadata = mapOf("framework_version" to "mock"),
        )

        assertTrue(memory.events.first() is RunStarted)
        assertTrue(memory.events.last() is RunCompleted)
        assertTrue(memory.events.any { event -> event is MetricRowLogged })

        val runId = memory.events.first().runId
        val lfTrace = langfuse.getTrace(runId)
        assertTrue(lfTrace != null)
        assertTrue(lfTrace.endTimeMs != null)

        val mlRun = mlflow.getRun(runId)
        assertTrue(mlRun != null)
        assertEquals("FINISHED", mlRun.status)
        assertTrue(mlRun.metrics.isNotEmpty())
    }

    @Test
    fun llamaIndexAdapterEmitsTraceLifecycleAndRows() {
        val records =
            listOf(
                LlamaIndexRecord(
                    query = "What is Kotlin language",
                    response = "Kotlin is language.",
                    retrievedNodes = listOf("Kotlin language on JVM", "Python language"),
                    referenceNodes = listOf("Kotlin language", "JVM runtime"),
                    reference = "Kotlin language",
                ),
            )

        val memory = InMemoryTraceObserver()
        LlamaIndexIntegration.evaluateRecords(
            records = records,
            observers = listOf(memory),
            runName = "li-run",
        )

        assertTrue(memory.events.first() is RunStarted)
        assertTrue(memory.events.any { event -> event is MetricRowLogged })
        assertTrue(memory.events.last() is RunCompleted)
    }

    @Test
    fun observerFailureOnRunFailedDoesNotMaskOriginalError() {
        val root = IllegalStateException("root failure")
        val observerFailure = RuntimeException("observer failure")
        val observer =
            TraceObserver { event ->
                if (event is RunFailed) {
                    throw observerFailure
                }
            }

        val thrown =
            assertFailsWith<IllegalStateException> {
                traceEvaluation(
                    framework = "test",
                    runName = "run",
                    tags = emptyMap(),
                    metadata = emptyMap(),
                    observers = listOf(observer),
                ) {
                    throw root
                }
            }

        assertTrue(thrown === root)
        assertTrue(thrown.suppressedExceptions.contains(observerFailure))
    }
}
