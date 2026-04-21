package ragas

import ragas.integrations.traceEvaluation
import ragas.integrations.tracing.RunFailed
import ragas.integrations.tracing.TraceObserver
import kotlin.test.Test
import kotlin.test.assertTrue

class TracingSelfSuppressionTest {
    @Test
    fun observerSelfSuppressionDoesNotThrow() {
        val root = IllegalStateException("root failure")
        val observer =
            TraceObserver { event ->
                if (event is RunFailed) {
                    // Rethrow the same exception instance
                    throw root
                }
            }

        // This should not throw IllegalArgumentException("Self-suppression not permitted")
        val thrown =
            try {
                traceEvaluation(
                    framework = "test",
                    runName = "run",
                    tags = emptyMap(),
                    metadata = emptyMap(),
                    observers = listOf(observer),
                ) {
                    throw root
                }
                null
            } catch (e: Throwable) {
                e
            }

        assertTrue(thrown === root)
        // Suppressed exceptions should be empty because we didn't add it (it was the same instance)
        assertTrue(thrown.suppressedExceptions.isEmpty())
    }
}
