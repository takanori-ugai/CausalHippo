package ragas.integrations

import ragas.integrations.tracing.InMemoryTraceObserver
import ragas.integrations.tracing.RunFailed
import ragas.integrations.tracing.RunStarted
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class OpikIntegrationTest {
    @Test
    fun toDatasetMapsOpikRecordFields() {
        val dataset =
            OpikIntegration.toDataset(
                listOf(
                    OpikRecord(
                        input = "What is a metric?",
                        output = "A measurable score.",
                        retrievedContexts = listOf("Metrics quantify system quality"),
                        referenceContexts = listOf("A metric is a measurable standard"),
                        reference = "Metrics are numeric signals.",
                    ),
                ),
            )

        val sample = dataset.samples.single()
        assertEquals("What is a metric?", sample.userInput)
        assertEquals("A measurable score.", sample.response)
        assertEquals(listOf("Metrics quantify system quality"), sample.retrievedContexts)
        assertEquals(listOf("A metric is a measurable standard"), sample.referenceContexts)
        assertEquals("Metrics are numeric signals.", sample.reference)
    }

    @Suppress("DEPRECATION_ERROR")
    @Test
    fun evaluateRecordsEmitsRunStartedWithMetadataThenFailsUnsupported() {
        val observer = InMemoryTraceObserver()

        val thrown =
            assertFailsWith<UnsupportedOperationException> {
                OpikIntegration.evaluateRecords(
                    records = listOf(OpikRecord(input = "q", output = "a")),
                    runName = "opik-phase1",
                    tags = mapOf("env" to "test"),
                    metadata = mapOf("tenant" to "acme"),
                    observers = listOf(observer),
                )
            }

        assertTrue(thrown.message?.contains("Integration 'opik' is not yet implemented") == true)
        assertEquals(2, observer.events.size)
        val started = observer.events[0] as RunStarted
        assertEquals("opik", started.framework)
        assertEquals("opik-phase1", started.runName)
        assertEquals(mapOf("env" to "test"), started.tags)
        assertEquals(mapOf("tenant" to "acme"), started.metadata)

        val failed = observer.events[1] as RunFailed
        assertEquals("UnsupportedOperationException", failed.errorType)
        assertTrue(failed.errorMessage.contains("opik"))
    }
}
