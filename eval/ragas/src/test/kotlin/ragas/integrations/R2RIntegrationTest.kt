package ragas.integrations

import ragas.integrations.tracing.InMemoryTraceObserver
import ragas.integrations.tracing.RunFailed
import ragas.integrations.tracing.RunStarted
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class R2RIntegrationTest {
    @Test
    fun toDatasetMapsR2RRecordFields() {
        val dataset =
            R2RIntegration.toDataset(
                listOf(
                    R2RRecord(
                        input = "What is retrieval-to-response?",
                        output = "A retrieval-aware response pipeline.",
                        retrievedContexts = listOf("R2R pipelines connect retrieval and generation"),
                        referenceContexts = listOf("Retrieval provides grounding context"),
                        reference = "R2R links retrieval and response generation.",
                    ),
                ),
            )

        val sample = dataset.samples.single()
        assertEquals("What is retrieval-to-response?", sample.userInput)
        assertEquals("A retrieval-aware response pipeline.", sample.response)
        assertEquals(listOf("R2R pipelines connect retrieval and generation"), sample.retrievedContexts)
        assertEquals(listOf("Retrieval provides grounding context"), sample.referenceContexts)
        assertEquals("R2R links retrieval and response generation.", sample.reference)
    }

    @Test
    fun evaluateRecordsEmitsRunStartedWithMetadataThenFailsUnsupported() {
        val observer = InMemoryTraceObserver()

        val thrown =
            assertFailsWith<UnsupportedOperationException> {
                R2RIntegration.evaluateRecords(
                    records = listOf(R2RRecord(input = "q", output = "a")),
                    runName = "r2r-phase2",
                    tags = mapOf("env" to "test"),
                    metadata = mapOf("tenant" to "acme"),
                    observers = listOf(observer),
                )
            }

        assertTrue(thrown.message?.contains("Integration 'r2r' is not yet implemented") == true)
        val started = observer.events.first() as RunStarted
        assertEquals("r2r", started.framework)
        assertEquals("r2r-phase2", started.runName)
        assertEquals(mapOf("env" to "test"), started.tags)
        assertEquals(mapOf("tenant" to "acme"), started.metadata)

        val failed = observer.events.last() as RunFailed
        assertEquals("UnsupportedOperationException", failed.errorType)
        assertTrue(failed.errorMessage.contains("r2r"))
    }
}
