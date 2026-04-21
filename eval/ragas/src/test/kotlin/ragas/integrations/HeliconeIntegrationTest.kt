package ragas.integrations

import ragas.integrations.tracing.InMemoryTraceObserver
import ragas.integrations.tracing.RunFailed
import ragas.integrations.tracing.RunStarted
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class HeliconeIntegrationTest {
    @Test
    fun toDatasetMapsHeliconeRecordFields() {
        val dataset =
            HeliconeIntegration.toDataset(
                listOf(
                    HeliconeRecord(
                        input = "What is RAG?",
                        output = "Retrieval-augmented generation.",
                        retrievedContexts = listOf("RAG combines retrieval with generation"),
                        referenceContexts = listOf("RAG stands for retrieval augmented generation"),
                        reference = "RAG uses retrieved contexts.",
                    ),
                ),
            )

        val sample = dataset.samples.single()
        assertEquals("What is RAG?", sample.userInput)
        assertEquals("Retrieval-augmented generation.", sample.response)
        assertEquals(listOf("RAG combines retrieval with generation"), sample.retrievedContexts)
        assertEquals(listOf("RAG stands for retrieval augmented generation"), sample.referenceContexts)
        assertEquals("RAG uses retrieved contexts.", sample.reference)
    }

    @Test
    fun evaluateRecordsEmitsRunStartedWithMetadataThenFailsUnsupported() {
        val observer = InMemoryTraceObserver()

        val thrown =
            assertFailsWith<UnsupportedOperationException> {
                HeliconeIntegration.evaluateRecords(
                    records = listOf(HeliconeRecord(input = "q", output = "a")),
                    runName = "helicone-phase1",
                    tags = mapOf("env" to "test"),
                    metadata = mapOf("tenant" to "acme"),
                    observers = listOf(observer),
                )
            }

        assertTrue(thrown.message?.contains("Integration 'helicone' is not yet implemented") == true)
        val started = observer.events.first() as RunStarted
        assertEquals("helicone", started.framework)
        assertEquals("helicone-phase1", started.runName)
        assertEquals(mapOf("env" to "test"), started.tags)
        assertEquals(mapOf("tenant" to "acme"), started.metadata)

        val failed = observer.events.last() as RunFailed
        assertEquals("UnsupportedOperationException", failed.errorType)
        assertTrue(failed.errorMessage.contains("helicone"))
    }
}
