package ragas.integrations

import ragas.integrations.tracing.InMemoryTraceObserver
import ragas.integrations.tracing.RunFailed
import ragas.integrations.tracing.RunStarted
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LangGraphIntegrationTest {
    @Test
    fun toDatasetMapsLangGraphRecordFields() {
        val dataset =
            LangGraphIntegration.toDataset(
                listOf(
                    LangGraphRecord(
                        input = "What is a graph workflow?",
                        output = "A node and edge based workflow.",
                        retrievedContexts = listOf("LangGraph composes graph-based agents"),
                        referenceContexts = listOf("Graphs model stateful workflows"),
                        reference = "Graph workflows use nodes and edges.",
                    ),
                ),
            )

        val sample = dataset.samples.single()
        assertEquals("What is a graph workflow?", sample.userInput)
        assertEquals("A node and edge based workflow.", sample.response)
        assertEquals(listOf("LangGraph composes graph-based agents"), sample.retrievedContexts)
        assertEquals(listOf("Graphs model stateful workflows"), sample.referenceContexts)
        assertEquals("Graph workflows use nodes and edges.", sample.reference)
    }

    @Test
    fun evaluateRecordsEmitsRunStartedWithMetadataThenFailsUnsupported() {
        val observer = InMemoryTraceObserver()

        val thrown =
            assertFailsWith<UnsupportedOperationException> {
                LangGraphIntegration.evaluateRecords(
                    records = listOf(LangGraphRecord(input = "q", output = "a")),
                    runName = "langgraph-phase2",
                    tags = mapOf("env" to "test"),
                    metadata = mapOf("tenant" to "acme"),
                    observers = listOf(observer),
                )
            }

        assertTrue(thrown.message?.contains("Integration 'langgraph' is not yet implemented") == true)
        val started = observer.events.first() as RunStarted
        assertEquals("langgraph", started.framework)
        assertEquals("langgraph-phase2", started.runName)
        assertEquals(mapOf("env" to "test"), started.tags)
        assertEquals(mapOf("tenant" to "acme"), started.metadata)

        val failed = observer.events.last() as RunFailed
        assertEquals("UnsupportedOperationException", failed.errorType)
        assertTrue(failed.errorMessage.contains("langgraph"))
    }
}
