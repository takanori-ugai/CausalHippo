package ragas.integrations

import ragas.integrations.tracing.InMemoryTraceObserver
import ragas.integrations.tracing.RunFailed
import ragas.integrations.tracing.RunStarted
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SwarmIntegrationTest {
    @Test
    fun toDatasetMapsSwarmRecordFields() {
        val dataset =
            SwarmIntegration.toDataset(
                listOf(
                    SwarmRecord(
                        input = "What is a swarm workflow?",
                        output = "A multi-agent coordination flow.",
                        retrievedContexts = listOf("Swarm patterns coordinate specialized agents"),
                        referenceContexts = listOf("Swarm systems orchestrate multiple agents"),
                        reference = "Swarm workflows coordinate multiple agents.",
                    ),
                ),
            )

        val sample = dataset.samples.single()
        assertEquals("What is a swarm workflow?", sample.userInput)
        assertEquals("A multi-agent coordination flow.", sample.response)
        assertEquals(listOf("Swarm patterns coordinate specialized agents"), sample.retrievedContexts)
        assertEquals(listOf("Swarm systems orchestrate multiple agents"), sample.referenceContexts)
        assertEquals("Swarm workflows coordinate multiple agents.", sample.reference)
    }

    @Test
    fun evaluateRecordsEmitsRunStartedWithMetadataThenFailsUnsupported() {
        val observer = InMemoryTraceObserver()

        val thrown =
            assertFailsWith<UnsupportedOperationException> {
                SwarmIntegration.evaluateRecords(
                    records = listOf(SwarmRecord(input = "q", output = "a")),
                    runName = "swarm-phase3",
                    tags = mapOf("env" to "test"),
                    metadata = mapOf("tenant" to "acme"),
                    observers = listOf(observer),
                )
            }

        assertTrue(thrown.message?.contains("Integration 'swarm' is not yet implemented") == true)
        assertEquals(2, observer.events.size)
        val started = observer.events[0] as RunStarted
        assertEquals("swarm", started.framework)
        assertEquals("swarm-phase3", started.runName)
        assertEquals(mapOf("env" to "test"), started.tags)
        assertEquals(mapOf("tenant" to "acme"), started.metadata)

        val failed = observer.events[1] as RunFailed
        assertEquals("UnsupportedOperationException", failed.errorType)
        assertTrue(failed.errorMessage.contains("swarm"))
    }
}
