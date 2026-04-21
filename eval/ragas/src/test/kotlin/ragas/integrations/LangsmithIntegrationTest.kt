package ragas.integrations

import ragas.integrations.tracing.InMemoryTraceObserver
import ragas.integrations.tracing.RunFailed
import ragas.integrations.tracing.RunStarted
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LangsmithIntegrationTest {
    @Test
    fun toDatasetMapsLangsmithRecordFields() {
        val dataset =
            LangsmithIntegration.toDataset(
                listOf(
                    LangsmithRecord(
                        input = "What is Kotlin?",
                        output = "A JVM language.",
                        retrievedContexts = listOf("Kotlin runs on JVM"),
                        referenceContexts = listOf("Kotlin is a JVM language"),
                        reference = "Kotlin is statically typed.",
                    ),
                ),
            )

        val sample = dataset.samples.single()
        assertEquals("What is Kotlin?", sample.userInput)
        assertEquals("A JVM language.", sample.response)
        assertEquals(listOf("Kotlin runs on JVM"), sample.retrievedContexts)
        assertEquals(listOf("Kotlin is a JVM language"), sample.referenceContexts)
        assertEquals("Kotlin is statically typed.", sample.reference)
    }

    @Test
    fun evaluateRecordsEmitsRunStartedWithMetadataThenFailsUnsupported() {
        val observer = InMemoryTraceObserver()

        val thrown =
            assertFailsWith<UnsupportedOperationException> {
                LangsmithIntegration.evaluateRecords(
                    records = listOf(LangsmithRecord(input = "q", output = "a")),
                    runName = "langsmith-phase1",
                    tags = mapOf("env" to "test"),
                    metadata = mapOf("tenant" to "acme"),
                    observers = listOf(observer),
                )
            }

        assertTrue(thrown.message?.contains("Integration 'langsmith' is not yet implemented") == true)
        val started = observer.events.first() as RunStarted
        assertEquals("langsmith", started.framework)
        assertEquals("langsmith-phase1", started.runName)
        assertEquals(mapOf("env" to "test"), started.tags)
        assertEquals(mapOf("tenant" to "acme"), started.metadata)

        val failed = observer.events.last() as RunFailed
        assertEquals("UnsupportedOperationException", failed.errorType)
        assertTrue(failed.errorMessage.contains("langsmith"))
    }
}
