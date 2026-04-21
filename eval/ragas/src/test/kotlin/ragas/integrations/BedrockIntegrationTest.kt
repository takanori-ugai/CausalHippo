package ragas.integrations

import ragas.integrations.tracing.InMemoryTraceObserver
import ragas.integrations.tracing.RunFailed
import ragas.integrations.tracing.RunStarted
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BedrockIntegrationTest {
    @Test
    fun toDatasetMapsBedrockRecordFields() {
        val dataset =
            BedrockIntegration.toDataset(
                listOf(
                    BedrockRecord(
                        input = "What is Amazon Bedrock?",
                        output = "A managed foundation model service.",
                        retrievedContexts = listOf("Bedrock offers managed FM APIs"),
                        referenceContexts = listOf("Bedrock integrates model providers on AWS"),
                        reference = "Bedrock is an AWS managed model platform.",
                    ),
                ),
            )

        val sample = dataset.samples.single()
        assertEquals("What is Amazon Bedrock?", sample.userInput)
        assertEquals("A managed foundation model service.", sample.response)
        assertEquals(listOf("Bedrock offers managed FM APIs"), sample.retrievedContexts)
        assertEquals(listOf("Bedrock integrates model providers on AWS"), sample.referenceContexts)
        assertEquals("Bedrock is an AWS managed model platform.", sample.reference)
    }

    @Test
    fun toDatasetRejectsRecordMetadataUntilSupported() {
        val error =
            assertFailsWith<IllegalArgumentException> {
                BedrockIntegration.toDataset(
                    listOf(
                        BedrockRecord(
                            input = "q",
                            output = "a",
                            metadata = mapOf("tenant" to "acme"),
                        ),
                    ),
                )
            }

        assertTrue(error.message.orEmpty().contains("BedrockRecord.metadata is not supported yet"))
    }

    @Test
    fun evaluateRecordsEmitsRunStartedWithMetadataThenFailsUnsupported() {
        val observer = InMemoryTraceObserver()

        val thrown =
            assertFailsWith<UnsupportedOperationException> {
                BedrockIntegration.evaluateRecords(
                    records = listOf(BedrockRecord(input = "q", output = "a")),
                    runName = "bedrock-phase3",
                    tags = mapOf("env" to "test"),
                    metadata = mapOf("tenant" to "acme"),
                    observers = listOf(observer),
                )
            }

        assertTrue(thrown.message?.contains("Integration 'bedrock' is not yet implemented") == true)
        val started = observer.events.first() as RunStarted
        assertEquals("bedrock", started.framework)
        assertEquals("bedrock-phase3", started.runName)
        assertEquals(mapOf("env" to "test"), started.tags)
        assertEquals(mapOf("tenant" to "acme"), started.metadata)

        val failed = observer.events.last() as RunFailed
        assertEquals("UnsupportedOperationException", failed.errorType)
        assertTrue(failed.errorMessage.contains("bedrock"))
    }
}
