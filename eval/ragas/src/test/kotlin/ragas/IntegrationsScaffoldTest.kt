package ragas

import ragas.integrations.LangChainIntegration
import ragas.integrations.LangChainRecord
import ragas.integrations.LlamaIndexIntegration
import ragas.integrations.LlamaIndexRecord
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class IntegrationsScaffoldTest {
    @Test
    fun langchainIntegrationConvertsAndEvaluates() {
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

        val dataset = LangChainIntegration.toDataset(records)
        val result = LangChainIntegration.evaluateRecords(records)
        val payload = LangChainIntegration.toMetricPayload(result)

        assertEquals(1, dataset.samples.size)
        assertEquals(1, payload.size)
        assertTrue(payload.first().containsKey("answer_relevancy"))
    }

    @Test
    fun llamaIndexIntegrationConvertsAndEvaluates() {
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

        val dataset = LlamaIndexIntegration.toDataset(records)
        val result = LlamaIndexIntegration.evaluateRecords(records)
        val payload = LlamaIndexIntegration.toMetricPayload(result)

        assertEquals(1, dataset.samples.size)
        assertEquals(1, payload.size)
        assertTrue(payload.first().containsKey("answer_relevancy"))
    }
}
