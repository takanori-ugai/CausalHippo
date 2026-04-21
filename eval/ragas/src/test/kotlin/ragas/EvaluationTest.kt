package ragas

import ragas.evaluation.evaluate
import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample
import kotlin.test.Test
import kotlin.test.assertEquals

class EvaluationTest {
    @Test
    fun evaluateRunsSingleTurnMetrics() {
        val dataset =
            EvaluationDataset(
                listOf(
                    SingleTurnSample(
                        userInput = "What is Kotlin?",
                        response = "A JVM language",
                        reference = "A programming language",
                    ),
                    SingleTurnSample(
                        userInput = "What is RAG?",
                        response = "Retrieval-augmented generation",
                        reference = "Retrieval augmented generation",
                    ),
                ),
            )

        val metrics = listOf(ResponseLengthMetric(), ContainsReferenceWordMetric())
        val result = evaluate(dataset = dataset, metrics = metrics)

        assertEquals(2, result.scores.size)
        assertEquals(14, result.scores[0]["response_length"])
        assertEquals(1.0, result.scores[1]["contains_reference_word"])
    }
}

private class ResponseLengthMetric :
    BaseMetric(
        name = "response_length",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric {
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any = sample.response?.length ?: 0
}

private class ContainsReferenceWordMetric :
    BaseMetric(
        name = "contains_reference_word",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response", "reference")),
        outputType = MetricOutputType.BINARY,
    ),
    SingleTurnMetric {
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val responseTokens =
            sample.response
                .orEmpty()
                .lowercase()
                .split("\\s+".toRegex())
                .filter { token -> token.isNotBlank() }
                .toSet()
        val referenceTokens =
            sample.reference
                .orEmpty()
                .lowercase()
                .split("\\s+".toRegex())
                .filter { token -> token.isNotBlank() }
        val contains = referenceTokens.any { token -> token in responseTokens }
        return if (contains) 1.0 else 0.0
    }
}
