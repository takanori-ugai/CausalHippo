package ragas

import kotlinx.coroutines.runBlocking
import ragas.metrics.collections.QuotedSpansAlignmentMetric
import ragas.model.SingleTurnSample
import kotlin.test.Test
import kotlin.test.assertEquals

class QuotedSpansAlignmentMetricParityTest {
    @Test
    fun returnsOneWhenNoQuotedSpansEvenIfContextsMissing() =
        runBlocking {
            val metric = QuotedSpansAlignmentMetric()
            val sample = SingleTurnSample(response = "No quote here.", retrievedContexts = emptyList())

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(1.0, score, 1e-9)
        }

    @Test
    fun matchesPythonApostropheRegexBehavior() =
        runBlocking {
            val metric = QuotedSpansAlignmentMetric(casefold = true, minSpanWords = 2)
            val sample =
                SingleTurnSample(
                    response = "It's called 'Alpha Beta' in the report.",
                    retrievedContexts = listOf("alpha beta appears in the report."),
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(1.0, score, 1e-9)
        }
}
