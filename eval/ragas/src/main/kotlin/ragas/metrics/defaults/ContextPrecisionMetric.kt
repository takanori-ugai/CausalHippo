package ragas.metrics.defaults

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.metrics.tokenSet
import ragas.model.SingleTurnSample

/**
 * Estimates context precision using a token-overlap heuristic.
 *
 * A retrieved context is counted as relevant when it shares at least one token
 * with the response, and the score is `relevant_contexts / total_contexts` in [0.0, 1.0].
 */
class ContextPrecisionMetric :
    BaseMetric(
        name = "context_precision",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("retrieved_contexts", "response")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric {
    /**
     * Computes context precision for one sample using response/context token overlap.
     *
     * Returns `0.0` when retrieved contexts are empty or when the response has no tokens.
     * Otherwise returns a score in [0.0, 1.0].
     *
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val contexts = sample.retrievedContexts.orEmpty()
        if (contexts.isEmpty()) {
            return 0.0
        }

        val responseTokens = tokenSet(sample.response.orEmpty())
        if (responseTokens.isEmpty()) {
            return 0.0
        }

        val relevant =
            contexts.count { context ->
                val contextTokens = tokenSet(context)
                contextTokens.intersect(responseTokens).isNotEmpty()
            }

        return clamp01(relevant.toDouble() / contexts.size.toDouble())
    }
}
