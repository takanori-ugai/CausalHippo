package ragas.metrics.collections

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.metrics.tokenSet
import ragas.model.SingleTurnSample

/**
 * Measures how well retrieved contexts cover the user query.
 *
 * Uses token-overlap coverage between `user_input` and `retrieved_contexts`,
 * blending global query coverage and average per-context coverage.
 */
class ContextRelevanceMetric :
    BaseMetric(
        name = "context_relevance",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("user_input", "retrieved_contexts")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric {
    /**
     * Computes context relevance from query/context token overlap signals.
     *
     * Returns `0.0` when query or contexts are missing/empty; otherwise returns
     * a continuous score in [0.0, 1.0].
     *
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val userInput = sample.userInput.orEmpty().trim()
        val contexts = sample.retrievedContexts.orEmpty().filter { it.isNotBlank() }
        if (userInput.isBlank() || contexts.isEmpty()) {
            return 0.0
        }

        val queryTokens = tokenSet(userInput)
        if (queryTokens.isEmpty()) {
            return 0.0
        }

        val contextTokenSets = contexts.map { context -> tokenSet(context) }
        val allContextTokens = contextTokenSets.flatten().toSet()
        val globalCoverage = queryTokens.intersect(allContextTokens).size.toDouble() / queryTokens.size.toDouble()

        val avgContextCoverage =
            contextTokenSets
                .map { contextTokens ->
                    queryTokens.intersect(contextTokens).size.toDouble() / queryTokens.size.toDouble()
                }.average()

        return clamp01((0.75 * globalCoverage) + (0.25 * avgContextCoverage))
    }
}
