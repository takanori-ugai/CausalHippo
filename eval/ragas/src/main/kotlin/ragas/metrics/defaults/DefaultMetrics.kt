package ragas.metrics.defaults

import ragas.metrics.Metric

/**
 * Returns the default single-turn metric preset.
 *
 * The preset combines answer relevance, context precision, faithfulness, and
 * context recall for a balanced baseline RAG quality evaluation.
 */
fun defaultSingleTurnMetrics(): List<Metric> =
    listOf(
        AnswerRelevancyMetric(allowHeuristicFallback = true),
        ContextPrecisionMetric(),
        FaithfulnessMetric(allowHeuristicFallback = true),
        ContextRecallMetric(),
    )
