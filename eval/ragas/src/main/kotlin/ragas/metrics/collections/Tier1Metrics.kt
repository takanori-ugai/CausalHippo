package ragas.metrics.collections

import ragas.metrics.Metric

/**
 * Returns the Tier-1 retrieval-groundedness metric preset.
 *
 * This bundle targets baseline RAG evaluation: context relevance/precision,
 * response grounding to retrieved context, and entity/id-based retrieval coverage.
 */
fun retrievalGroundednessTier1Metrics(): List<Metric> =
    listOf(
        ContextRelevanceMetric(),
        ResponseGroundednessMetric(),
        ContextPrecisionWithReferenceMetric(),
        ContextPrecisionWithoutReferenceMetric(),
        IdBasedContextPrecisionMetric(),
        ContextEntityRecallMetric(),
    )
