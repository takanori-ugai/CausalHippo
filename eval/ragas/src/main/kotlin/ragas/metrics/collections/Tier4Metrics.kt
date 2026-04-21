package ragas.metrics.collections

import ragas.metrics.Metric

/**
 * Returns the full Tier-4 advanced metric preset.
 *
 * Includes advanced rubric-based scoring, SQL semantic equivalence, data-comparison
 * scoring, and multimodal relevance/faithfulness metrics.
 */
fun tier4CollectionMetrics(): List<Metric> =
    listOf(
        DomainSpecificRubricsMetric(),
        RubricsScoreWithoutReferenceMetric(),
        RubricsScoreWithReferenceMetric(),
        InstanceSpecificRubricsMetric(),
        SqlSemanticEquivalenceMetric(),
        DataCompyScoreMetric(),
        MultiModalRelevanceMetric(),
        MultiModalFaithfulnessMetric(),
    )

/**
 * Returns the Tier-4 advanced rubrics-focused preset.
 *
 * Includes domain/instance-specific rubric scoring with and without references.
 */
fun advancedRubricsTier4Metrics(): List<Metric> =
    listOf(
        DomainSpecificRubricsMetric(),
        RubricsScoreWithoutReferenceMetric(),
        RubricsScoreWithReferenceMetric(),
        InstanceSpecificRubricsMetric(),
    )
