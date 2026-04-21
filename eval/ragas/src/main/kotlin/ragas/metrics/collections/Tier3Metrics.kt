package ragas.metrics.collections

import ragas.metrics.Metric

/**
 * Returns the default Tier-3 answer-quality metric preset.
 */
fun answerQualityTier3Metrics(): List<Metric> =
    listOf(
        AnswerAccuracyMetric(),
        AnswerCorrectnessMetric(),
        FactualCorrectnessMetric(),
        TopicAdherenceMetric(),
        NoiseSensitivityMetric(),
        SummaryScoreMetric(),
        QuotedSpansAlignmentMetric(),
        ChrfScoreMetric(),
        BleuScoreMetric(),
        RougeScoreMetric(),
        SemanticSimilarityMetric(),
    )
