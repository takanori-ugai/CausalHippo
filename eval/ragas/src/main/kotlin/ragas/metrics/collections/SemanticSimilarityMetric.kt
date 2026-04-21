package ragas.metrics.collections

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.metrics.tokenize
import ragas.model.SingleTurnSample
import kotlin.math.sqrt

/**
 * Implements [SemanticSimilarityMetric].
 *
 * @property threshold Similarity threshold.
 */
class SemanticSimilarityMetric(
    name: String = "semantic_similarity",
    private val threshold: Double? = null,
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("reference", "response")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric {
    init {
        require(threshold == null || (threshold.isFinite() && threshold in 0.0..1.0)) {
            "threshold must be a finite value in [0.0, 1.0], got $threshold"
        }
    }

    /**
     * Computes semantic similarity between response and reference text.
     *
     * Returns a continuous similarity score in [0.0, 1.0] when [threshold] is null.
     * When [threshold] is set, returns binary output: `1.0` if score >= threshold, else `0.0`.
     *
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val reference = sample.reference.orEmpty().ifBlank { " " }
        val response = sample.response.orEmpty().ifBlank { " " }

        val score = cosineSimilarity(reference, response)
        if (threshold != null) {
            return if (score >= threshold) 1.0 else 0.0
        }
        return score
    }

    private fun cosineSimilarity(
        left: String,
        right: String,
    ): Double {
        val leftCounts = tokenCounts(left)
        val rightCounts = tokenCounts(right)
        if (leftCounts.isEmpty() || rightCounts.isEmpty()) {
            return 0.0
        }

        val dot =
            leftCounts.entries.sumOf { (token, leftVal) ->
                leftVal * (rightCounts[token] ?: 0.0)
            }
        val leftNorm = sqrt(leftCounts.values.sumOf { it * it })
        val rightNorm = sqrt(rightCounts.values.sumOf { it * it })
        if (leftNorm == 0.0 || rightNorm == 0.0) {
            return 0.0
        }
        return clamp01(dot / (leftNorm * rightNorm))
    }

    private fun tokenCounts(text: String): Map<String, Double> {
        val counts = linkedMapOf<String, Double>()
        tokenize(text)
            .filter { token -> token.length > 1 }
            .forEach { token ->
                counts[token] = (counts[token] ?: 0.0) + 1.0
            }
        return counts
    }
}
