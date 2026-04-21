package ragas.metrics.collections

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.model.SingleTurnSample
import kotlin.math.pow

/**
 * Implements [QuotedSpansAlignmentMetric].
 *
 * @property casefold Whether matching ignores case.
 * @property minSpanWords Minimum span length in words.
 */
class QuotedSpansAlignmentMetric(
    name: String = "quoted_spans_alignment",
    private val casefold: Boolean = true,
    private val minSpanWords: Int = 3,
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response", "retrieved_contexts")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric {
    /**
     * Executes singleTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val response = sample.response.orEmpty()
        val contexts = sample.retrievedContexts.orEmpty()

        val spans = extractQuotedSpans(response, minSpanWords)
        if (spans.isEmpty()) {
            return 1.0
        }

        val normalizedSources = normalizeForMatch(contexts.joinToString(" "), casefold)
        val matched =
            spans.count { span ->
                val normalizedSpan = normalizeForMatch(span, casefold)
                normalizedSpan.isNotBlank() && normalizedSpan in normalizedSources
            }

        return matched.toDouble() / spans.size.toDouble()
    }

    private fun extractQuotedSpans(
        answer: String,
        minLenWords: Int,
    ): List<String> =
        QUOTE_REGEX
            .findAll(answer)
            .mapNotNull { match ->
                val span =
                    match.groupValues
                        .getOrNull(1)
                        .orEmpty()
                        .trim()
                if (span.split(WHITESPACE_REGEX).count { it.isNotBlank() } >= minLenWords) span else null
            }.toList()

    private fun normalizeForMatch(
        text: String,
        doCasefold: Boolean,
    ): String {
        val normalizedWhitespace = text.replace(WHITESPACE_REGEX, " ").trim()
        return if (doCasefold) normalizedWhitespace.lowercase() else normalizedWhitespace
    }

    private companion object {
        val QUOTE_REGEX =
            Regex(
                "(?<![\\p{L}\\p{N}])[\"\\u201c\\u201d\\u201e\\u201f'\\u2018\\u2019`\\u00b4]" +
                    "(.*?)" +
                    "[\"\\u201c\\u201d\\u201e\\u201f'\\u2018\\u2019`\\u00b4](?![\\p{L}\\p{N}])",
            )
        val WHITESPACE_REGEX = Regex("\\s+")
    }
}

/**
 * Computes chrF-style character n-gram similarity between reference and response text.
 *
 * @property charOrder Maximum character n-gram order to include.
 * @property beta F-score beta parameter.
 */
class ChrfScoreMetric(
    name: String = "chrf_score",
    private val charOrder: Int = 6,
    private val beta: Double = 2.0,
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("reference", "response")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric {
    init {
        require(charOrder > 0) { "charOrder must be positive." }
        require(beta > 0.0 && beta.isFinite()) { "beta must be positive and finite." }
    }

    /**
     * Computes the average F-beta score across character n-gram orders from 1 to [charOrder].
     *
     * Returns `0.0` when either side is blank after normalization; otherwise returns
     * a continuous score in [0.0, 1.0].
     *
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val reference = sample.reference.orEmpty()
        val response = sample.response.orEmpty()
        if (reference.isBlank() || response.isBlank()) {
            return 0.0
        }

        val normalizedRef = normalizeText(reference)
        val normalizedResp = normalizeText(response)
        if (normalizedRef.isBlank() || normalizedResp.isBlank()) {
            return 0.0
        }

        val maxComparableOrder = minOf(charOrder, normalizedRef.length, normalizedResp.length)
        if (maxComparableOrder == 0) {
            return 0.0
        }

        val fScores =
            (1..maxComparableOrder).map { n ->
                val refNgrams = charNgramCounts(normalizedRef, n)
                val respNgrams = charNgramCounts(normalizedResp, n)

                val overlap =
                    respNgrams.entries.sumOf { (ngram, countResp) ->
                        minOf(countResp, refNgrams[ngram] ?: 0).toDouble()
                    }
                val precision = overlap / respNgrams.values.sum().toDouble()
                val recall = overlap / refNgrams.values.sum().toDouble()
                fBeta(precision, recall, beta)
            }

        return clamp01(fScores.average())
    }

    private fun normalizeText(text: String): String =
        text
            .lowercase()
            .replace(Regex("\\s+"), " ")
            .trim()

    private fun charNgramCounts(
        text: String,
        n: Int,
    ): Map<String, Int> {
        if (text.length < n) {
            return emptyMap()
        }
        val counts = linkedMapOf<String, Int>()
        for (i in 0..text.length - n) {
            val ngram = text.substring(i, i + n)
            counts[ngram] = (counts[ngram] ?: 0) + 1
        }
        return counts
    }

    private fun fBeta(
        precision: Double,
        recall: Double,
        beta: Double,
    ): Double {
        if (precision + recall == 0.0) {
            return 0.0
        }
        val betaSq = beta.pow(2.0)
        return ((1.0 + betaSq) * precision * recall) / ((betaSq * precision) + recall)
    }
}
