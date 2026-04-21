package ragas.metrics.collections

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.metrics.tokenSet
import ragas.model.SingleTurnSample

abstract class BaseContextPrecisionMetric(
    name: String,
    requiredColumns: Set<String>,
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to requiredColumns),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric {
    protected abstract fun answerText(sample: SingleTurnSample): String

    protected open val matchThreshold: Double = 0.35

    /**
     * Executes singleTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val userInput = sample.userInput.orEmpty().trim()
        val answer = answerText(sample).trim()
        val contexts = sample.retrievedContexts.orEmpty().filter { it.isNotBlank() }
        if (userInput.isBlank() || answer.isBlank() || contexts.isEmpty()) {
            return 0.0
        }

        val queryTokens = meaningfulTokens(userInput)
        val answerTokens = meaningfulTokens(answer)
        if (queryTokens.isEmpty() || answerTokens.isEmpty()) {
            return 0.0
        }
        val answerEntities = entityTokens(answer)

        val verdicts =
            contexts.map { context ->
                val contextTokens = meaningfulTokens(context)
                if (contextTokens.isEmpty()) {
                    return@map 0
                }

                val queryCoverage = queryTokens.intersect(contextTokens).size.toDouble() / queryTokens.size.toDouble()
                val answerCoverage = answerTokens.intersect(contextTokens).size.toDouble() / answerTokens.size.toDouble()

                val contextEntities = entityTokens(context)
                val entityCoverage =
                    if (answerEntities.isEmpty()) {
                        1.0
                    } else {
                        answerEntities.intersect(contextEntities).size.toDouble() / answerEntities.size.toDouble()
                    }

                val utility = (0.5 * answerCoverage) + (0.3 * entityCoverage) + (0.2 * queryCoverage)
                if (utility >= matchThreshold) 1 else 0
            }

        return clamp01(averagePrecision(verdicts))
    }

    private fun averagePrecision(verdicts: List<Int>): Double {
        if (verdicts.isEmpty()) {
            return 0.0
        }

        var cumulativeHits = 0
        var numerator = 0.0

        verdicts.forEachIndexed { index, verdict ->
            if (verdict == 1) {
                cumulativeHits += 1
                numerator += cumulativeHits.toDouble() / (index + 1).toDouble()
            }
        }

        return numerator / cumulativeHits.coerceAtLeast(1).toDouble()
    }

    private fun meaningfulTokens(text: String): Set<String> =
        tokenSet(text).filter { token -> token.length > 2 && token !in STOP_WORDS }.toSet()

    private fun entityTokens(text: String): Set<String> =
        ENTITY_TOKEN_REGEX
            .findAll(text)
            .map { match -> match.value.lowercase() }
            .filter { token -> token !in STOP_WORDS }
            .toSet()

    private companion object {
        val ENTITY_TOKEN_REGEX = Regex("\\b[A-Z][A-Za-z0-9-]*\\b")

        val STOP_WORDS =
            setOf(
                "a",
                "an",
                "and",
                "are",
                "as",
                "at",
                "be",
                "by",
                "for",
                "from",
                "how",
                "in",
                "is",
                "it",
                "its",
                "of",
                "on",
                "or",
                "the",
                "to",
                "was",
                "were",
                "what",
                "when",
                "where",
                "who",
                "with",
            )
    }
}

open class ContextPrecisionWithReferenceMetric(
    name: String = "context_precision_with_reference",
) : BaseContextPrecisionMetric(
        name = name,
        requiredColumns = setOf("user_input", "retrieved_contexts", "reference"),
    ) {
    /**
     * Executes answerText.
     * @param sample Evaluation sample to score.
     */
    override fun answerText(sample: SingleTurnSample): String = sample.reference.orEmpty()
}

open class ContextPrecisionWithoutReferenceMetric(
    name: String = "context_precision_without_reference",
) : BaseContextPrecisionMetric(
        name = name,
        requiredColumns = setOf("user_input", "retrieved_contexts", "response"),
    ) {
    override val matchThreshold: Double = 0.20

    /**
     * Executes answerText.
     * @param sample Evaluation sample to score.
     */
    override fun answerText(sample: SingleTurnSample): String = sample.response.orEmpty()
}

/**
 * Implements [ContextPrecisionCollectionMetric].
 */
class ContextPrecisionCollectionMetric : ContextPrecisionWithReferenceMetric(name = "context_precision")

/**
 * Implements [ContextUtilizationMetric].
 */
class ContextUtilizationMetric : ContextPrecisionWithoutReferenceMetric(name = "context_utilization")
