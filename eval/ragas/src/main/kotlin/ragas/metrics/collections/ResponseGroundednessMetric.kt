package ragas.metrics.collections

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.metrics.tokenSet
import ragas.model.SingleTurnSample

/**
 * Scores how well the model response is grounded in the retrieved contexts.
 *
 * Returns a continuous score in [0.0, 1.0], where higher is better groundedness.
 */
class ResponseGroundednessMetric :
    BaseMetric(
        name = "response_groundedness",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response", "retrieved_contexts")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric {
    /**
     * Computes groundedness for a single-turn sample by comparing response support
     * against retrieved contexts, with penalties for unsupported entities and numbers.
     *
     * @param sample Evaluation sample to score.
     * @return Groundedness score in [0.0, 1.0].
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val response = sample.response.orEmpty().trim()
        val contexts = sample.retrievedContexts.orEmpty().filter { it.isNotBlank() }
        if (response.isBlank() || contexts.isEmpty()) {
            return 0.0
        }

        val mergedContext = contexts.joinToString("\n").trim()
        if (mergedContext.isBlank()) {
            return 0.0
        }

        val normalizedResponse = normalizeForContainment(response)
        val normalizedContext = normalizeForContainment(mergedContext)
        val phraseMatch = " $normalizedResponse " in " $normalizedContext "
        if (normalizedResponse == normalizedContext || phraseMatch) {
            return 1.0
        }

        val contextTokenSet = tokenSet(mergedContext)
        val responseTokens =
            tokenSet(response)
                .filter { token -> token !in STOP_WORDS && token.length > 2 }
                .toSet()

        if (responseTokens.isEmpty()) {
            return 0.0
        }

        val supported = responseTokens.intersect(contextTokenSet).size.toDouble()
        var score = supported / responseTokens.size.toDouble()

        val unsupportedCapitalized =
            extractCapitalizedTokens(response)
                .filter { token -> token !in STOP_WORDS && token.length > 2 }
                .count { token -> token !in contextTokenSet }
        if (unsupportedCapitalized > 0) {
            score *= (1.0 - (0.45 * unsupportedCapitalized.coerceAtMost(2)))
        }

        val contextNumbers = extractNumericTokens(mergedContext)
        val unsupportedNumeric =
            extractNumericTokens(response)
                .count { token -> token !in contextNumbers }
        if (unsupportedNumeric > 0) {
            score *= (1.0 - (0.2 * unsupportedNumeric.coerceAtMost(2)))
        }

        return clamp01(score)
    }

    private fun normalizeForContainment(text: String): String =
        text
            .lowercase()
            .replace(Regex("\\s+"), " ")
            .trim()

    private fun extractCapitalizedTokens(text: String): Set<String> =
        CAPITALIZED_TOKEN_REGEX
            .findAll(text)
            .map { match -> match.value.lowercase() }
            .toSet()

    private fun extractNumericTokens(text: String): Set<String> =
        NUMERIC_TOKEN_REGEX
            .findAll(text)
            .map { match -> match.value.lowercase() }
            .toSet()

    private companion object {
        val CAPITALIZED_TOKEN_REGEX = Regex("\\b[A-Z][A-Za-z0-9-]*\\b")
        val NUMERIC_TOKEN_REGEX = Regex("\\b\\d+[A-Za-z]*\\b")

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
                "has",
                "in",
                "is",
                "it",
                "its",
                "of",
                "on",
                "or",
                "that",
                "the",
                "to",
                "was",
                "were",
                "with",
            )
    }
}
