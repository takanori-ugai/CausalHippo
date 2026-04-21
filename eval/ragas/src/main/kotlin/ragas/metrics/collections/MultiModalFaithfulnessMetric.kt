package ragas.metrics.collections

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.model.SingleTurnSample

/**
 * Evaluates whether a response is faithful to multimodal (text and image) context evidence.
 */
class MultiModalFaithfulnessMetric(
    name: String = "multi_modal_faithfulness",
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response", "retrieved_contexts")),
        outputType = MetricOutputType.BINARY,
    ),
    SingleTurnMetric {
    /**
     * Computes a binary faithfulness score using token-overlap support and multimodal heuristics.
     *
     * Combines textual support with penalties for unsupported critical claims/numbers and an
     * image-context compensation heuristic, then thresholds to `1.0` or `0.0`.
     *
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val response = sample.response.orEmpty().trim()
        val contexts = sample.retrievedContexts.orEmpty().filter { it.isNotBlank() }
        require(response.isNotBlank()) { "response is missing. Please provide a response to evaluate." }
        require(contexts.isNotEmpty()) { "retrieved_contexts is missing. Please provide contexts to check against." }

        val responseTokens = multimodalMeaningfulTokens(response)
        if (responseTokens.isEmpty()) {
            return 0.0
        }

        val textContexts = contexts.filterNot { it.looksLikeImageContext() }
        val imageContexts = contexts.filter { it.looksLikeImageContext() }
        if (textContexts.isEmpty() && imageContexts.isNotEmpty()) {
            return if (responseMentionsVisibleProperties(response)) 1.0 else 0.0
        }
        val support =
            textContexts.maxOfOrNull { context ->
                tokenOverlapF1(responseTokens, multimodalMeaningfulTokens(context))
            } ?: 0.0

        val unsupportedNumberPenalty = unsupportedNumberPenalty(response, textContexts)
        val unsupportedCriticalPenalty = unsupportedCriticalTokenPenalty(responseTokens, textContexts)
        val imageCompensation =
            if (imageContexts.isNotEmpty() && support < 0.2 && responseMentionsVisibleProperties(response)) {
                0.15
            } else {
                0.0
            }

        val score = (support + imageCompensation - unsupportedNumberPenalty - unsupportedCriticalPenalty).coerceIn(0.0, 1.0)
        return if (score >= 0.3) 1.0 else 0.0
    }

    private fun unsupportedNumberPenalty(
        response: String,
        textContexts: List<String>,
    ): Double {
        val responseNumbers = NUMBER_PATTERN.findAll(response.lowercase()).map { it.value }.toSet()
        if (responseNumbers.isEmpty()) {
            return 0.0
        }
        val contextNumbers =
            textContexts
                .flatMap { context ->
                    NUMBER_PATTERN.findAll(context.lowercase()).map { it.value }.toList()
                }.toSet()
        return if (contextNumbers.isNotEmpty() && responseNumbers.intersect(contextNumbers).isEmpty()) 0.25 else 0.0
    }

    private fun responseMentionsVisibleProperties(response: String): Boolean {
        val lower = response.lowercase()
        return VISUAL_WORDS.any { word -> lower.contains(word) }
    }

    private fun unsupportedCriticalTokenPenalty(
        responseTokens: Set<String>,
        textContexts: List<String>,
    ): Double {
        val contextTokens = textContexts.flatMap { ctx -> multimodalMeaningfulTokens(ctx) }.toSet()
        if (contextTokens.isEmpty()) {
            return 0.0
        }
        val unsupported = responseTokens.minus(contextTokens).count { token -> token.length >= 5 && token !in GENERIC_TOKENS }
        return when {
            unsupported >= 2 -> 0.4
            unsupported == 1 -> 0.3
            else -> 0.0
        }
    }

    private companion object {
        val NUMBER_PATTERN = Regex("\\b\\d+(?:[.,]\\d+)?\\b")
        val VISUAL_WORDS =
            setOf(
                "image",
                "photo",
                "picture",
                "shows",
                "appears",
                "visible",
                "color",
                "shape",
            )
        val GENERIC_TOKENS =
            setOf(
                "located",
                "called",
                "named",
                "about",
                "there",
                "their",
            )
    }
}
