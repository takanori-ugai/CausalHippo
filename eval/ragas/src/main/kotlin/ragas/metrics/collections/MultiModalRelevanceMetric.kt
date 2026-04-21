package ragas.metrics.collections

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.model.SingleTurnSample

/**
 * Evaluates whether a response is relevant to the user input and multimodal context.
 *
 * Uses weighted token-overlap signals from question relevance and textual context support,
 * with an additional image-anchor heuristic when image-like contexts are present.
 */
class MultiModalRelevanceMetric(
    name: String = "multi_modal_relevance",
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("user_input", "response", "retrieved_contexts")),
        outputType = MetricOutputType.BINARY,
    ),
    SingleTurnMetric {
    /**
     * Computes a binary relevance score from weighted overlap/anchor heuristics.
     *
     * Returns `1.0` when the blended relevance score passes the threshold, else `0.0`.
     * Throws if required fields (`user_input`, `response`, `retrieved_contexts`) are missing.
     *
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val userInput = sample.userInput.orEmpty().trim()
        val response = sample.response.orEmpty().trim()
        val contexts = sample.retrievedContexts.orEmpty().filter { it.isNotBlank() }
        require(userInput.isNotBlank()) { "user_input is missing. Please provide a question to evaluate against." }
        require(response.isNotBlank()) { "response is missing. Please provide a response to evaluate." }
        require(contexts.isNotEmpty()) { "retrieved_contexts is missing. Please provide contexts to check against." }

        val questionTokens = multimodalMeaningfulTokens(userInput)
        val responseTokens = multimodalMeaningfulTokens(response)
        if (responseTokens.isEmpty()) {
            return 0.0
        }

        val questionRelevance = tokenOverlapF1(responseTokens, questionTokens)
        val textContexts = contexts.filterNot { it.looksLikeImageContext() }
        val imageContexts = contexts.filter { it.looksLikeImageContext() }
        val textSupport =
            textContexts.maxOfOrNull { context ->
                tokenOverlapF1(responseTokens, multimodalMeaningfulTokens(context))
            } ?: 0.0

        if (textContexts.isEmpty() && imageContexts.isNotEmpty()) {
            val imageQuestion =
                userInput.lowercase().contains("image") || userInput.lowercase().contains("photo") ||
                    userInput.lowercase().contains("picture")
            val imageResponse =
                response.lowercase().contains("image") || response.lowercase().contains("photo") || response.lowercase().contains("picture")
            return if (imageQuestion && imageResponse && questionRelevance >= IMAGE_ONLY_QUESTION_RELEVANCE_MIN) 1.0 else 0.0
        }

        val imageAnchor =
            if (imageContexts.isEmpty()) {
                0.0
            } else {
                if (response.lowercase().contains("image") || response.lowercase().contains("photo") ||
                    response.lowercase().contains("picture")
                ) {
                    IMAGE_ANCHOR_WITH_VISUAL_CUE
                } else {
                    IMAGE_ANCHOR_WITHOUT_VISUAL_CUE
                }
            }

        val score =
            (QUESTION_RELEVANCE_WEIGHT * questionRelevance) +
                (TEXT_SUPPORT_WEIGHT * textSupport) +
                (IMAGE_ANCHOR_WEIGHT * imageAnchor)
        return if (score >= RELEVANCE_PASS_THRESHOLD) 1.0 else 0.0
    }

    private companion object {
        // Heuristic blend tuned to prioritize direct question relevance and textual grounding,
        // while still giving a small boost when image context is explicitly acknowledged.
        const val QUESTION_RELEVANCE_WEIGHT = 0.55
        const val TEXT_SUPPORT_WEIGHT = 0.35
        const val IMAGE_ANCHOR_WEIGHT = 0.10
        const val RELEVANCE_PASS_THRESHOLD = 0.28

        const val IMAGE_ONLY_QUESTION_RELEVANCE_MIN = 0.2
        const val IMAGE_ANCHOR_WITH_VISUAL_CUE = 0.2
        const val IMAGE_ANCHOR_WITHOUT_VISUAL_CUE = 0.1
    }
}
