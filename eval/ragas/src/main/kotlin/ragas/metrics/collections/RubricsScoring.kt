package ragas.metrics.collections

import ragas.metrics.COMMON_STOP_WORDS
import ragas.metrics.clamp01
import ragas.metrics.tokenize
import kotlin.math.floor

internal val DEFAULT_REFERENCE_FREE_RUBRICS: Map<String, String> =
    mapOf(
        "score1_description" to "The response is entirely incorrect and fails to address any aspect of the user input.",
        "score2_description" to
            "The response contains partial accuracy but includes major errors or " +
            "significant omissions that affect its relevance to the user input.",
        "score3_description" to
            "The response is mostly accurate but lacks clarity, thoroughness, or minor details needed to fully address the user input.",
        "score4_description" to
            "The response is accurate and clear, with only minor omissions or slight inaccuracies in addressing the user input.",
        "score5_description" to
            "The response is completely accurate, clear, and thoroughly addresses the user input without any errors or omissions.",
    )

internal val DEFAULT_WITH_REFERENCE_RUBRICS: Map<String, String> =
    mapOf(
        "score1_description" to
            "The response is entirely incorrect, irrelevant, or does not align with the reference in any meaningful way.",
        "score2_description" to
            "The response partially matches the reference but contains major errors, significant omissions, or irrelevant information.",
        "score3_description" to
            "The response aligns with the reference overall but lacks sufficient detail, clarity, or contains minor inaccuracies.",
        "score4_description" to
            "The response is mostly accurate, aligns closely with the reference, and contains only minor issues or omissions.",
        "score5_description" to
            "The response is fully accurate, completely aligns with the reference, and is clear, thorough, and detailed.",
    )

private val SCORE_KEY_REGEX = Regex("score(\\d+)_description", RegexOption.IGNORE_CASE)
private val NUMBER_TOKEN_REGEX = Regex("\\b\\d+(?:[.,]\\d+)?\\b")
private val NEGATION_REGEX = Regex("\\b(no|not|never|none|without|cannot|can't|won't|isn't|aren't|didn't|doesn't|don't)\\b")

internal fun normalizeRubrics(rubrics: Map<String, String>): Map<Int, String> {
    val normalized =
        rubrics
            .mapNotNull { (key, description) ->
                val match = SCORE_KEY_REGEX.matchEntire(key.trim()) ?: return@mapNotNull null
                val score = match.groupValues[1].toIntOrNull() ?: return@mapNotNull null
                if (description.isBlank()) {
                    null
                } else {
                    score to description
                }
            }.toMap()

    require(normalized.isNotEmpty()) {
        "Rubrics must include at least one score description in the form scoreN_description"
    }
    return normalized
}

internal fun computeRubricScore(
    rubrics: Map<Int, String>,
    userInput: String,
    response: String,
    reference: String?,
    retrievedContexts: List<String>?,
    referenceContexts: List<String>?,
    withReference: Boolean,
): Double {
    val quality =
        computeResponseQuality(
            userInput = userInput,
            response = response,
            reference = reference,
            retrievedContexts = retrievedContexts,
            referenceContexts = referenceContexts,
            withReference = withReference,
        )

    val sortedScores = rubrics.keys.sorted()
    if (sortedScores.size == 1) {
        return sortedScores.first().toDouble()
    }

    val scaled = quality * sortedScores.size
    val index = floor(scaled).toInt().coerceIn(0, sortedScores.size - 1)
    return sortedScores[index].toDouble()
}

private fun computeResponseQuality(
    userInput: String,
    response: String,
    reference: String?,
    retrievedContexts: List<String>?,
    referenceContexts: List<String>?,
    withReference: Boolean,
): Double {
    val responseText = response.trim()
    if (responseText.isBlank()) {
        return 0.0
    }

    val responseTokens = meaningfulTokens(responseText)
    if (responseTokens.isEmpty()) {
        return 0.0
    }

    val questionRelevance = lexicalF1(responseTokens, meaningfulTokens(userInput))
    val contextSupport = contextSupportScore(responseTokens, retrievedContexts.orEmpty() + referenceContexts.orEmpty())
    val clarity = clarityScore(responseTokens)

    val quality =
        if (withReference) {
            val referenceAlignment = referenceAlignmentScore(responseText, reference.orEmpty())
            (WITH_REFERENCE_ALIGNMENT_WEIGHT * referenceAlignment) +
                (WITH_REFERENCE_QUESTION_WEIGHT * questionRelevance) +
                (WITH_REFERENCE_CONTEXT_WEIGHT * contextSupport) +
                (WITH_REFERENCE_CLARITY_WEIGHT * clarity)
        } else {
            (REFERENCE_FREE_QUESTION_WEIGHT * questionRelevance) +
                (REFERENCE_FREE_CONTEXT_WEIGHT * contextSupport) +
                (REFERENCE_FREE_CLARITY_WEIGHT * clarity)
        }

    return clamp01(quality)
}

private fun referenceAlignmentScore(
    response: String,
    reference: String,
): Double {
    val referenceTokens = meaningfulTokens(reference)
    if (referenceTokens.isEmpty()) {
        return 0.0
    }

    val responseTokens = meaningfulTokens(response)
    val lexical = lexicalF1(responseTokens, referenceTokens)

    val responseNumbers = NUMBER_TOKEN_REGEX.findAll(response.lowercase()).map { it.value }.toSet()
    val referenceNumbers = NUMBER_TOKEN_REGEX.findAll(reference.lowercase()).map { it.value }.toSet()
    val numberAlignment =
        when {
            referenceNumbers.isEmpty() -> 1.0
            responseNumbers.isEmpty() -> 0.0
            else -> responseNumbers.intersect(referenceNumbers).size.toDouble() / referenceNumbers.size.toDouble()
        }

    // Coarse contradiction signal: penalize when only one side contains a negation cue.
    // This intentionally does not attempt deep semantic contradiction detection.
    val negationMismatch = NEGATION_REGEX.containsMatchIn(response.lowercase()) xor NEGATION_REGEX.containsMatchIn(reference.lowercase())
    val contradictionPenalty = if (negationMismatch) NEGATION_MISMATCH_PENALTY else 1.0

    return clamp01(
        ((REFERENCE_ALIGNMENT_LEXICAL_WEIGHT * lexical) + (REFERENCE_ALIGNMENT_NUMBER_WEIGHT * numberAlignment)) * contradictionPenalty,
    )
}

private fun contextSupportScore(
    responseTokens: Set<String>,
    contexts: List<String>,
): Double {
    if (contexts.isEmpty()) {
        return 0.5
    }

    val best =
        contexts.maxOfOrNull { context ->
            val contextTokens = meaningfulTokens(context)
            lexicalF1(responseTokens, contextTokens)
        } ?: 0.0

    return best
}

private fun clarityScore(responseTokens: Set<String>): Double {
    val size = responseTokens.size.toDouble()
    // 24 meaningful tokens is treated as a "fully clear" response for this heuristic.
    return clamp01((size / CLARITY_FULL_CREDIT_TOKEN_COUNT).coerceAtLeast(CLARITY_MIN_SCORE))
}

private fun lexicalF1(
    left: Set<String>,
    right: Set<String>,
): Double {
    if (left.isEmpty() || right.isEmpty()) {
        return 0.0
    }

    val overlap = left.intersect(right).size.toDouble()
    val precision = overlap / left.size.toDouble()
    val recall = overlap / right.size.toDouble()
    return if (precision + recall == 0.0) 0.0 else (2.0 * precision * recall) / (precision + recall)
}

private fun meaningfulTokens(text: String): Set<String> =
    tokenize(text)
        .filter { token -> token.length > 2 && token !in COMMON_STOP_WORDS }
        .toSet()

private const val WITH_REFERENCE_ALIGNMENT_WEIGHT = 0.65
private const val WITH_REFERENCE_QUESTION_WEIGHT = 0.20
private const val WITH_REFERENCE_CONTEXT_WEIGHT = 0.10
private const val WITH_REFERENCE_CLARITY_WEIGHT = 0.05

private const val REFERENCE_FREE_QUESTION_WEIGHT = 0.55
private const val REFERENCE_FREE_CONTEXT_WEIGHT = 0.30
private const val REFERENCE_FREE_CLARITY_WEIGHT = 0.15

private const val REFERENCE_ALIGNMENT_LEXICAL_WEIGHT = 0.8
private const val REFERENCE_ALIGNMENT_NUMBER_WEIGHT = 0.2
private const val NEGATION_MISMATCH_PENALTY = 0.85

private const val CLARITY_FULL_CREDIT_TOKEN_COUNT = 24.0
private const val CLARITY_MIN_SCORE = 0.2
