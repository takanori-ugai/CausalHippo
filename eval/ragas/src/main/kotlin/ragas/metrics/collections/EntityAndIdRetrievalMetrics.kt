package ragas.metrics.collections

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.model.SingleTurnSample

/**
 * Implements [IdBasedContextPrecisionMetric].
 */
class IdBasedContextPrecisionMetric :
    BaseMetric(
        name = "id_based_context_precision",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("retrieved_context_ids", "reference_context_ids")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric {
    /**
     * Executes singleTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val retrieved =
            sample.retrievedContextIds
                .orEmpty()
                .map { id -> id.trim() }
                .filter { it.isNotBlank() }
        val reference =
            sample.referenceContextIds
                .orEmpty()
                .map { id -> id.trim() }
                .filter { it.isNotBlank() }

        if (retrieved.isEmpty() || reference.isEmpty()) {
            return 0.0
        }

        val retrievedSet = retrieved.toSet()
        val referenceSet = reference.toSet()
        val hits = retrievedSet.count { id -> id in referenceSet }

        return clamp01(hits.toDouble() / retrievedSet.size.toDouble())
    }
}

/**
 * Implements [ContextEntityRecallMetric].
 */
class ContextEntityRecallMetric :
    BaseMetric(
        name = "context_entity_recall",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("reference", "retrieved_contexts")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric {
    /**
     * Executes singleTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val reference = sample.reference.orEmpty().trim()
        val contexts = sample.retrievedContexts.orEmpty().filter { context -> context.isNotBlank() }
        if (reference.isBlank() || contexts.isEmpty()) {
            return 0.0
        }

        val referenceEntities = extractEntities(reference)
        if (referenceEntities.isEmpty()) {
            return 0.0
        }

        val mergedContexts = contexts.joinToString("\n")
        val contextEntities = extractEntities(mergedContexts)
        val overlap = referenceEntities.intersect(contextEntities)
        val score = overlap.size.toDouble() / referenceEntities.size.toDouble()

        return clamp01(score)
    }

    private fun extractEntities(text: String): Set<String> {
        val entities = linkedSetOf<String>()

        entities += extractNumericEntities(text)

        ENTITY_PHRASE_REGEX.findAll(text).forEach { match ->
            val value = normalizeEntity(match.value)
            if (value.length >= 3 && value !in STOP_ENTITY_WORDS) {
                entities += value
                value.split(" ").forEach { token ->
                    if (token.length >= 4 && token !in STOP_ENTITY_WORDS) {
                        entities += token
                    }
                }
            }
        }

        return entities
    }

    private fun extractNumericEntities(text: String): Set<String> {
        val entities = linkedSetOf<String>()
        YEAR_REGEX.findAll(text).forEach { match -> entities += normalizeEntity(match.value) }
        ORDINAL_NUMBER_REGEX.findAll(text).forEach { match -> entities += normalizeEntity(match.value) }
        THOUSANDS_NUMBER_REGEX.findAll(text).forEach { match -> entities += normalizeEntity(match.value) }
        return entities
    }

    private fun normalizeEntity(entity: String): String =
        entity
            .trim()
            .lowercase()
            .replace(NON_ENTITY_CHARS_REGEX, "")
            .replace(WHITESPACE_REGEX, " ")

    private companion object {
        val YEAR_REGEX = Regex("\\b\\d{4}\\b")
        val ORDINAL_NUMBER_REGEX = Regex("\\b\\d+(?:st|nd|rd|th)\\b")
        val THOUSANDS_NUMBER_REGEX = Regex("\\b\\d{1,3}(?:,\\d{3})+\\b")
        val ENTITY_PHRASE_REGEX = Regex("\\b(?:[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)\\b")
        val NON_ENTITY_CHARS_REGEX = Regex("[^\\p{L}\\p{N}, ]")
        val WHITESPACE_REGEX = Regex("\\s+")

        val STOP_ENTITY_WORDS =
            setOf(
                "the",
                "it",
                "its",
                "he",
                "she",
                "they",
                "this",
                "that",
            )
    }
}
