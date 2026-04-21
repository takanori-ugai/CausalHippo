package hipporag.utils

import io.github.oshai.kotlinlogging.KotlinLogging
import java.security.MessageDigest

private val logger = KotlinLogging.logger {}

/**
 * Computes an MD5 hash for [content] with an optional [prefix].
 */
fun computeMdHashId(
    content: String,
    prefix: String = "",
): String {
    val md = MessageDigest.getInstance("MD5")
    val digest = md.digest(content.toByteArray())
    val hex = digest.joinToString("") { "%02x".format(it) }
    return prefix + hex
}

/**
 * Normalizes text or nested lists into lowercase alphanumeric tokens.
 */
fun textProcessing(text: Any): List<String> =
    when (text) {
        is List<*> -> {
            text.map { t -> textProcessing(t ?: "").joinToString(" ") }
        }

        is String -> {
            listOf(text.lowercase().replace(Regex("[^A-Za-z0-9 ]"), " ").trim())
        }

        else -> {
            listOf(
                text
                    .toString()
                    .lowercase()
                    .replace(Regex("[^A-Za-z0-9 ]"), " ")
                    .trim(),
            )
        }
    }

/**
 * Applies min-max normalization to [values].
 */
fun minMaxNormalize(values: DoubleArray): DoubleArray {
    if (values.isEmpty()) return values
    val minVal = values.minOrNull() ?: 0.0
    val maxVal = values.maxOrNull() ?: 0.0
    val range = maxVal - minVal
    if (range == 0.0) {
        return DoubleArray(values.size) { 1.0 }
    }
    return DoubleArray(values.size) { idx -> (values[idx] - minVal) / range }
}

/**
 * Extracts unique entity strings and per-chunk entity lists from triples.
 */
fun extractEntityNodes(chunkTriples: List<List<List<String>>>): Pair<List<String>, List<List<String>>> {
    val chunkTripleEntities = mutableListOf<List<String>>()
    for (triples in chunkTriples) {
        val tripleEntities = mutableSetOf<String>()
        for (t in triples) {
            if (t.size == 3) {
                tripleEntities.add(t[0])
                tripleEntities.add(t[2])
            } else {
                logger.warn { "During graph construction, invalid triple is found: $t" }
            }
        }
        chunkTripleEntities.add(tripleEntities.toList())
    }
    val graphNodes = chunkTripleEntities.flatten().distinct()
    return graphNodes to chunkTripleEntities
}

/**
 * Flattens nested triples into a unique list of facts.
 */
fun flattenFacts(chunkTriples: List<List<List<String>>>): List<List<String>> {
    val graphTriples = mutableSetOf<List<String>>()
    for (triples in chunkTriples) {
        graphTriples.addAll(triples.map { it.toList() })
    }
    return graphTriples.toList()
}

/**
 * Converts OpenIE docs into NER and triple maps keyed by chunk ID.
 */
fun reformatOpenieResults(
    corpusOpenieResults: List<OpenieDoc>,
): Pair<MutableMap<String, NerRawOutput>, MutableMap<String, TripleRawOutput>> {
    val nerOutput = mutableMapOf<String, NerRawOutput>()
    val tripleOutput = mutableMapOf<String, TripleRawOutput>()

    for (doc in corpusOpenieResults) {
        nerOutput[doc.idx] =
            NerRawOutput(
                chunkId = doc.idx,
                response = null,
                metadata = emptyMap(),
                uniqueEntities = doc.extractedEntities.distinct(),
            )
        tripleOutput[doc.idx] =
            TripleRawOutput(
                chunkId = doc.idx,
                response = null,
                metadata = emptyMap(),
                triples = filterInvalidTriples(doc.extractedTriples),
            )
    }

    return nerOutput to tripleOutput
}

/**
 * Returns true if all values in [data] have the same length.
 */
fun allValuesOfSameLength(data: Map<*, *>): Boolean {
    val values = data.values.iterator()
    if (!values.hasNext()) return true
    val first = values.next()
    val firstLength = lengthOf(first)
    if (firstLength == null) return false
    return values.asSequence().all { lengthOf(it) == firstLength }
}

private fun lengthOf(value: Any?): Int? =
    when (value) {
        is Collection<*> -> value.size
        is Array<*> -> value.size
        is String -> value.length
        else -> null
    }

/**
 * Parses common truthy/falsey strings into a boolean.
 */
fun stringToBool(value: Any?): Boolean {
    if (value is Boolean) return value
    val text = value?.toString()?.lowercase() ?: ""
    return when (text) {
        "yes", "true", "t", "y", "1" -> true

        "no", "false", "f", "n", "0" -> false

        else -> throw IllegalArgumentException(
            "Truthy value expected: got $value but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive).",
        )
    }
}

/**
 * Parses a stringified triple into a list of elements.
 */
fun parseTripleString(value: String): List<String> {
    val cleaned =
        value
            .trim()
            .removePrefix("(")
            .removeSuffix(")")
            .removePrefix("[")
            .removeSuffix("]")
    return cleaned.split(",").map { it.trim().trim('\'', '"') }.filter { it.isNotEmpty() }
}
