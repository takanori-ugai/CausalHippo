package ragas.metrics

private val tokenPattern = Regex("[\\p{L}\\p{M}\\p{N}]+")

/**
 * Common English stop words used by token-based metrics.
 */
val COMMON_STOP_WORDS =
    setOf(
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "about",
        "your",
        "you",
        "are",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "will",
        "would",
        "could",
        "should",
    )

/**
 * Tokenizes text into normalized alphanumeric tokens.
 *
 * @param text Input text string.
 */
fun tokenize(text: String): List<String> = tokenPattern.findAll(text.lowercase()).map { match -> match.value }.toList()

/**
 * Returns the unique normalized token set for a text.
 *
 * @param text Input text string.
 */
fun tokenSet(text: String): Set<String> = tokenize(text).toSet()

/**
 * Computes Jaccard similarity between two token sets.
 *
 * @param left Left-hand input value.
 * @param right Right-hand input value.
 */
fun jaccardSimilarity(
    left: Set<String>,
    right: Set<String>,
): Double {
    if (left.isEmpty() || right.isEmpty()) {
        return 0.0
    }
    val intersection = left.intersect(right).size.toDouble()
    val union = left.union(right).size.toDouble()
    return intersection / union
}

/**
 * Clamps a numeric value into the inclusive range [0, 1].
 *
 * @param value Value payload.
 */
fun clamp01(value: Double): Double =
    when {
        value < 0.0 -> 0.0
        value > 1.0 -> 1.0
        else -> value
    }
