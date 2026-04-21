package ragas.metrics.collections

internal fun String.looksLikeImageContext(): Boolean {
    val value = trim().lowercase()
    if (value.startsWith("data:image/") && value.contains(";base64,")) {
        return true
    }
    if (value.startsWith("http://") || value.startsWith("https://")) {
        return IMAGE_EXTENSIONS.any { ext -> value.substringBefore('?').endsWith(ext) }
    }
    return IMAGE_EXTENSIONS.any { ext -> value.endsWith(ext) }
}

internal fun multimodalMeaningfulTokens(text: String): Set<String> =
    Regex("[\\p{L}\\p{M}\\p{N}]+")
        .findAll(text.lowercase())
        .map { it.value }
        .filter { it.length > 2 && it !in MULTIMODAL_STOP_WORDS }
        .toSet()

internal fun tokenOverlapF1(
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

private val IMAGE_EXTENSIONS =
    setOf(
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
    )

private val MULTIMODAL_STOP_WORDS =
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
