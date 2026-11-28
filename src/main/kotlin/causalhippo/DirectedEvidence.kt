package causalhippo

// Match explicit positive causal statements within a clause. Ambiguous wording contributes only node overlap.
internal fun supportsDirectedRelation(
    passage: String,
    cause: String,
    effect: String,
): Boolean {
    val source = phrasePattern(cause)
    val target = phrasePattern(effect)
    val active = """(?:causes?|caused|leads?\s+to|led\s+to|results?\s+in|resulted\s+in|triggers?|triggered)"""
    val passive = """(?:(?:is|are|was|were)\s+caused\s+by|results?\s+from|resulted\s+from|(?:is|are|was|were)\s+due\s+to)"""
    val assertion = Regex("""(?:$source\s+$active\s+$target|$target\s+$passive\s+$source)""", RegexOption.IGNORE_CASE)
    return clause.findAll(passage).any { match ->
        val text = match.value
        !text.endsWith('?') && !uncertainty.containsMatchIn(text) && assertion.containsMatchIn(text)
    }
}

private val clause = Regex("""[^.!?;]+[.!?;]?""")
private val uncertainty = Regex("""\b(?:no|not|never|false|denies?|neither|cannot|may|might|could|whether)\b""", RegexOption.IGNORE_CASE)

private fun phrasePattern(phrase: String): String =
    phrase.trim().split(Regex("\\s+")).joinToString("\\s+", "(?<![\\p{L}\\p{N}_])", "(?![\\p{L}\\p{N}_])", transform = Regex::escape)
