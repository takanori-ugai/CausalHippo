package causalrag.examples

object EvalUtils {
    fun pickBestReferenceForPrediction(
        prediction: String,
        golds: List<String>,
    ): String =
        golds
            .maxByOrNull { candidate ->
                if (normalize(prediction) == normalize(candidate)) {
                    10_000
                } else {
                    tokenOverlapScore(prediction, candidate)
                }
            } ?: ""

    fun bestExactMatch(
        prediction: String,
        golds: List<String>,
    ): Double = golds.maxOfOrNull { if (normalize(prediction) == normalize(it)) 1.0 else 0.0 } ?: 0.0

    private fun tokenOverlapScore(
        prediction: String,
        gold: String,
    ): Int {
        val predTokens = tokenize(normalize(prediction))
        val goldTokens = tokenize(normalize(gold))
        if (predTokens.isEmpty() || goldTokens.isEmpty()) return 0
        val predCounts = predTokens.groupingBy { it }.eachCount()
        val goldCounts = goldTokens.groupingBy { it }.eachCount()
        var overlap = 0
        for ((token, pCount) in predCounts) {
            val gCount = goldCounts[token] ?: 0
            overlap += minOf(pCount, gCount)
        }
        return overlap
    }

    private fun normalize(text: String): String {
        val lowered = text.lowercase()
        val noPunc = lowered.replace(Regex("[^a-z0-9\\s]"), " ")
        val noArticles = noPunc.replace(Regex("\\b(a|an|the)\\b"), " ")
        return noArticles.replace(Regex("\\s+"), " ").trim()
    }

    private fun tokenize(text: String): List<String> =
        if (text.isBlank()) {
            emptyList()
        } else {
            text.split(' ')
        }
}
