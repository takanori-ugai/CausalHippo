package causalrag.retriever

import hipporag.HippoRag

/**
 * Retrieval mode exposed by [HippoRagSemanticRetrieverAdapter].
 */
enum class HippoRagSemanticMode {
    GRAPH,
    DPR,
}

/**
 * Adapts HippoRAG retrieval APIs to the causal hybrid retrieval contract.
 *
 * @param hippoRag Indexed HippoRAG instance to query.
 * @param mode Whether to use graph-aware HippoRAG retrieval or dense-only retrieval.
 * @param fallbackToDpr When enabled, graph retrieval falls back to dense retrieval if it returns no passages.
 */
class HippoRagSemanticRetrieverAdapter(
    private val hippoRag: HippoRag,
    private val mode: HippoRagSemanticMode = HippoRagSemanticMode.GRAPH,
    private val fallbackToDpr: Boolean = true,
) : SemanticRetriever {
    override fun searchWithScores(
        query: String,
        topK: Int,
    ): List<Pair<String, Double>> {
        val primary =
            when (mode) {
                HippoRagSemanticMode.GRAPH -> {
                    hippoRag.retrieve(listOf(query), numToRetrieve = topK).firstOrNull()
                }

                HippoRagSemanticMode.DPR -> {
                    hippoRag.retrieveDpr(listOf(query), numToRetrieve = topK).firstOrNull()
                }
            }

        val primaryPairs = primary.toPassageScores()
        if (primaryPairs.isNotEmpty() || !fallbackToDpr || mode == HippoRagSemanticMode.DPR) {
            return primaryPairs
        }

        return hippoRag
            .retrieveDpr(listOf(query), numToRetrieve = topK)
            .firstOrNull()
            .toPassageScores()
    }

    private fun Pair<List<hipporag.utils.QuerySolution>, Map<String, Double>?>.firstOrNull(): hipporag.utils.QuerySolution? =
        first.firstOrNull()

    private fun hipporag.utils.QuerySolution?.toPassageScores(): List<Pair<String, Double>> {
        if (this == null || docs.isEmpty()) return emptyList()
        val scores = docScores?.toList().orEmpty()
        return docs.mapIndexed { index, passage ->
            passage to scores.getOrElse(index) { 0.0 }
        }
    }
}
