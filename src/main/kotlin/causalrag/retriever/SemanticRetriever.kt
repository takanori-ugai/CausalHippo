package causalrag.retriever

/**
 * Minimal semantic retrieval contract used by hybrid retrieval.
 *
 * Implementations may be plain vector stores, graph-augmented retrievers,
 * or adapters over external RAG systems.
 */
interface SemanticRetriever {
    /**
     * Returns top semantic matches for a query as passage-score pairs.
     *
     * @param query User query.
     * @param topK Maximum number of matches to return.
     * @return Passage-score pairs ordered from best to worst.
     */
    fun searchWithScores(
        query: String,
        topK: Int = 5,
    ): List<Pair<String, Double>>
}
