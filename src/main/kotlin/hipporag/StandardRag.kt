package hipporag

import hipporag.config.BaseConfig
import hipporag.utils.QuerySolution
import hipporag.utils.RagQaResult

/**
 * Lightweight DPR-only RAG wrapper over [HippoRag].
 *
 * This variant skips graph-based retrieval and relies on dense passage retrieval.
 */
class StandardRag(
    config: BaseConfig? = null,
    saveDir: String? = null,
    llmModelName: String? = null,
    embeddingModelName: String? = null,
    llmBaseUrl: String? = null,
    embeddingBaseUrl: String? = null,
) {
    private val hippoRag =
        HippoRag(
            initialConfig = config,
            saveDir = saveDir,
            llmModelName = llmModelName,
            embeddingModelName = embeddingModelName,
            llmBaseUrl = llmBaseUrl,
            embeddingBaseUrl = embeddingBaseUrl,
        )

    /**
     * Indexes documents for DPR-only retrieval.
     * Note: this delegates to [HippoRag.index], which runs OpenIE + graph construction even though DPR-only
     * retrieval does not use the graph. This keeps the index compatible with graph-based retrieval if needed.
     */
    fun index(docs: List<String>) {
        hippoRag.index(docs)
    }

    /** Deletes documents from the underlying index. */
    fun delete(docsToDelete: List<String>) {
        hippoRag.delete(docsToDelete)
    }

    /**
     * Retrieves passages using DPR only.
     *
     * @return a pair of solutions and optional recall metrics (when [goldDocs] is provided).
     */
    fun retrieve(
        queries: List<String>,
        numToRetrieve: Int? = null,
        goldDocs: List<List<String>>? = null,
    ): Pair<List<QuerySolution>, Map<String, Double>?> =
        hippoRag.retrieveDpr(
            queries = queries,
            numToRetrieve = numToRetrieve,
            goldDocs = goldDocs,
        )

    /**
     * Runs DPR-only retrieval + QA for [queries].
     */
    fun ragQa(
        queries: List<String>,
        goldDocs: List<List<String>>? = null,
        goldAnswers: List<List<String>>? = null,
    ): RagQaResult =
        hippoRag.ragQaDpr(
            queries = queries,
            goldDocs = goldDocs,
            goldAnswers = goldAnswers,
        )

    /**
     * Runs QA over precomputed DPR solutions.
     */
    fun ragQaWithSolutions(
        queries: List<QuerySolution>,
        goldDocs: List<List<String>>? = null,
        goldAnswers: List<List<String>>? = null,
    ): RagQaResult =
        hippoRag.ragQaDprWithSolutions(
            queries = queries,
            goldDocs = goldDocs,
            goldAnswers = goldAnswers,
        )
}
