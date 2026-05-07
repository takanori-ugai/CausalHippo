package shared.rag.unified

import hipporag.HippoRAG
import hipporag.QueryParam as HippoQueryParam

/**
 * Unified adapter for HippoRAG.
 */
class HippoRagUnifiedAdapter(
    private val delegate: HippoRAG,
) : UnifiedRag {
    val capabilities: RagCapabilities = CAPABILITIES

    override fun upsert(data: String) = delegate.upsert(data)

    override fun upsert(data: Collection<String>) = delegate.upsert(data)

    override suspend fun aupsert(data: String) = delegate.aupsert(data)

    override suspend fun aupsert(data: Collection<String>) = delegate.aupsert(data)

    override fun drop() = delegate.drop()

    override suspend fun adrop() = delegate.adrop()

    override fun saveGraph(path: String) = delegate.saveGraph(path)

    override suspend fun asaveGraph(path: String) = delegate.asaveGraph(path)

    override fun loadGraph(path: String) = delegate.loadGraph(path)

    override suspend fun aloadGraph(path: String) = delegate.aloadGraph(path)

    override fun inspectGraph(): Map<String, Any?> = normalizeGraphInspection(delegate.inspectGraph())

    override suspend fun ainspectGraph(): Map<String, Any?> = normalizeGraphInspection(delegate.ainspectGraph())

    override fun query(
        query: String,
        param: UnifiedQuery,
    ): UnifiedResponse =
        delegate
            .query(
                query,
                buildParam(param),
            ).toUnifiedResponse(param, modeFor(param.mode), collectUnsupported(param, capabilities))

    override suspend fun aquery(
        query: String,
        param: UnifiedQuery,
    ): UnifiedResponse {
        if (!param.includeAnswer && !param.includeContext) {
            return UnifiedResponse(
                metadata =
                    buildMetadata(
                        ragId = RagId.HIPPO_RAG,
                        modeUsed = modeFor(param.mode),
                        unsupported = collectUnsupported(param, capabilities),
                    ),
            )
        }

        val result = delegate.aquery(query, buildParam(param))
        return result.toUnifiedResponse(param, modeFor(param.mode), collectUnsupported(param, capabilities))
    }

    private fun buildParam(query: UnifiedQuery): HippoQueryParam =
        HippoQueryParam(
            mode = modeFor(query.mode),
            topK = query.coercedTopK(),
            includeAnswer = query.includeAnswer,
        )

    private fun modeFor(mode: UnifiedMode): String =
        when (mode) {
            UnifiedMode.DPR -> "dpr"
            UnifiedMode.GRAPH -> "graph"
            else -> "graph"
        }

    private fun hipporag.utils.QuerySolution.toUnifiedResponse(
        query: UnifiedQuery,
        modeUsed: String,
        unsupported: List<String>,
    ): UnifiedResponse {
        val selectedDocs = if (query.includeContext) docs else emptyList()
        return UnifiedResponse(
            answer = answer?.takeIf { query.includeAnswer && it.isNotBlank() },
            context = textContextItems(selectedDocs),
            references = if (query.includeReferences) textReferences(selectedDocs) else emptyList(),
            metadata =
                buildMetadata(
                    ragId = RagId.HIPPO_RAG,
                    modeUsed = modeUsed,
                    unsupported = unsupported,
                    extra = mapOf("topK" to query.coercedTopK()),
                ),
            raw = this,
        )
    }

    companion object {
        val CAPABILITIES =
            RagCapabilities(
                supportedModes = setOf(UnifiedMode.HYBRID, UnifiedMode.GRAPH, UnifiedMode.DPR),
                supportsStreaming = false,
                supportsGraphPaths = false,
                supportsFollowUpQueries = false,
                supportsReferences = true,
            )
    }
}
