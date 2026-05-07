package shared.rag.unified

import causalhippo.CausalHippoQueryParam
import causalhippo.CausalHippoRAG

/**
 * Unified adapter for CausalHippoRAG.
 */
class CausalHippoRagUnifiedAdapter(
    private val delegate: CausalHippoRAG,
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
            ).toUnifiedResponse(param, collectUnsupported(param, capabilities))

    override suspend fun aquery(
        query: String,
        param: UnifiedQuery,
    ): UnifiedResponse {
        if (!param.includeAnswer && !param.includeContext && !param.includeGraphPaths) {
            return UnifiedResponse(
                metadata =
                    buildMetadata(
                        ragId = RagId.CAUSAL_HIPPO_RAG,
                        modeUsed = "causal_hippo",
                        unsupported = collectUnsupported(param, capabilities),
                    ),
            )
        }

        val result = delegate.aquery(query, buildParam(param))
        return result.toUnifiedResponse(param, collectUnsupported(param, capabilities))
    }

    private fun buildParam(query: UnifiedQuery): CausalHippoQueryParam =
        CausalHippoQueryParam(
            topK = query.coercedTopK(),
            maxPaths = query.intExtra("maxPaths", 3).coerceAtLeast(1),
            onlyNeedContext = !query.includeAnswer && query.includeContext,
            onlyNeedCausalPaths = !query.includeAnswer && query.includeGraphPaths,
        )

    private fun causalrag.CausalRagRunResult.toUnifiedResponse(
        query: UnifiedQuery,
        unsupported: List<String>,
    ): UnifiedResponse {
        val selectedContext = if (query.includeContext) context else emptyList()
        val selectedPaths = if (query.includeGraphPaths) causalPaths else emptyList()
        return UnifiedResponse(
            answer = answer.takeIf { query.includeAnswer && it.isNotBlank() },
            context = textContextItems(selectedContext),
            references = if (query.includeReferences) textReferences(selectedContext) else emptyList(),
            graphPaths = selectedPaths,
            metadata =
                buildMetadata(
                    ragId = RagId.CAUSAL_HIPPO_RAG,
                    modeUsed = "causal_hippo",
                    unsupported = unsupported,
                    extra =
                        mapOf(
                            "topK" to query.coercedTopK(),
                            "maxPaths" to query.intExtra("maxPaths", 3).coerceAtLeast(1),
                        ),
                ),
            raw = this,
        )
    }

    companion object {
        val CAPABILITIES =
            RagCapabilities(
                supportedModes = setOf(UnifiedMode.HYBRID, UnifiedMode.CAUSAL),
                supportsStreaming = false,
                supportsGraphPaths = true,
                supportsFollowUpQueries = false,
                supportsReferences = true,
            )
    }
}
