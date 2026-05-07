package shared.rag.unified

import pathrag.PathRAG
import pathrag.base.QueryParam as PathQueryParam

/**
 * Unified adapter for PathRAG.
 */
class PathRagUnifiedAdapter(
    private val delegate: PathRAG,
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
    ): UnifiedResponse = kotlinx.coroutines.runBlocking { aquery(query, param) }

    override suspend fun aquery(
        query: String,
        param: UnifiedQuery,
    ): UnifiedResponse {
        val unsupported = collectUnsupported(param, capabilities)
        val mode = modeFor(param.mode)

        if (!param.includeAnswer && !param.includeContext) {
            return UnifiedResponse(
                metadata =
                    buildMetadata(
                        ragId = RagId.PATH_RAG,
                        modeUsed = mode,
                        unsupported = unsupported,
                    ),
            )
        }

        val pathParamBase = buildQueryParam(param, mode)

        val contextText =
            if (param.includeContext) {
                delegate.aquery(
                    query,
                    pathParamBase.copy(onlyNeedContext = true),
                )
            } else {
                null
            }

        val answerText =
            if (param.includeAnswer) {
                delegate.aquery(
                    query,
                    pathParamBase.copy(onlyNeedContext = false),
                )
            } else {
                null
            }

        val contextItems =
            contextText
                ?.trim()
                ?.takeIf { it.isNotBlank() }
                ?.let { listOf(ContextItem(id = "0", text = it)) }
                ?: emptyList()

        return UnifiedResponse(
            answer = answerText?.takeIf { it.isNotBlank() },
            context = contextItems,
            references = if (param.includeReferences) contextItems.map { ReferenceItem(id = it.id, snippet = it.text) } else emptyList(),
            metadata =
                buildMetadata(
                    ragId = RagId.PATH_RAG,
                    modeUsed = mode,
                    unsupported = unsupported,
                    extra = mapOf("topK" to param.coercedTopK()),
                ),
            raw =
                mapOf(
                    "answer" to answerText,
                    "context" to contextText,
                ),
        )
    }

    private fun modeFor(mode: UnifiedMode): String =
        when (mode) {
            UnifiedMode.LOCAL -> "local"
            UnifiedMode.GLOBAL -> "global"
            else -> "hybrid"
        }

    private fun buildQueryParam(
        query: UnifiedQuery,
        mode: String,
    ): PathQueryParam {
        val defaultMax = query.maxContextTokens ?: 4000
        return PathQueryParam(
            mode = mode,
            onlyNeedContext = false,
            onlyNeedPrompt = false,
            responseType = query.stringExtra("responseType", "Multiple Paragraphs") ?: "Multiple Paragraphs",
            stream = false,
            topK = query.coercedTopK(),
            maxTokenForTextUnit = query.intExtra("maxTokenForTextUnit", defaultMax),
            maxTokenForGlobalContext = query.intExtra("maxTokenForGlobalContext", defaultMax),
            maxTokenForLocalContext = query.intExtra("maxTokenForLocalContext", defaultMax),
        )
    }

    companion object {
        val CAPABILITIES =
            RagCapabilities(
                supportedModes = setOf(UnifiedMode.LOCAL, UnifiedMode.GLOBAL, UnifiedMode.HYBRID),
                supportsStreaming = false,
                supportsGraphPaths = false,
                supportsFollowUpQueries = false,
                supportsReferences = true,
            )
    }
}
