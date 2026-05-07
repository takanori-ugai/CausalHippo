package shared.rag.unified

import lightrag.core.LightRAG
import lightrag.core.QueryParam as LightQueryParam
import lightrag.core.QueryResult as LightQueryResult
import kotlinx.coroutines.runBlocking

/**
 * Unified adapter for LightRAG.
 */
class LightRagUnifiedAdapter(
    private val delegate: LightRAG,
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
    ): UnifiedResponse = runBlocking { aquery(query, param) }

    override suspend fun aquery(
        query: String,
        param: UnifiedQuery,
    ): UnifiedResponse {
        if (!param.includeAnswer && !param.includeContext) {
            return UnifiedResponse(
                metadata =
                    buildMetadata(
                        ragId = RagId.LIGHT_RAG,
                        modeUsed = modeFor(param.mode),
                        unsupported = collectUnsupported(param, capabilities),
                    ),
            )
        }

        val unsupported = collectUnsupported(param, capabilities)
        val mode = modeFor(param.mode)
        val result = delegate.aquery(query, buildQueryParam(param, mode))

        if (result == null) {
            return UnifiedResponse(
                metadata =
                    buildMetadata(
                        ragId = RagId.LIGHT_RAG,
                        modeUsed = mode,
                        unsupported = unsupported,
                        extra = mapOf("result" to "null"),
                    ),
                raw = null,
            )
        }

        return result.toUnifiedResponse(param, mode, unsupported)
    }

    private fun buildQueryParam(
        query: UnifiedQuery,
        mode: String,
    ): LightQueryParam {
        val defaultHistory = query.conversationHistory.map { text -> mapOf("role" to "user", "content" to text) }
        return LightQueryParam(
            mode = mode,
            onlyNeedContext = !query.includeAnswer && query.includeContext,
            onlyNeedPrompt = false,
            responseType = query.stringExtra("responseType", "Multiple Paragraphs"),
            stream = query.streaming,
            topK = query.coercedTopK(),
            chunkTopK = query.intExtra("chunkTopK", 20),
            maxEntityTokens = query.intExtra("maxEntityTokens", query.maxContextTokens ?: 6000),
            maxRelationTokens = query.intExtra("maxRelationTokens", query.maxContextTokens ?: 8000),
            maxTotalTokens = query.intExtra("maxTotalTokens", query.maxContextTokens ?: 30000),
            hlKeywords = query.listStringExtra("hlKeywords"),
            llKeywords = query.listStringExtra("llKeywords"),
            conversationHistory = defaultHistory,
            userPrompt = query.stringExtra("userPrompt"),
            enableRerank = (query.extras["enableRerank"] as? Boolean) ?: true,
            includeReferences = query.includeReferences,
        )
    }

    private fun modeFor(mode: UnifiedMode): String =
        when (mode) {
            UnifiedMode.LOCAL -> "local"
            UnifiedMode.GLOBAL -> "global"
            UnifiedMode.NAIVE -> "naive"
            UnifiedMode.BYPASS -> "bypass"
            else -> "hybrid"
        }

    private fun LightQueryResult.toUnifiedResponse(
        query: UnifiedQuery,
        modeUsed: String,
        unsupported: List<String>,
    ): UnifiedResponse {
        val contextItems = if (query.includeContext) extractContextItems(this) else emptyList()
        val references = if (query.includeReferences) referenceList.map { it.toReferenceItem() } else emptyList()
        val followUps = if (query.includeFollowUps) extractFollowUpQueries(rawData) else emptyList()

        return UnifiedResponse(
            answer = content?.takeIf { query.includeAnswer && it.isNotBlank() },
            context = contextItems,
            references = references,
            followUpQueries = followUps,
            metadata =
                buildMetadata(
                    ragId = RagId.LIGHT_RAG,
                    modeUsed = modeUsed,
                    unsupported = unsupported,
                    extra = metadata,
                ),
            raw = this,
        )
    }

    private fun extractContextItems(result: LightQueryResult): List<ContextItem> {
        val raw = result.rawData ?: return result.content?.let { textContextItems(listOf(it)) } ?: emptyList()
        val data = asStringMap(raw["data"])

        val contextList = asStringList(data["context"])
        if (contextList.isNotEmpty()) return textContextItems(contextList)

        val contextText = data["context"]?.toString()?.trim().orEmpty()
        if (contextText.isNotBlank()) return textContextItems(listOf(contextText))

        val records = asStringMap(data["context_records"])
        if (records.isNotEmpty()) {
            val rows = mutableListOf<ContextItem>()
            records.forEach { (group, value) ->
                val groupItems = value as? List<*> ?: return@forEach
                groupItems.forEachIndexed { index, row ->
                    val rowMap = asStringMap(row)
                    val text = rowMap["content"]?.toString()?.ifBlank { null }
                        ?: rowMap["text"]?.toString()?.ifBlank { null }
                        ?: rowMap["description"]?.toString()?.ifBlank { null }
                        ?: rowMap.toString()
                    rows +=
                        ContextItem(
                            id = "$group:$index",
                            text = text,
                            metadata = rowMap + mapOf("group" to group),
                        )
                }
            }
            if (rows.isNotEmpty()) return rows
        }

        return result.content?.let { textContextItems(listOf(it)) } ?: emptyList()
    }

    private fun Map<String, String>.toReferenceItem(): ReferenceItem =
        ReferenceItem(
            id = this["id"],
            title = this["title"],
            url = this["url"],
            snippet = this["content"] ?: this["snippet"] ?: this.toString(),
            metadata = this,
        )

    private fun extractFollowUpQueries(rawData: Map<String, Any?>?): List<String> {
        val data = asStringMap(rawData?.get("data"))
        val direct = asStringList(data["follow_up_queries"])
        if (direct.isNotEmpty()) return direct
        return asStringList(rawData?.get("follow_up_queries"))
    }

    companion object {
        val CAPABILITIES =
            RagCapabilities(
                supportedModes =
                    setOf(
                        UnifiedMode.LOCAL,
                        UnifiedMode.GLOBAL,
                        UnifiedMode.HYBRID,
                        UnifiedMode.NAIVE,
                        UnifiedMode.BYPASS,
                    ),
                supportsStreaming = true,
                supportsGraphPaths = false,
                supportsFollowUpQueries = true,
                supportsReferences = true,
            )
    }
}
