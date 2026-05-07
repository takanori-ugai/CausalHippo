package shared.rag.unified

import com.microsoft.graphrag.GraphRAG
import com.microsoft.graphrag.QueryParam as GraphQueryParam
import com.microsoft.graphrag.query.QueryResult as GraphQueryResult
import kotlinx.coroutines.runBlocking

/**
 * Unified adapter for GraphRAG.
 */
class GraphRagUnifiedAdapter(
    private val delegate: GraphRAG,
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
        if (!param.includeAnswer && !param.includeContext && !param.includeReferences && !param.includeFollowUps) {
            return UnifiedResponse(
                metadata =
                    buildMetadata(
                        ragId = RagId.GRAPH_RAG,
                        modeUsed = modeFor(param.mode),
                        unsupported = collectUnsupported(param, capabilities),
                    ),
            )
        }

        val unsupported = collectUnsupported(param, capabilities)
        val mode = modeFor(param.mode)
        val result = delegate.aquery(query, buildQueryParam(param, mode))

        return result.toUnifiedResponse(param, mode, unsupported)
    }

    private fun buildQueryParam(
        query: UnifiedQuery,
        mode: String,
    ): GraphQueryParam =
        GraphQueryParam(
            mode = mode,
            responseType = query.stringExtra("responseType", DEFAULT_RESPONSE_TYPE) ?: DEFAULT_RESPONSE_TYPE,
            streaming = query.streaming,
            topK = query.coercedTopK(),
            maxContextTokens = query.intExtra("maxContextTokens", query.maxContextTokens ?: 12000),
            topKEntities = query.intExtra("topKEntities", 5),
            topKRelationships = query.intExtra("topKRelationships", 10),
            communityLevel = (query.extras["communityLevel"] as? Number)?.toInt(),
            driftQuery = query.stringExtra("driftQuery"),
            conversationHistory = query.conversationHistory,
            maxIterations = query.intExtra("maxIterations", 3),
            chatModelName = query.stringExtra("chatModelName"),
            embeddingModelName = query.stringExtra("embeddingModelName"),
        )

    private fun modeFor(mode: UnifiedMode): String =
        when (mode) {
            UnifiedMode.BASIC -> "basic"
            UnifiedMode.LOCAL -> "local"
            UnifiedMode.DRIFT -> "drift"
            else -> "global"
        }

    private fun GraphQueryResult.toUnifiedResponse(
        query: UnifiedQuery,
        modeUsed: String,
        unsupported: List<String>,
    ): UnifiedResponse {
        val contextItems =
            if (query.includeContext) {
                if (context.isNotEmpty()) {
                    context.map { row ->
                        ContextItem(
                            id = row.id,
                            text = row.text,
                            score = row.score,
                        )
                    }
                } else {
                    contextText
                        .trim()
                        .takeIf { it.isNotBlank() }
                        ?.let { listOf(ContextItem(id = "context", text = it)) }
                        ?: emptyList()
                }
            } else {
                emptyList()
            }

        val references =
            if (query.includeReferences) {
                contextRecords.entries.flatMap { (group, rows) ->
                    rows.mapIndexed { index, row ->
                        val snippet = row["content"] ?: row["text"] ?: row["description"] ?: row.toString()
                        ReferenceItem(
                            id = "$group:$index",
                            snippet = snippet,
                            metadata = row + mapOf("group" to group),
                        )
                    }
                }
            } else {
                emptyList()
            }

        return UnifiedResponse(
            answer = answer.takeIf { query.includeAnswer && it.isNotBlank() },
            context = contextItems,
            references = references,
            followUpQueries = if (query.includeFollowUps) followUpQueries else emptyList(),
            metadata =
                buildMetadata(
                    ragId = RagId.GRAPH_RAG,
                    modeUsed = modeUsed,
                    unsupported = unsupported,
                    extra =
                        mapOf(
                            "llmCalls" to llmCalls,
                            "promptTokens" to promptTokens,
                            "outputTokens" to outputTokens,
                            "score" to score,
                        ),
                ),
            raw = this,
        )
    }

    companion object {
        private const val DEFAULT_RESPONSE_TYPE = "JSON response (response, score, follow_up_queries)"

        val CAPABILITIES =
            RagCapabilities(
                supportedModes = setOf(UnifiedMode.BASIC, UnifiedMode.LOCAL, UnifiedMode.GLOBAL, UnifiedMode.HYBRID, UnifiedMode.DRIFT),
                supportsStreaming = true,
                supportsGraphPaths = false,
                supportsFollowUpQueries = true,
                supportsReferences = true,
            )
    }
}
