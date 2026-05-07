package shared.rag.unified

/**
 * Unified query request shape for all RAG backends.
 */
data class UnifiedQuery(
    // Legacy field kept for compatibility with earlier call sites.
    // Unified adapters use the first argument of CommonRag.query(...) as the effective query text.
    val text: String = "",
    val mode: UnifiedMode = UnifiedMode.HYBRID,
    val topK: Int = 5,
    val includeAnswer: Boolean = true,
    val includeContext: Boolean = true,
    val includeReferences: Boolean = true,
    val includeGraphPaths: Boolean = false,
    val includeFollowUps: Boolean = false,
    val streaming: Boolean = false,
    val maxContextTokens: Int? = null,
    val conversationHistory: List<String> = emptyList(),
    val extras: Map<String, Any?> = emptyMap(),
)
