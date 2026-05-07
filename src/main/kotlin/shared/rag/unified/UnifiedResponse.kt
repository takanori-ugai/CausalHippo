package shared.rag.unified

/**
 * Unified query response shape for all RAG backends.
 */
data class UnifiedResponse(
    val answer: String? = null,
    val context: List<ContextItem> = emptyList(),
    val references: List<ReferenceItem> = emptyList(),
    val graphPaths: List<List<String>> = emptyList(),
    val followUpQueries: List<String> = emptyList(),
    val metadata: Map<String, Any?> = emptyMap(),
    val raw: Any? = null,
)

/**
 * Structured context row exposed to callers.
 */
data class ContextItem(
    val id: String? = null,
    val text: String,
    val score: Double? = null,
    val metadata: Map<String, Any?> = emptyMap(),
)

/**
 * Structured reference row exposed to callers.
 */
data class ReferenceItem(
    val id: String? = null,
    val title: String? = null,
    val url: String? = null,
    val snippet: String? = null,
    val metadata: Map<String, Any?> = emptyMap(),
)
