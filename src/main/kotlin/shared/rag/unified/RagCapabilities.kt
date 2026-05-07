package shared.rag.unified

/**
 * Feature flags for a backend exposed through the unified API.
 */
data class RagCapabilities(
    val supportedModes: Set<UnifiedMode>,
    val supportsStreaming: Boolean = false,
    val supportsGraphPaths: Boolean = false,
    val supportsFollowUpQueries: Boolean = false,
    val supportsReferences: Boolean = false,
)
