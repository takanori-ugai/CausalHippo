package shared.rag.unified

import shared.rag.CommonRag
import shared.rag.spi.persistence.PersistenceSession

/**
 * Unified RAG contract reusing the existing shared CommonRag abstraction.
 */
typealias UnifiedRag = CommonRag<UnifiedQuery, UnifiedResponse>

/**
 * Handle returned by the factory so callers can inspect identity and capabilities.
 */
data class UnifiedRagHandle(
    val id: RagId,
    val capabilities: RagCapabilities,
    val rag: UnifiedRag,
    val persistence: PersistenceSession? = null,
) : AutoCloseable {
    override fun close() {
        persistence?.close()
    }
}
