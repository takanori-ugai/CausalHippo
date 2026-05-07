package shared.rag.spi.persistence

/**
 * Factory for persistence backend sessions.
 */
interface PersistenceBackendFactory {
    fun open(config: Map<String, Any?> = emptyMap()): PersistenceSession
}
