package shared.rag.spi.persistence

/**
 * Active persistence session.
 */
interface PersistenceSession : AutoCloseable {
    val backendId: String
    val capabilities: PersistenceCapabilities

    fun graph(namespace: String): GraphStore

    fun vector(
        namespace: String,
        dimensions: Int? = null,
        metric: String = "cosine",
    ): VectorIndexStore

    fun kv(namespace: String): KvStore

    fun artifacts(namespace: String = "default"): ArtifactStore

    suspend fun checkpoint(path: String): PersistenceManifest

    suspend fun restore(path: String)

    override fun close()
}
