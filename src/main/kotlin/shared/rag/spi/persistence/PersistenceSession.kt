package shared.rag.spi.persistence

import java.nio.file.Path

/**
 * Active persistence session.
 */
interface PersistenceSession : AutoCloseable {
    val backendId: String
    val capabilities: PersistenceCapabilities
    val checkpointMetadata: Map<String, String> get() = emptyMap()

    fun graph(namespace: String): GraphStore

    fun vector(
        namespace: String,
        dimensions: Int? = null,
        metric: String = "cosine",
    ): VectorIndexStore

    fun kv(namespace: String): KvStore

    fun artifacts(namespace: String = "default"): ArtifactStore

    fun resolveCheckpointPath(path: String): Path = Path.of(path).toAbsolutePath().normalize()

    suspend fun checkpoint(path: String): PersistenceManifest

    suspend fun restore(path: String)

    override fun close()
}
