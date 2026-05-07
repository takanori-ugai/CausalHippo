package shared.rag.spi.persistence

/**
 * Versioned checkpoint manifest for unified persistence.
 */
data class PersistenceManifest(
    val manifestVersion: Int = 1,
    val backendId: String,
    val ragId: String? = null,
    val createdAt: String,
    val graph: List<GraphManifestEntry> = emptyList(),
    val vector: List<VectorManifestEntry> = emptyList(),
    val kv: List<KvManifestEntry> = emptyList(),
    val artifacts: List<ArtifactManifestEntry> = emptyList(),
    val metadata: Map<String, String> = emptyMap(),
)

data class GraphManifestEntry(
    val namespace: String,
    val nodeCount: Int,
    val edgeCount: Int,
    val payloadPath: String,
)

data class VectorManifestEntry(
    val namespace: String,
    val dimensions: Int?,
    val metric: String,
    val recordCount: Int,
    val payloadPath: String,
)

data class KvManifestEntry(
    val namespace: String,
    val entryCount: Int,
    val payloadPath: String,
)

data class ArtifactManifestEntry(
    val namespace: String,
    val fileCount: Int,
    val payloadPath: String,
)
