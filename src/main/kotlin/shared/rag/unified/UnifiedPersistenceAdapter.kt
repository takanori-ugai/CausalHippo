package shared.rag.unified

import kotlinx.coroutines.runBlocking
import shared.rag.spi.persistence.FilesystemSnapshotBackend
import shared.rag.spi.persistence.GraphEdgeRecord
import shared.rag.spi.persistence.GraphNodeRecord
import shared.rag.spi.persistence.GraphSnapshot
import shared.rag.spi.persistence.InMemoryPersistenceBackend
import shared.rag.spi.persistence.KvSnapshot
import shared.rag.spi.persistence.MongoDbPersistenceBackend
import shared.rag.spi.persistence.Neo4jPersistenceBackend
import shared.rag.spi.persistence.PersistenceBackendFactory
import shared.rag.spi.persistence.PersistenceSession
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant

internal class UnifiedPersistenceAdapter(
    private val ragId: RagId,
    private val delegate: UnifiedRag,
    private val session: PersistenceSession,
) : UnifiedRag {
    private val graphNamespace = "${ragId.name.lowercase()}_graph"
    private val metadataNamespace = "${ragId.name.lowercase()}_metadata"

    override fun upsert(data: String) = delegate.upsert(data)

    override fun upsert(data: Collection<String>) = delegate.upsert(data)

    override suspend fun aupsert(data: String) = delegate.aupsert(data)

    override suspend fun aupsert(data: Collection<String>) = delegate.aupsert(data)

    override fun drop() {
        delegate.drop()
        clearPersistedState()
    }

    override suspend fun adrop() {
        delegate.adrop()
        clearPersistedState()
    }

    override fun saveGraph(path: String) {
        delegate.saveGraph(path)
        val inspection = delegate.inspectGraph()
        persistFromInspection(inspection)
        runBlocking { checkpoint(path) }
    }

    override suspend fun asaveGraph(path: String) {
        delegate.asaveGraph(path)
        val inspection = delegate.ainspectGraph()
        persistFromInspection(inspection)
        checkpoint(path)
    }

    override fun loadGraph(path: String) {
        delegate.loadGraph(path)
        runBlocking { restore(path) }
        persistFromInspection(delegate.inspectGraph())
    }

    override suspend fun aloadGraph(path: String) {
        delegate.aloadGraph(path)
        restore(path)
        persistFromInspection(delegate.ainspectGraph())
    }

    override fun inspectGraph(): Map<String, Any?> {
        persistFromInspection(delegate.inspectGraph())
        return readPersistedInspection()
    }

    override suspend fun ainspectGraph(): Map<String, Any?> {
        persistFromInspection(delegate.ainspectGraph())
        return readPersistedInspection()
    }

    override fun query(
        query: String,
        param: UnifiedQuery,
    ): UnifiedResponse = delegate.query(query, param)

    override suspend fun aquery(
        query: String,
        param: UnifiedQuery,
    ): UnifiedResponse = delegate.aquery(query, param)

    private fun persistFromInspection(rawInspection: Map<String, Any?>) {
        val normalized = normalizeGraphInspection(rawInspection)
        val snapshot = normalized.toSnapshot()
        session.graph(graphNamespace).restore(snapshot)
        session
            .kv(metadataNamespace)
            .putAll(
                mapOf(
                    "ragId" to ragId.name,
                    "backendId" to session.backendId,
                    "updatedAt" to Instant.now().toString(),
                ),
            )
    }

    private fun readPersistedInspection(): Map<String, Any?> {
        val snapshot = session.graph(graphNamespace).snapshot()
        val nodes =
            snapshot.nodes.map { node ->
                linkedMapOf<String, Any?>("id" to node.id).apply { putAll(node.data) }
            }
        val edges =
            snapshot.edges.map { edge ->
                linkedMapOf<String, Any?>(
                    "source" to edge.source,
                    "target" to edge.target,
                ).apply { putAll(edge.data) }
            }
        val metadata =
            snapshot.metadata
                .toMutableMap()
                .apply {
                    put("nodeCount", nodes.size)
                    put("edgeCount", edges.size)
                    put("persistenceBackend", session.backendId)
                }
        return mapOf(
            "nodes" to nodes,
            "edges" to edges,
            "metadata" to metadata,
        )
    }

    private suspend fun checkpoint(path: String) {
        val checkpointPath = checkpointPath(path)
        session.checkpoint(checkpointPath)
        session.kv(metadataNamespace).put("lastCheckpointPath", checkpointPath)
    }

    private suspend fun restore(path: String) {
        val checkpointPath = checkpointPath(path)
        if (!checkpointExists(checkpointPath)) return
        session.restore(checkpointPath)
    }

    private fun checkpointPath(path: String): String = "$path.unified_spi"

    private fun checkpointExists(path: String): Boolean = Files.exists(Path.of(path).resolve("manifest.json"))

    private fun clearPersistedState() {
        session.graph(graphNamespace).restore(GraphSnapshot())
        session.kv(metadataNamespace).restore(KvSnapshot())
    }
}

internal object UnifiedPersistenceFactory {
    fun openSession(overrides: Map<String, Any?>): PersistenceSession {
        val backendName = overrides.string("persistenceBackend") ?: "filesystem_snapshot"
        val backend = resolveBackend(backendName)
        val config = buildConfig(overrides)
        return backend.open(config)
    }

    private fun resolveBackend(name: String): PersistenceBackendFactory =
        when (name.trim().lowercase()) {
            "filesystem", "filesystem_snapshot", "fs" -> {
                FilesystemSnapshotBackend()
            }

            "in_memory", "inmemory", "memory" -> {
                InMemoryPersistenceBackend()
            }

            "neo4j" -> {
                Neo4jPersistenceBackend()
            }

            "mongodb", "mongo" -> {
                MongoDbPersistenceBackend()
            }

            else -> {
                error(
                    "Unsupported unified persistence backend '$name'. " +
                        "Supported: filesystem_snapshot, in_memory, neo4j, mongodb",
                )
            }
        }

    private fun buildConfig(overrides: Map<String, Any?>): Map<String, Any?> {
        val fromOverrides = overrides.anyMap("persistenceConfig").toMutableMap()
        overrides.string("persistenceRootDir")?.let { fromOverrides["rootDir"] = it }
        val metadata =
            fromOverrides.anyMap("metadata").toMutableMap().apply {
                putIfAbsent("unifiedPersistence", "true")
            }
        fromOverrides["metadata"] = metadata
        return fromOverrides
    }

    private fun Map<String, Any?>.anyMap(key: String): Map<String, Any?> {
        val map = this[key] as? Map<*, *> ?: return emptyMap()
        return map.entries.associate { (k, v) -> k.toString() to v }
    }

    private fun Map<String, Any?>.string(key: String): String? = this[key] as? String
}

private fun Map<String, Any?>.toSnapshot(): GraphSnapshot {
    val nodes =
        (this["nodes"] as? List<*>).orEmpty().mapNotNull { raw ->
            val row = asStringMap(raw)
            val id = row["id"]?.toString()?.trim().orEmpty()
            if (id.isBlank()) {
                null
            } else {
                GraphNodeRecord(
                    id = id,
                    data = row - "id",
                )
            }
        }
    val edges =
        (this["edges"] as? List<*>).orEmpty().mapNotNull { raw ->
            val row = asStringMap(raw)
            val source = row["source"]?.toString()?.trim().orEmpty()
            val target = row["target"]?.toString()?.trim().orEmpty()
            if (source.isBlank() || target.isBlank()) {
                null
            } else {
                GraphEdgeRecord(
                    source = source,
                    target = target,
                    data = row - setOf("source", "target"),
                )
            }
        }
    return GraphSnapshot(
        nodes = nodes,
        edges = edges,
        metadata = asStringMap(this["metadata"]),
    )
}
