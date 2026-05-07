package shared.rag.spi.persistence

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.time.Instant
import java.util.Base64
import kotlin.math.sqrt

internal abstract class SnapshotPersistenceSession(
    final override val backendId: String,
    final override val capabilities: PersistenceCapabilities,
    private val manifestMetadata: Map<String, String> = emptyMap(),
) : PersistenceSession {
    private val objectMapper = jacksonObjectMapper().findAndRegisterModules()

    private val graphStates = mutableMapOf<String, GraphState>()
    private val vectorStates = mutableMapOf<String, VectorState>()
    private val kvStates = mutableMapOf<String, KvState>()
    private val artifactStates = mutableMapOf<String, ArtifactState>()

    private var closed: Boolean = false

    override fun graph(namespace: String): GraphStore {
        checkOpen()
        val normalized = normalizeNamespace(namespace)
        val state = graphStates.getOrPut(normalized) { GraphState() }
        return GraphStoreImpl(state)
    }

    override fun vector(
        namespace: String,
        dimensions: Int?,
        metric: String,
    ): VectorIndexStore {
        checkOpen()
        val normalized = normalizeNamespace(namespace)
        val state =
            vectorStates.getOrPut(normalized) {
                VectorState(
                    dimensions = dimensions,
                    metric = metric,
                )
            }
        if (dimensions != null) {
            if (state.dimensions == null) {
                state.dimensions = dimensions
            } else {
                require(state.dimensions == dimensions) {
                    "Vector namespace '$normalized' dimension mismatch: expected=${state.dimensions}, got=$dimensions"
                }
            }
        }
        require(state.metric == metric) {
            "Vector namespace '$normalized' metric mismatch: expected=${state.metric}, got=$metric"
        }
        return VectorIndexStoreImpl(state)
    }

    override fun kv(namespace: String): KvStore {
        checkOpen()
        val normalized = normalizeNamespace(namespace)
        val state = kvStates.getOrPut(normalized) { KvState() }
        return KvStoreImpl(state)
    }

    override fun artifacts(namespace: String): ArtifactStore {
        checkOpen()
        val normalized = normalizeNamespace(namespace)
        val state = artifactStates.getOrPut(normalized) { ArtifactState() }
        return ArtifactStoreImpl(state)
    }

    override suspend fun checkpoint(path: String): PersistenceManifest {
        checkOpen()
        val root = resolvePath(path)
        resetDirectory(root)

        val graphEntries =
            graphStates
                .toSortedMap()
                .map { (namespace, state) ->
                    val snapshot = GraphStoreImpl(state).snapshot()
                    val relPath = payloadPath("graph", namespace)
                    writeJson(root.resolve(relPath), snapshot)
                    GraphManifestEntry(
                        namespace = namespace,
                        nodeCount = snapshot.nodes.size,
                        edgeCount = snapshot.edges.size,
                        payloadPath = relPath,
                    )
                }

        val vectorEntries =
            vectorStates
                .toSortedMap()
                .map { (namespace, state) ->
                    val snapshot = VectorIndexStoreImpl(state).snapshot()
                    val relPath = payloadPath("vector", namespace)
                    writeJson(root.resolve(relPath), snapshot)
                    VectorManifestEntry(
                        namespace = namespace,
                        dimensions = snapshot.dimensions,
                        metric = snapshot.metric,
                        recordCount = snapshot.records.size,
                        payloadPath = relPath,
                    )
                }

        val kvEntries =
            kvStates
                .toSortedMap()
                .map { (namespace, state) ->
                    val snapshot = KvStoreImpl(state).snapshot()
                    val relPath = payloadPath("kv", namespace)
                    writeJson(root.resolve(relPath), snapshot)
                    KvManifestEntry(
                        namespace = namespace,
                        entryCount = snapshot.entries.size,
                        payloadPath = relPath,
                    )
                }

        val artifactEntries =
            artifactStates
                .toSortedMap()
                .map { (namespace, state) ->
                    val snapshot = ArtifactStoreImpl(state).snapshot()
                    val relPath = payloadPath("artifacts", namespace)
                    writeJson(root.resolve(relPath), snapshot)
                    ArtifactManifestEntry(
                        namespace = namespace,
                        fileCount = snapshot.filesBase64.size,
                        payloadPath = relPath,
                    )
                }

        val manifest =
            PersistenceManifest(
                backendId = backendId,
                createdAt = Instant.now().toString(),
                graph = graphEntries,
                vector = vectorEntries,
                kv = kvEntries,
                artifacts = artifactEntries,
                metadata = manifestMetadata,
            )
        writeJson(root.resolve("manifest.json"), manifest)
        return manifest
    }

    override suspend fun restore(path: String) {
        checkOpen()
        val root = resolvePath(path)
        val manifestPath = root.resolve("manifest.json")
        require(Files.exists(manifestPath)) { "Persistence manifest not found: $manifestPath" }

        val manifest = readJson(manifestPath, PersistenceManifest::class.java)

        graphStates.clear()
        vectorStates.clear()
        kvStates.clear()
        artifactStates.clear()

        manifest.graph.forEach { entry ->
            val snapshot = readJson(root.resolve(entry.payloadPath), GraphSnapshot::class.java)
            graph(entry.namespace).restore(snapshot)
        }

        manifest.vector.forEach { entry ->
            val snapshot = readJson(root.resolve(entry.payloadPath), VectorSnapshot::class.java)
            vector(entry.namespace, snapshot.dimensions, snapshot.metric).restore(snapshot)
        }

        manifest.kv.forEach { entry ->
            val snapshot = readJson(root.resolve(entry.payloadPath), KvSnapshot::class.java)
            kv(entry.namespace).restore(snapshot)
        }

        manifest.artifacts.forEach { entry ->
            val snapshot = readJson(root.resolve(entry.payloadPath), ArtifactSnapshot::class.java)
            artifacts(entry.namespace).restore(snapshot)
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        graphStates.clear()
        vectorStates.clear()
        kvStates.clear()
        artifactStates.clear()
    }

    protected open fun resolvePath(path: String): Path = Path.of(path).toAbsolutePath().normalize()

    private fun checkOpen() {
        check(!closed) { "PersistenceSession is closed." }
    }

    private fun normalizeNamespace(namespace: String): String {
        val trimmed = namespace.trim()
        require(trimmed.isNotEmpty()) { "namespace must not be blank" }
        return trimmed
    }

    private fun payloadPath(
        prefix: String,
        namespace: String,
    ): String = "$prefix/${sanitizeForFileName(namespace)}.json"

    private fun sanitizeForFileName(raw: String): String {
        val trimmed = raw.trim()
        if (trimmed.isEmpty()) return "default"
        return trimmed.replace(Regex("[^A-Za-z0-9._-]"), "_")
    }

    private fun writeJson(
        path: Path,
        value: Any,
    ) {
        Files.createDirectories(path.parent)
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(path.toFile(), value)
    }

    private fun <T> readJson(
        path: Path,
        type: Class<T>,
    ): T {
        require(Files.exists(path)) { "Snapshot payload not found: $path" }
        return objectMapper.readValue(path.toFile(), type)
    }

    private fun resetDirectory(path: Path) {
        deleteDirectory(path)
        Files.createDirectories(path)
    }

    private fun deleteDirectory(path: Path) {
        if (!Files.exists(path)) return
        Files.walk(path).use { stream ->
            stream
                .sorted(Comparator.reverseOrder())
                .forEach { Files.deleteIfExists(it) }
        }
    }

    private data class GraphState(
        val nodes: MutableMap<String, MutableMap<String, Any?>> = linkedMapOf(),
        val edges: MutableMap<String, GraphEdgeRecord> = linkedMapOf(),
    )

    private class GraphStoreImpl(
        private val state: GraphState,
    ) : GraphStore {
        override fun upsertNode(
            id: String,
            data: Map<String, Any?>,
        ) {
            require(id.isNotBlank()) { "node id must not be blank" }
            state.nodes[id] = data.toMutableMap()
        }

        override fun upsertEdge(
            source: String,
            target: String,
            data: Map<String, Any?>,
        ) {
            require(source.isNotBlank()) { "edge source must not be blank" }
            require(target.isNotBlank()) { "edge target must not be blank" }
            state.edges[edgeKey(source, target)] = GraphEdgeRecord(source = source, target = target, data = data.toMap())
        }

        override fun deleteNode(id: String) {
            state.nodes.remove(id)
            state.edges.entries.removeIf { it.value.source == id || it.value.target == id }
        }

        override fun deleteEdge(
            source: String,
            target: String,
        ) {
            state.edges.remove(edgeKey(source, target))
        }

        override fun nodes(): List<GraphNodeRecord> =
            state.nodes
                .entries
                .sortedBy { it.key }
                .map { (id, data) -> GraphNodeRecord(id = id, data = data.toMap()) }

        override fun edges(): List<GraphEdgeRecord> =
            state.edges.values
                .sortedWith(compareBy({ it.source }, { it.target }))
                .map { edge -> edge.copy(data = edge.data.toMap()) }

        override fun snapshot(): GraphSnapshot {
            val nodes = nodes()
            val edges = edges()
            return GraphSnapshot(
                nodes = nodes,
                edges = edges,
                metadata =
                    mapOf(
                        "nodeCount" to nodes.size,
                        "edgeCount" to edges.size,
                    ),
            )
        }

        override fun restore(snapshot: GraphSnapshot) {
            state.nodes.clear()
            state.edges.clear()
            snapshot.nodes.forEach { node ->
                state.nodes[node.id] = node.data.toMutableMap()
            }
            snapshot.edges.forEach { edge ->
                state.edges[edgeKey(edge.source, edge.target)] = edge.copy(data = edge.data.toMap())
            }
        }

        private fun edgeKey(
            source: String,
            target: String,
        ): String = "$source\u0000$target"
    }

    private data class VectorState(
        var dimensions: Int?,
        var metric: String,
        val records: MutableMap<String, VectorRecord> = linkedMapOf(),
    )

    private class VectorIndexStoreImpl(
        private val state: VectorState,
    ) : VectorIndexStore {
        override fun upsert(record: VectorRecord) {
            require(record.id.isNotBlank()) { "vector id must not be blank" }
            val vectorSize = record.vector.size
            if (state.dimensions == null) {
                state.dimensions = vectorSize
            } else {
                require(state.dimensions == vectorSize) {
                    "Vector dimension mismatch for '${record.id}': expected=${state.dimensions}, got=$vectorSize"
                }
            }
            state.records[record.id] =
                record.copy(
                    vector = record.vector.toList(),
                    metadata = record.metadata.toMap(),
                )
        }

        override fun delete(id: String) {
            state.records.remove(id)
        }

        override fun query(
            vector: List<Double>,
            topK: Int,
        ): List<VectorRecord> {
            require(topK > 0) { "topK must be positive" }
            val scored =
                state.records.values.mapNotNull { record ->
                    val score = cosineSimilarity(vector, record.vector) ?: return@mapNotNull null
                    record to score
                }
            return scored
                .sortedByDescending { it.second }
                .take(topK)
                .map { it.first.copy(vector = it.first.vector.toList(), metadata = it.first.metadata.toMap()) }
        }

        override fun snapshot(): VectorSnapshot {
            val records =
                state.records.values
                    .sortedBy { it.id }
                    .map { it.copy(vector = it.vector.toList(), metadata = it.metadata.toMap()) }
            return VectorSnapshot(
                dimensions = state.dimensions,
                metric = state.metric,
                records = records,
                metadata = mapOf("recordCount" to records.size),
            )
        }

        override fun restore(snapshot: VectorSnapshot) {
            state.records.clear()
            state.dimensions = snapshot.dimensions
            state.metric = snapshot.metric
            snapshot.records.forEach(::upsert)
        }

        private fun cosineSimilarity(
            a: List<Double>,
            b: List<Double>,
        ): Double? {
            if (a.isEmpty() || b.isEmpty() || a.size != b.size) return null
            var dot = 0.0
            var normA = 0.0
            var normB = 0.0
            for (i in a.indices) {
                dot += a[i] * b[i]
                normA += a[i] * a[i]
                normB += b[i] * b[i]
            }
            if (normA <= 0.0 || normB <= 0.0) return null
            return dot / (sqrt(normA) * sqrt(normB))
        }
    }

    private data class KvState(
        val entries: MutableMap<String, Any?> = linkedMapOf(),
    )

    private class KvStoreImpl(
        private val state: KvState,
    ) : KvStore {
        override fun put(
            key: String,
            value: Any?,
        ) {
            require(key.isNotBlank()) { "kv key must not be blank" }
            state.entries[key] = value
        }

        override fun get(key: String): Any? = state.entries[key]

        override fun delete(key: String) {
            state.entries.remove(key)
        }

        override fun entries(): Map<String, Any?> = state.entries.toMap()

        override fun snapshot(): KvSnapshot =
            KvSnapshot(
                entries = state.entries.toMap(),
                metadata = mapOf("entryCount" to state.entries.size),
            )

        override fun restore(snapshot: KvSnapshot) {
            state.entries.clear()
            state.entries.putAll(snapshot.entries)
        }
    }

    private data class ArtifactState(
        val files: MutableMap<String, ByteArray> = linkedMapOf(),
    )

    private class ArtifactStoreImpl(
        private val state: ArtifactState,
    ) : ArtifactStore {
        override fun writeBytes(
            path: String,
            bytes: ByteArray,
        ) {
            require(path.isNotBlank()) { "artifact path must not be blank" }
            state.files[path] = bytes.copyOf()
        }

        override fun readBytes(path: String): ByteArray? = state.files[path]?.copyOf()

        override fun listPaths(): List<String> = state.files.keys.sorted()

        override fun delete(path: String) {
            state.files.remove(path)
        }

        override fun snapshot(): ArtifactSnapshot {
            val encoded =
                state.files
                    .entries
                    .sortedBy { it.key }
                    .associate { (path, bytes) -> path to Base64.getEncoder().encodeToString(bytes) }
            return ArtifactSnapshot(
                filesBase64 = encoded,
                metadata = mapOf("fileCount" to encoded.size),
            )
        }

        override fun restore(snapshot: ArtifactSnapshot) {
            state.files.clear()
            snapshot.filesBase64.forEach { (path, encoded) ->
                state.files[path] = Base64.getDecoder().decode(encoded)
            }
        }
    }
}

internal fun copyDirectory(
    source: Path,
    target: Path,
) {
    Files.walk(source).use { stream ->
        stream.forEach { from ->
            val rel = source.relativize(from)
            val to = target.resolve(rel.toString())
            if (Files.isDirectory(from)) {
                Files.createDirectories(to)
            } else {
                Files.createDirectories(to.parent)
                Files.copy(
                    from,
                    to,
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.COPY_ATTRIBUTES,
                )
            }
        }
    }
}
