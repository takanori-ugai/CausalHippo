package shared.rag.spi.persistence

import tools.jackson.core.type.TypeReference
import tools.jackson.module.kotlin.jacksonObjectMapper
import com.mongodb.client.model.Filters
import com.mongodb.client.model.ReplaceOptions
import com.mongodb.kotlin.client.coroutine.MongoClient
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.bson.Document
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant
import java.util.Base64
import kotlin.math.sqrt

/**
 * MongoDB-backed persistence backend for graph/vector/kv/artifact state.
 */
class MongoDbPersistenceBackend : PersistenceBackendFactory {
    override fun open(config: Map<String, Any?>): PersistenceSession {
        val connectionString =
            config.string("connectionString")
                ?: config.string("uri")
                ?: config.string("mongoUri")
                ?: "mongodb://localhost:27017"
        val database =
            config.string("database")
                ?: config.string("mongoDatabase")
                ?: "unified_rag"
        val rootDir =
            config
                .string("rootDir")
                ?.trim()
                ?.takeIf { it.isNotEmpty() }
                ?.let { Path.of(it).toAbsolutePath().normalize() }

        return MongoDbPersistenceSession(
            connectionString = connectionString,
            databaseName = database,
            rootDir = rootDir,
            metadata = config.metadataMap(),
        )
    }
}

private class MongoDbPersistenceSession(
    connectionString: String,
    databaseName: String,
    private val rootDir: Path?,
    private val metadata: Map<String, String>,
) : PersistenceSession {
    private val objectMapper = jacksonObjectMapper()
    private val client = MongoClient.create(connectionString)
    private val database = client.getDatabase(databaseName)

    private val nodesCollection = database.getCollection<Document>("unified_graph_nodes")
    private val edgesCollection = database.getCollection<Document>("unified_graph_edges")
    private val vectorsCollection = database.getCollection<Document>("unified_vectors")
    private val kvCollection = database.getCollection<Document>("unified_kv")
    private val artifactsCollection = database.getCollection<Document>("unified_artifacts")

    override val backendId: String = "mongodb"
    override val capabilities: PersistenceCapabilities =
        PersistenceCapabilities(
            supportsGraphPersistence = true,
            supportsVectorPersistence = true,
            supportsKvPersistence = true,
            supportsAtomicCheckpoint = true,
            supportsIncrementalCheckpoint = false,
            supportsCrossBackendImport = true,
        )

    private var closed = false

    override fun graph(namespace: String): GraphStore {
        checkOpen()
        return MongoGraphStore(normalizeNamespace(namespace), nodesCollection, edgesCollection, objectMapper)
    }

    override fun vector(
        namespace: String,
        dimensions: Int?,
        metric: String,
    ): VectorIndexStore {
        checkOpen()
        return MongoVectorStore(normalizeNamespace(namespace), dimensions, metric, vectorsCollection, objectMapper)
    }

    override fun kv(namespace: String): KvStore {
        checkOpen()
        return MongoKvStore(normalizeNamespace(namespace), kvCollection, objectMapper)
    }

    override fun artifacts(namespace: String): ArtifactStore {
        checkOpen()
        return MongoArtifactStore(normalizeNamespace(namespace), artifactsCollection)
    }

    override suspend fun checkpoint(path: String): PersistenceManifest {
        checkOpen()
        val root = resolvePath(path)
        resetDirectory(root)

        val graphEntries =
            graphNamespaces()
                .sorted()
                .map { namespace ->
                    val snapshot = graph(namespace).snapshot()
                    val payloadPath = payloadPath("graph", namespace)
                    writeJson(root.resolve(payloadPath), snapshot)
                    GraphManifestEntry(
                        namespace = namespace,
                        nodeCount = snapshot.nodes.size,
                        edgeCount = snapshot.edges.size,
                        payloadPath = payloadPath,
                    )
                }

        val vectorEntries =
            vectorNamespaces()
                .sorted()
                .map { namespace ->
                    val snapshot = vector(namespace).snapshot()
                    val payloadPath = payloadPath("vector", namespace)
                    writeJson(root.resolve(payloadPath), snapshot)
                    VectorManifestEntry(
                        namespace = namespace,
                        dimensions = snapshot.dimensions,
                        metric = snapshot.metric,
                        recordCount = snapshot.records.size,
                        payloadPath = payloadPath,
                    )
                }

        val kvEntries =
            kvNamespaces()
                .sorted()
                .map { namespace ->
                    val snapshot = kv(namespace).snapshot()
                    val payloadPath = payloadPath("kv", namespace)
                    writeJson(root.resolve(payloadPath), snapshot)
                    KvManifestEntry(
                        namespace = namespace,
                        entryCount = snapshot.entries.size,
                        payloadPath = payloadPath,
                    )
                }

        val artifactEntries =
            artifactNamespaces()
                .sorted()
                .map { namespace ->
                    val snapshot = artifacts(namespace).snapshot()
                    val payloadPath = payloadPath("artifacts", namespace)
                    writeJson(root.resolve(payloadPath), snapshot)
                    ArtifactManifestEntry(
                        namespace = namespace,
                        fileCount = snapshot.filesBase64.size,
                        payloadPath = payloadPath,
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
                metadata = metadata,
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
        clearAll()

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
        client.close()
    }

    private fun checkOpen() {
        check(!closed) { "PersistenceSession is closed." }
    }

    private fun normalizeNamespace(namespace: String): String {
        val trimmed = namespace.trim()
        require(trimmed.isNotEmpty()) { "namespace must not be blank" }
        return trimmed
    }

    private fun resolvePath(path: String): Path {
        val raw = Path.of(path)
        return when {
            raw.isAbsolute -> raw.normalize()
            rootDir != null -> rootDir.resolve(raw).toAbsolutePath().normalize()
            else -> raw.toAbsolutePath().normalize()
        }
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

    private suspend fun graphNamespaces(): Set<String> =
        nodesCollection
            .find()
            .toList()
            .mapNotNull { it.getString("namespace") }
            .toSet() +
            edgesCollection
                .find()
                .toList()
                .mapNotNull { it.getString("namespace") }
                .toSet()

    private suspend fun vectorNamespaces(): Set<String> =
        vectorsCollection
            .find()
            .toList()
            .mapNotNull { it.getString("namespace") }
            .toSet()

    private suspend fun kvNamespaces(): Set<String> =
        kvCollection
            .find()
            .toList()
            .mapNotNull { it.getString("namespace") }
            .toSet()

    private suspend fun artifactNamespaces(): Set<String> =
        artifactsCollection
            .find()
            .toList()
            .mapNotNull { it.getString("namespace") }
            .toSet()

    private suspend fun clearAll() {
        nodesCollection.deleteMany(Document())
        edgesCollection.deleteMany(Document())
        vectorsCollection.deleteMany(Document())
        kvCollection.deleteMany(Document())
        artifactsCollection.deleteMany(Document())
    }
}

private class MongoGraphStore(
    private val namespace: String,
    private val nodesCollection: com.mongodb.kotlin.client.coroutine.MongoCollection<Document>,
    private val edgesCollection: com.mongodb.kotlin.client.coroutine.MongoCollection<Document>,
    private val objectMapper: tools.jackson.databind.ObjectMapper,
) : GraphStore {
    override fun upsertNode(
        id: String,
        data: Map<String, Any?>,
    ) {
        require(id.isNotBlank()) { "node id must not be blank" }
        runBlocking {
            val document =
                Document("_id", "$namespace:$id")
                    .append("namespace", namespace)
                    .append("id", id)
                    .append("data_json", objectMapper.writeValueAsString(data))
            nodesCollection.replaceOne(
                Filters.eq("_id", "$namespace:$id"),
                document,
                ReplaceOptions().upsert(true),
            )
        }
    }

    override fun upsertEdge(
        source: String,
        target: String,
        data: Map<String, Any?>,
    ) {
        require(source.isNotBlank()) { "edge source must not be blank" }
        require(target.isNotBlank()) { "edge target must not be blank" }
        runBlocking {
            upsertNode(source, emptyMap())
            upsertNode(target, emptyMap())
            val edgeId = "$namespace:$source:$target"
            val document =
                Document("_id", edgeId)
                    .append("namespace", namespace)
                    .append("source", source)
                    .append("target", target)
                    .append("data_json", objectMapper.writeValueAsString(data))
            edgesCollection.replaceOne(
                Filters.eq("_id", edgeId),
                document,
                ReplaceOptions().upsert(true),
            )
        }
    }

    override fun deleteNode(id: String) {
        runBlocking {
            nodesCollection.deleteOne(Filters.eq("_id", "$namespace:$id"))
            edgesCollection.deleteMany(
                Filters.and(
                    Filters.eq("namespace", namespace),
                    Filters.or(
                        Filters.eq("source", id),
                        Filters.eq("target", id),
                    ),
                ),
            )
        }
    }

    override fun deleteEdge(
        source: String,
        target: String,
    ) {
        runBlocking {
            edgesCollection.deleteOne(Filters.eq("_id", "$namespace:$source:$target"))
        }
    }

    override fun nodes(): List<GraphNodeRecord> =
        runBlocking {
            nodesCollection
                .find(Filters.eq("namespace", namespace))
                .toList()
                .sortedBy { it.getString("id") }
                .map { document ->
                    GraphNodeRecord(
                        id = document.getString("id"),
                        data = parseMap(document.getString("data_json") ?: "{}"),
                    )
                }
        }

    override fun edges(): List<GraphEdgeRecord> =
        runBlocking {
            edgesCollection
                .find(Filters.eq("namespace", namespace))
                .toList()
                .sortedWith(compareBy({ it.getString("source") }, { it.getString("target") }))
                .map { document ->
                    GraphEdgeRecord(
                        source = document.getString("source"),
                        target = document.getString("target"),
                        data = parseMap(document.getString("data_json") ?: "{}"),
                    )
                }
        }

    override fun snapshot(): GraphSnapshot {
        val nodes = nodes()
        val edges = edges()
        return GraphSnapshot(
            nodes = nodes,
            edges = edges,
            metadata = mapOf("nodeCount" to nodes.size, "edgeCount" to edges.size),
        )
    }

    override fun restore(snapshot: GraphSnapshot) {
        runBlocking {
            nodesCollection.deleteMany(Filters.eq("namespace", namespace))
            edgesCollection.deleteMany(Filters.eq("namespace", namespace))
        }
        snapshot.nodes.forEach { node -> upsertNode(node.id, node.data) }
        snapshot.edges.forEach { edge -> upsertEdge(edge.source, edge.target, edge.data) }
    }

    private fun parseMap(value: String): Map<String, Any?> {
        if (value.isBlank()) return emptyMap()
        return objectMapper.readValue(value, object : TypeReference<Map<String, Any?>>() {})
    }
}

private class MongoVectorStore(
    private val namespace: String,
    private val dimensions: Int?,
    private val metric: String,
    private val vectorsCollection: com.mongodb.kotlin.client.coroutine.MongoCollection<Document>,
    private val objectMapper: tools.jackson.databind.ObjectMapper,
) : VectorIndexStore {
    init {
        ensureSchema(dimensions, metric)
    }

    override fun upsert(record: VectorRecord) {
        require(record.id.isNotBlank()) { "vector id must not be blank" }
        val effectiveDimensions = dimensions ?: record.vector.size
        require(record.vector.size == effectiveDimensions) {
            "Vector dimension mismatch: expected=$effectiveDimensions, got=${record.vector.size}"
        }
        ensureSchema(effectiveDimensions, metric)

        runBlocking {
            val document =
                Document("_id", "$namespace:${record.id}")
                    .append("namespace", namespace)
                    .append("id", record.id)
                    .append("dimensions", effectiveDimensions)
                    .append("metric", metric)
                    .append("vector_json", objectMapper.writeValueAsString(record.vector))
                    .append("metadata_json", objectMapper.writeValueAsString(record.metadata))
            vectorsCollection.replaceOne(
                Filters.eq("_id", "$namespace:${record.id}"),
                document,
                ReplaceOptions().upsert(true),
            )
        }
    }

    override fun delete(id: String) {
        runBlocking {
            vectorsCollection.deleteOne(Filters.eq("_id", "$namespace:$id"))
        }
    }

    override fun query(
        vector: List<Double>,
        topK: Int,
    ): List<VectorRecord> {
        if (topK <= 0) return emptyList()
        val records = records()
        return records
            .asSequence()
            .filter { it.vector.size == vector.size }
            .map { record -> record to score(metric, vector, record.vector) }
            .sortedByDescending { (_, score) -> score }
            .take(topK)
            .map { (record, _) -> record }
            .toList()
    }

    override fun snapshot(): VectorSnapshot {
        val records = records()
        val dims = dimensions ?: records.firstOrNull()?.vector?.size
        return VectorSnapshot(
            dimensions = dims,
            metric = metric,
            records = records,
            metadata = mapOf("recordCount" to records.size),
        )
    }

    override fun restore(snapshot: VectorSnapshot) {
        runBlocking {
            vectorsCollection.deleteMany(Filters.eq("namespace", namespace))
        }
        val targetDims = snapshot.dimensions
        val targetMetric = snapshot.metric
        ensureSchema(targetDims, targetMetric)
        snapshot.records.forEach { upsert(it) }
    }

    private fun records(): List<VectorRecord> =
        runBlocking {
            vectorsCollection
                .find(Filters.eq("namespace", namespace))
                .toList()
                .sortedBy { it.getString("id") }
                .map { document ->
                    VectorRecord(
                        id = document.getString("id"),
                        vector = parseDoubleList(document.getString("vector_json") ?: "[]"),
                        metadata = parseMap(document.getString("metadata_json") ?: "{}"),
                    )
                }
        }

    private fun ensureSchema(
        requestedDimensions: Int?,
        requestedMetric: String,
    ) {
        val schema =
            runBlocking {
                vectorsCollection
                    .find(Filters.eq("namespace", namespace))
                    .toList()
                    .firstOrNull()
            }
        if (schema == null) return

        val existingDims = schema.get("dimensions") as? Int
        val existingMetric = schema.getString("metric") ?: requestedMetric

        if (requestedDimensions != null && existingDims != null) {
            require(existingDims == requestedDimensions) {
                "Vector namespace '$namespace' dimension mismatch: expected=$existingDims, got=$requestedDimensions"
            }
        }
        require(existingMetric == requestedMetric) {
            "Vector namespace '$namespace' metric mismatch: expected=$existingMetric, got=$requestedMetric"
        }
    }

    private fun parseMap(value: String): Map<String, Any?> {
        if (value.isBlank()) return emptyMap()
        return objectMapper.readValue(value, object : TypeReference<Map<String, Any?>>() {})
    }

    private fun parseDoubleList(value: String): List<Double> {
        if (value.isBlank()) return emptyList()
        return objectMapper.readValue(value, object : TypeReference<List<Double>>() {})
    }
}

private class MongoKvStore(
    private val namespace: String,
    private val kvCollection: com.mongodb.kotlin.client.coroutine.MongoCollection<Document>,
    private val objectMapper: tools.jackson.databind.ObjectMapper,
) : KvStore {
    override fun put(
        key: String,
        value: Any?,
    ) {
        require(key.isNotBlank()) { "key must not be blank" }
        runBlocking {
            val document =
                Document("_id", "$namespace:$key")
                    .append("namespace", namespace)
                    .append("key", key)
                    .append("value_json", objectMapper.writeValueAsString(value))
            kvCollection.replaceOne(
                Filters.eq("_id", "$namespace:$key"),
                document,
                ReplaceOptions().upsert(true),
            )
        }
    }

    override fun get(key: String): Any? =
        runBlocking {
            kvCollection
                .find(Filters.eq("_id", "$namespace:$key"))
                .toList()
                .firstOrNull()
                ?.getString("value_json")
                ?.let { parseAny(it) }
        }

    override fun delete(key: String) {
        runBlocking {
            kvCollection.deleteOne(Filters.eq("_id", "$namespace:$key"))
        }
    }

    override fun entries(): Map<String, Any?> =
        runBlocking {
            kvCollection
                .find(Filters.eq("namespace", namespace))
                .toList()
                .associate { document ->
                    val key = document.getString("key")
                    val value = parseAny(document.getString("value_json") ?: "null")
                    key to value
                }.toSortedMap()
        }

    override fun snapshot(): KvSnapshot {
        val entries = entries()
        return KvSnapshot(entries = entries, metadata = mapOf("entryCount" to entries.size))
    }

    override fun restore(snapshot: KvSnapshot) {
        runBlocking {
            kvCollection.deleteMany(Filters.eq("namespace", namespace))
        }
        snapshot.entries.forEach { (key, value) -> put(key, value) }
    }

    private fun parseAny(value: String): Any? {
        if (value.isBlank()) return null
        return objectMapper.readValue(value, object : TypeReference<Any?>() {})
    }
}

private class MongoArtifactStore(
    private val namespace: String,
    private val artifactsCollection: com.mongodb.kotlin.client.coroutine.MongoCollection<Document>,
) : ArtifactStore {
    override fun writeBytes(
        path: String,
        bytes: ByteArray,
    ) {
        require(path.isNotBlank()) { "artifact path must not be blank" }
        runBlocking {
            val document =
                Document("_id", "$namespace:$path")
                    .append("namespace", namespace)
                    .append("path", path)
                    .append("bytes_base64", Base64.getEncoder().encodeToString(bytes))
            artifactsCollection.replaceOne(
                Filters.eq("_id", "$namespace:$path"),
                document,
                ReplaceOptions().upsert(true),
            )
        }
    }

    override fun readBytes(path: String): ByteArray? =
        runBlocking {
            artifactsCollection
                .find(Filters.eq("_id", "$namespace:$path"))
                .toList()
                .firstOrNull()
                ?.getString("bytes_base64")
                ?.let { Base64.getDecoder().decode(it) }
        }

    override fun listPaths(): List<String> =
        runBlocking {
            artifactsCollection
                .find(Filters.eq("namespace", namespace))
                .toList()
                .mapNotNull { it.getString("path") }
                .sorted()
        }

    override fun delete(path: String) {
        runBlocking {
            artifactsCollection.deleteOne(Filters.eq("_id", "$namespace:$path"))
        }
    }

    override fun snapshot(): ArtifactSnapshot {
        val files =
            listPaths().associateWith { path ->
                val bytes = readBytes(path) ?: ByteArray(0)
                Base64.getEncoder().encodeToString(bytes)
            }
        return ArtifactSnapshot(filesBase64 = files, metadata = mapOf("fileCount" to files.size))
    }

    override fun restore(snapshot: ArtifactSnapshot) {
        runBlocking {
            artifactsCollection.deleteMany(Filters.eq("namespace", namespace))
        }
        snapshot.filesBase64.forEach { (path, base64) ->
            writeBytes(path, Base64.getDecoder().decode(base64))
        }
    }
}

private fun score(
    metric: String,
    query: List<Double>,
    candidate: List<Double>,
): Double =
    when (metric.lowercase()) {
        "cosine" -> cosine(query, candidate)
        "dot", "inner_product" -> dot(query, candidate)
        "euclidean", "l2" -> -euclidean(query, candidate)
        else -> cosine(query, candidate)
    }

private fun dot(
    left: List<Double>,
    right: List<Double>,
): Double = left.zip(right).sumOf { (a, b) -> a * b }

private fun cosine(
    left: List<Double>,
    right: List<Double>,
): Double {
    val denominator = sqrt(left.sumOf { it * it }) * sqrt(right.sumOf { it * it })
    if (denominator == 0.0) return 0.0
    return dot(left, right) / denominator
}

private fun euclidean(
    left: List<Double>,
    right: List<Double>,
): Double =
    sqrt(
        left.zip(right).sumOf { (a, b) ->
            val delta = a - b
            delta * delta
        },
    )

private fun Map<String, Any?>.string(key: String): String? = this[key] as? String
