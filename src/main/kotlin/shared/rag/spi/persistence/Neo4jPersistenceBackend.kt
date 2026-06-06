package shared.rag.spi.persistence

import tools.jackson.core.type.TypeReference
import tools.jackson.module.kotlin.jacksonObjectMapper
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.Record
import org.neo4j.driver.Session
import org.neo4j.driver.SessionConfig
import org.neo4j.driver.TransactionContext
import org.neo4j.driver.Values
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant
import java.util.Base64
import kotlin.math.sqrt

/**
 * Neo4j-backed persistence backend for graph/vector/kv/artifact state.
 */
class Neo4jPersistenceBackend : PersistenceBackendFactory {
    override fun open(config: Map<String, Any?>): PersistenceSession {
        val uri = config.string("uri") ?: config.string("neo4jUri") ?: "bolt://localhost:7687"
        val username = config.string("username") ?: config.string("neo4jUser") ?: "neo4j"
        val password = config.string("password") ?: config.string("neo4jPassword") ?: "password"
        val database = config.string("database") ?: config.string("neo4jDatabase")
        val rootDir =
            config
                .string("rootDir")
                ?.trim()
                ?.takeIf { it.isNotEmpty() }
                ?.let { Path.of(it).toAbsolutePath().normalize() }

        return Neo4jPersistenceSession(
            uri = uri,
            username = username,
            password = password,
            database = database,
            rootDir = rootDir,
            metadata = config.metadataMap(),
        )
    }
}

private class Neo4jPersistenceSession(
    uri: String,
    username: String,
    password: String,
    private val database: String?,
    private val rootDir: Path?,
    private val metadata: Map<String, String>,
) : PersistenceSession {
    private val objectMapper = jacksonObjectMapper()
    private val driver: Driver = GraphDatabase.driver(uri, AuthTokens.basic(username, password))

    override val backendId: String = "neo4j"
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
        return Neo4jGraphStore(normalizeNamespace(namespace), ::readTx, ::writeTx, objectMapper)
    }

    override fun vector(
        namespace: String,
        dimensions: Int?,
        metric: String,
    ): VectorIndexStore {
        checkOpen()
        return Neo4jVectorStore(normalizeNamespace(namespace), dimensions, metric, ::readTx, ::writeTx, objectMapper)
    }

    override fun kv(namespace: String): KvStore {
        checkOpen()
        return Neo4jKvStore(normalizeNamespace(namespace), ::readTx, ::writeTx, objectMapper)
    }

    override fun artifacts(namespace: String): ArtifactStore {
        checkOpen()
        return Neo4jArtifactStore(normalizeNamespace(namespace), ::readTx, ::writeTx)
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
        driver.close()
    }

    private fun <T> readTx(block: (TransactionContext) -> T): T {
        checkOpen()
        return session().use { neo4jSession ->
            neo4jSession.executeRead { tx -> block(tx) }
        }
    }

    private fun <T> writeTx(block: (TransactionContext) -> T): T {
        checkOpen()
        return session().use { neo4jSession ->
            neo4jSession.executeWrite { tx -> block(tx) }
        }
    }

    private fun session(): Session =
        if (database.isNullOrBlank()) {
            driver.session()
        } else {
            driver.session(SessionConfig.forDatabase(database))
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

    private fun clearAll() {
        writeTx { tx ->
            tx.run("MATCH ()-[e:UNIFIED_EDGE]->() DELETE e")
            tx.run("MATCH (n:UnifiedNode) DELETE n")
            tx.run("MATCH (n:UnifiedVector) DELETE n")
            tx.run("MATCH (n:UnifiedKv) DELETE n")
            tx.run("MATCH (n:UnifiedArtifact) DELETE n")
        }
    }

    private fun graphNamespaces(): Set<String> =
        readTx { tx ->
            tx
                .run(
                    """
                    MATCH (n:UnifiedNode)
                    RETURN DISTINCT n.namespace AS namespace
                    UNION
                    MATCH ()-[e:UNIFIED_EDGE]->()
                    RETURN DISTINCT e.namespace AS namespace
                    """.trimIndent(),
                ).list { record -> record.get("namespace").asString() }
                .filter { it.isNotBlank() }
                .toSet()
        }

    private fun vectorNamespaces(): Set<String> =
        readTx { tx ->
            tx
                .run("MATCH (n:UnifiedVector) RETURN DISTINCT n.namespace AS namespace")
                .list { record -> record.get("namespace").asString() }
                .filter { it.isNotBlank() }
                .toSet()
        }

    private fun kvNamespaces(): Set<String> =
        readTx { tx ->
            tx
                .run("MATCH (n:UnifiedKv) RETURN DISTINCT n.namespace AS namespace")
                .list { record -> record.get("namespace").asString() }
                .filter { it.isNotBlank() }
                .toSet()
        }

    private fun artifactNamespaces(): Set<String> =
        readTx { tx ->
            tx
                .run("MATCH (n:UnifiedArtifact) RETURN DISTINCT n.namespace AS namespace")
                .list { record -> record.get("namespace").asString() }
                .filter { it.isNotBlank() }
                .toSet()
        }
}

private class Neo4jGraphStore(
    private val namespace: String,
    private val readTx: ((TransactionContext) -> Any?) -> Any?,
    private val writeTx: ((TransactionContext) -> Any?) -> Any?,
    private val objectMapper: tools.jackson.databind.ObjectMapper,
) : GraphStore {
    override fun upsertNode(
        id: String,
        data: Map<String, Any?>,
    ) {
        require(id.isNotBlank()) { "node id must not be blank" }
        val dataJson = toJson(data)
        writeTx { tx ->
            tx.run(
                """
                MERGE (n:UnifiedNode {namespace: ${'$'}namespace, id: ${'$'}id})
                SET n.data_json = ${'$'}dataJson
                """.trimIndent(),
                Values.parameters("namespace", namespace, "id", id, "dataJson", dataJson),
            )
            null
        }
    }

    override fun upsertEdge(
        source: String,
        target: String,
        data: Map<String, Any?>,
    ) {
        require(source.isNotBlank()) { "edge source must not be blank" }
        require(target.isNotBlank()) { "edge target must not be blank" }
        val dataJson = toJson(data)
        writeTx { tx ->
            tx.run(
                """
                MERGE (s:UnifiedNode {namespace: ${'$'}namespace, id: ${'$'}source})
                ON CREATE SET s.data_json = '{}'
                MERGE (t:UnifiedNode {namespace: ${'$'}namespace, id: ${'$'}target})
                ON CREATE SET t.data_json = '{}'
                MERGE (s)-[e:UNIFIED_EDGE {namespace: ${'$'}namespace, source: ${'$'}source, target: ${'$'}target}]->(t)
                SET e.data_json = ${'$'}dataJson
                """.trimIndent(),
                Values.parameters(
                    "namespace",
                    namespace,
                    "source",
                    source,
                    "target",
                    target,
                    "dataJson",
                    dataJson,
                ),
            )
            null
        }
    }

    override fun deleteNode(id: String) {
        writeTx { tx ->
            tx.run(
                "MATCH (n:UnifiedNode {namespace: ${'$'}namespace, id: ${'$'}id}) DETACH DELETE n",
                Values.parameters("namespace", namespace, "id", id),
            )
            null
        }
    }

    override fun deleteEdge(
        source: String,
        target: String,
    ) {
        writeTx { tx ->
            tx.run(
                "MATCH ()-[e:UNIFIED_EDGE {namespace: ${'$'}namespace, source: ${'$'}source, target: ${'$'}target}]->() DELETE e",
                Values.parameters("namespace", namespace, "source", source, "target", target),
            )
            null
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun nodes(): List<GraphNodeRecord> =
        readTx { tx ->
            tx
                .run(
                    """
                    MATCH (n:UnifiedNode {namespace: ${'$'}namespace})
                    RETURN n.id AS id, coalesce(n.data_json, '{}') AS dataJson
                    ORDER BY id
                    """.trimIndent(),
                    Values.parameters("namespace", namespace),
                ).list { record ->
                    GraphNodeRecord(
                        id = record.get("id").asString(),
                        data = parseMap(record.get("dataJson").asString()),
                    )
                }
        } as List<GraphNodeRecord>

    @Suppress("UNCHECKED_CAST")
    override fun edges(): List<GraphEdgeRecord> =
        readTx { tx ->
            tx
                .run(
                    """
                    MATCH ()-[e:UNIFIED_EDGE {namespace: ${'$'}namespace}]->()
                    RETURN e.source AS source, e.target AS target, coalesce(e.data_json, '{}') AS dataJson
                    ORDER BY source, target
                    """.trimIndent(),
                    Values.parameters("namespace", namespace),
                ).list { record ->
                    GraphEdgeRecord(
                        source = record.get("source").asString(),
                        target = record.get("target").asString(),
                        data = parseMap(record.get("dataJson").asString()),
                    )
                }
        } as List<GraphEdgeRecord>

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
        writeTx { tx ->
            tx.run(
                "MATCH (n:UnifiedNode {namespace: ${'$'}namespace}) DETACH DELETE n",
                Values.parameters("namespace", namespace),
            )
            null
        }
        snapshot.nodes.forEach { node -> upsertNode(node.id, node.data) }
        snapshot.edges.forEach { edge -> upsertEdge(edge.source, edge.target, edge.data) }
    }

    private fun toJson(value: Any?): String = objectMapper.writeValueAsString(value)

    private fun parseMap(value: String): Map<String, Any?> {
        if (value.isBlank()) return emptyMap()
        return objectMapper.readValue(value, object : TypeReference<Map<String, Any?>>() {})
    }
}

private class Neo4jVectorStore(
    private val namespace: String,
    private val dimensions: Int?,
    private val metric: String,
    private val readTx: ((TransactionContext) -> Any?) -> Any?,
    private val writeTx: ((TransactionContext) -> Any?) -> Any?,
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
        writeTx { tx ->
            tx.run(
                """
                MERGE (v:UnifiedVector {namespace: ${'$'}namespace, id: ${'$'}id})
                SET
                    v.vector_json = ${'$'}vectorJson,
                    v.metadata_json = ${'$'}metadataJson,
                    v.dimensions = ${'$'}dimensions,
                    v.metric = ${'$'}metric
                """.trimIndent(),
                Values.parameters(
                    "namespace",
                    namespace,
                    "id",
                    record.id,
                    "vectorJson",
                    toJson(record.vector),
                    "metadataJson",
                    toJson(record.metadata),
                    "dimensions",
                    effectiveDimensions,
                    "metric",
                    metric,
                ),
            )
            null
        }
    }

    override fun delete(id: String) {
        writeTx { tx ->
            tx.run(
                "MATCH (v:UnifiedVector {namespace: ${'$'}namespace, id: ${'$'}id}) DELETE v",
                Values.parameters("namespace", namespace, "id", id),
            )
            null
        }
    }

    @Suppress("UNCHECKED_CAST")
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
        writeTx { tx ->
            tx.run(
                "MATCH (v:UnifiedVector {namespace: ${'$'}namespace}) DELETE v",
                Values.parameters("namespace", namespace),
            )
            null
        }
        val targetMetric = snapshot.metric
        val targetDims = snapshot.dimensions
        ensureSchema(targetDims, targetMetric)
        snapshot.records.forEach { upsert(it) }
    }

    @Suppress("UNCHECKED_CAST")
    private fun records(): List<VectorRecord> =
        readTx { tx ->
            tx
                .run(
                    """
                    MATCH (v:UnifiedVector {namespace: ${'$'}namespace})
                    RETURN
                        v.id AS id,
                        coalesce(v.vector_json, '[]') AS vectorJson,
                        coalesce(v.metadata_json, '{}') AS metadataJson
                    ORDER BY id
                    """.trimIndent(),
                    Values.parameters("namespace", namespace),
                ).list { record ->
                    VectorRecord(
                        id = record.get("id").asString(),
                        vector = parseDoubleList(record.get("vectorJson").asString()),
                        metadata = parseMap(record.get("metadataJson").asString()),
                    )
                }
        } as List<VectorRecord>

    private fun ensureSchema(
        requestedDimensions: Int?,
        requestedMetric: String,
    ) {
        val schema =
            readTx { tx ->
                tx
                    .run(
                        """
                        MATCH (v:UnifiedVector {namespace: ${'$'}namespace})
                        RETURN v.dimensions AS dimensions, v.metric AS metric
                        LIMIT 1
                        """.trimIndent(),
                        Values.parameters("namespace", namespace),
                    ).list()
                    .singleOrNull()
            } as Record?
        if (schema == null) return

        val existingDims = if (schema.get("dimensions").isNull) null else schema.get("dimensions").asInt()
        val existingMetric = if (schema.get("metric").isNull) requestedMetric else schema.get("metric").asString()

        if (requestedDimensions != null && existingDims != null) {
            require(requestedDimensions == existingDims) {
                "Vector namespace '$namespace' dimension mismatch: expected=$existingDims, got=$requestedDimensions"
            }
        }
        require(existingMetric == requestedMetric) {
            "Vector namespace '$namespace' metric mismatch: expected=$existingMetric, got=$requestedMetric"
        }
    }

    private fun toJson(value: Any?): String = objectMapper.writeValueAsString(value)

    private fun parseMap(value: String): Map<String, Any?> {
        if (value.isBlank()) return emptyMap()
        return objectMapper.readValue(value, object : TypeReference<Map<String, Any?>>() {})
    }

    private fun parseDoubleList(value: String): List<Double> {
        if (value.isBlank()) return emptyList()
        return objectMapper.readValue(value, object : TypeReference<List<Double>>() {})
    }
}

private class Neo4jKvStore(
    private val namespace: String,
    private val readTx: ((TransactionContext) -> Any?) -> Any?,
    private val writeTx: ((TransactionContext) -> Any?) -> Any?,
    private val objectMapper: tools.jackson.databind.ObjectMapper,
) : KvStore {
    override fun put(
        key: String,
        value: Any?,
    ) {
        require(key.isNotBlank()) { "key must not be blank" }
        writeTx { tx ->
            tx.run(
                """
                MERGE (k:UnifiedKv {namespace: ${'$'}namespace, key: ${'$'}key})
                SET k.value_json = ${'$'}valueJson
                """.trimIndent(),
                Values.parameters(
                    "namespace",
                    namespace,
                    "key",
                    key,
                    "valueJson",
                    objectMapper.writeValueAsString(value),
                ),
            )
            null
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun get(key: String): Any? =
        readTx { tx ->
            tx
                .run(
                    "MATCH (k:UnifiedKv {namespace: ${'$'}namespace, key: ${'$'}key}) RETURN k.value_json AS valueJson LIMIT 1",
                    Values.parameters("namespace", namespace, "key", key),
                ).list()
                .singleOrNull()
                ?.let { record ->
                    parseAny(record.get("valueJson").asString())
                }
        }

    override fun delete(key: String) {
        writeTx { tx ->
            tx.run(
                "MATCH (k:UnifiedKv {namespace: ${'$'}namespace, key: ${'$'}key}) DELETE k",
                Values.parameters("namespace", namespace, "key", key),
            )
            null
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun entries(): Map<String, Any?> =
        readTx { tx ->
            tx
                .run(
                    "MATCH (k:UnifiedKv {namespace: ${'$'}namespace}) RETURN k.key AS key, k.value_json AS valueJson ORDER BY key",
                    Values.parameters("namespace", namespace),
                ).list { record ->
                    record.get("key").asString() to parseAny(record.get("valueJson").asString())
                }.toMap()
        } as Map<String, Any?>

    override fun snapshot(): KvSnapshot {
        val entries = entries()
        return KvSnapshot(entries = entries, metadata = mapOf("entryCount" to entries.size))
    }

    override fun restore(snapshot: KvSnapshot) {
        writeTx { tx ->
            tx.run(
                "MATCH (k:UnifiedKv {namespace: ${'$'}namespace}) DELETE k",
                Values.parameters("namespace", namespace),
            )
            null
        }
        snapshot.entries.forEach { (key, value) -> put(key, value) }
    }

    private fun parseAny(value: String): Any? {
        if (value.isBlank()) return null
        return objectMapper.readValue(value, object : TypeReference<Any?>() {})
    }
}

private class Neo4jArtifactStore(
    private val namespace: String,
    private val readTx: ((TransactionContext) -> Any?) -> Any?,
    private val writeTx: ((TransactionContext) -> Any?) -> Any?,
) : ArtifactStore {
    override fun writeBytes(
        path: String,
        bytes: ByteArray,
    ) {
        require(path.isNotBlank()) { "artifact path must not be blank" }
        val base64 = Base64.getEncoder().encodeToString(bytes)
        writeTx { tx ->
            tx.run(
                """
                MERGE (a:UnifiedArtifact {namespace: ${'$'}namespace, path: ${'$'}path})
                SET a.bytes_base64 = ${'$'}bytesBase64
                """.trimIndent(),
                Values.parameters(
                    "namespace",
                    namespace,
                    "path",
                    path,
                    "bytesBase64",
                    base64,
                ),
            )
            null
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun readBytes(path: String): ByteArray? =
        readTx { tx ->
            tx
                .run(
                    "MATCH (a:UnifiedArtifact {namespace: ${'$'}namespace, path: ${'$'}path}) RETURN a.bytes_base64 AS bytesBase64 LIMIT 1",
                    Values.parameters("namespace", namespace, "path", path),
                ).list()
                .singleOrNull()
                ?.let { record ->
                    val base64 = record.get("bytesBase64").asString()
                    Base64.getDecoder().decode(base64)
                }
        } as ByteArray?

    @Suppress("UNCHECKED_CAST")
    override fun listPaths(): List<String> =
        readTx { tx ->
            tx
                .run(
                    "MATCH (a:UnifiedArtifact {namespace: ${'$'}namespace}) RETURN a.path AS path ORDER BY path",
                    Values.parameters("namespace", namespace),
                ).list { record -> record.get("path").asString() }
        } as List<String>

    override fun delete(path: String) {
        writeTx { tx ->
            tx.run(
                "MATCH (a:UnifiedArtifact {namespace: ${'$'}namespace, path: ${'$'}path}) DELETE a",
                Values.parameters("namespace", namespace, "path", path),
            )
            null
        }
    }

    override fun snapshot(): ArtifactSnapshot {
        val entries =
            listPaths().associateWith { path ->
                val bytes = readBytes(path) ?: ByteArray(0)
                Base64.getEncoder().encodeToString(bytes)
            }
        return ArtifactSnapshot(filesBase64 = entries, metadata = mapOf("fileCount" to entries.size))
    }

    override fun restore(snapshot: ArtifactSnapshot) {
        writeTx { tx ->
            tx.run(
                "MATCH (a:UnifiedArtifact {namespace: ${'$'}namespace}) DELETE a",
                Values.parameters("namespace", namespace),
            )
            null
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
            val d = a - b
            d * d
        },
    )

private fun Map<String, Any?>.string(key: String): String? = this[key] as? String
