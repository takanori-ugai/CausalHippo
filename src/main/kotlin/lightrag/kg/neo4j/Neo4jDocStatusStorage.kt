package lightrag.kg.neo4j

import dev.langchain4j.model.embedding.EmbeddingModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import lightrag.core.Neo4jConfig
import lightrag.core.types.DocProcessingStatus
import lightrag.core.types.DocStatus
import lightrag.core.types.DocStatusStorage
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.Result
import org.neo4j.driver.Session
import org.neo4j.driver.SessionConfig
import org.neo4j.driver.Value
import org.neo4j.driver.Values
import java.time.Instant
import java.util.concurrent.TimeUnit
import kotlin.math.min

private val logger = KotlinLogging.logger {}

/**
 * A Neo4j-backed storage for document processing statuses.
 * @property namespace The namespace of the storage.
 * @property workspace The workspace of the storage.
 * @property globalConfig The global configuration for the storage.
 * @property embeddingFunc The embedding model to use.
 */
class Neo4jDocStatusStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any?> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
) : DocStatusStorage {
    private var driver: Driver? = null
    private val label: String = sanitizeLabel("${namespace}_${workspace}_doc_status")
    private val docs = mutableMapOf<String, DocProcessingStatus>()
    private val json =
        Json {
            ignoreUnknownKeys = true
            prettyPrint = true
        }

    private fun sanitizeLabel(raw: String): String {
        var result = raw.replace(Regex("[^A-Za-z0-9_]+"), "_").trim('_')
        if (result.isEmpty()) result = "base"
        if (!result.first().isLetter() && result.first() != '_') {
            result = "l_$result"
        }
        return result
    }

    private fun parseNeo4jConfig(): Neo4jConfig? =
        when (val cfg = globalConfig["neo4j"]) {
            is Neo4jConfig -> {
                cfg
            }

            is Map<*, *> -> {
                Neo4jConfig(
                    uri = cfg["uri"] as? String,
                    username = cfg["username"] as? String,
                    password = cfg["password"] as? String,
                    database = cfg["database"] as? String,
                )
            }

            else -> {
                null
            }
        }

    private fun sessionConfig(): SessionConfig {
        val database = System.getenv("NEO4J_DATABASE") ?: parseNeo4jConfig()?.database
        return database?.let { SessionConfig.forDatabase(it) } ?: SessionConfig.defaultConfig()
    }

    private fun databaseForLog(): String = System.getenv("NEO4J_DATABASE") ?: parseNeo4jConfig()?.database ?: "default"

    private fun Session.runLogged(
        query: String,
        params: Any? = null,
    ): Result =
        when (params) {
            null -> {
                logger.debug { "[$namespace/$workspace][${databaseForLog()}] CYPHER: $query" }
                run(query)
            }

            is Value -> {
                val mapped = params.asMap()
                logger.debug { "[$namespace/$workspace][${databaseForLog()}] CYPHER: $query params=$mapped" }
                run(query, mapped)
            }

            is Map<*, *> -> {
                val castParams = params.entries.associate { (k, v) -> k.toString() to v }
                logger.debug { "[$namespace/$workspace][${databaseForLog()}] CYPHER: $query params=$castParams" }
                run(query, castParams)
            }

            else -> {
                val wrapped = mapOf("param" to params)
                logger.debug { "[$namespace/$workspace][${databaseForLog()}] CYPHER: $query params=$wrapped" }
                run(query, wrapped)
            }
        }

    /**
     * Initializes the storage by creating a Neo4j driver and creating a constraint on the label.
     */
    override suspend fun initialize() {
        val cfg = parseNeo4jConfig()
        val uri = System.getenv("NEO4J_URI") ?: cfg?.uri
        val username = System.getenv("NEO4J_USERNAME") ?: cfg?.username
        val password = System.getenv("NEO4J_PASSWORD") ?: cfg?.password

        if (uri == null) {
            logger.error { "Neo4jDocStatusStorage[$namespace] NEO4J_URI not configured" }
            return
        }

        val auth =
            if (username != null && password != null) {
                AuthTokens.basic(username, password)
            } else {
                AuthTokens.none()
            }

        val maxPool = System.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE")?.toIntOrNull() ?: 50
        val timeoutMs = System.getenv("NEO4J_CONNECTION_TIMEOUT")?.toLongOrNull() ?: 30_000L
        val maxLifetimeMs = System.getenv("NEO4J_MAX_CONNECTION_LIFETIME")?.toLongOrNull() ?: 300_000L

        val config =
            org.neo4j.driver.Config
                .builder()
                .withMaxConnectionPoolSize(maxPool)
                .withConnectionTimeout(timeoutMs, TimeUnit.MILLISECONDS)
                .withMaxConnectionLifetime(maxLifetimeMs, TimeUnit.MILLISECONDS)
                .build()

        driver = GraphDatabase.driver(uri, auth, config)

        withContext(Dispatchers.IO) {
            driver?.session(sessionConfig())?.use { session ->
                session.runLogged("CREATE CONSTRAINT IF NOT EXISTS FOR (n:$label) REQUIRE n.id IS UNIQUE").consume()
            }
        }

        loadFromNeo()
    }

    /**
     * Closes the Neo4j driver.
     */
    override suspend fun finalize() {
        driver?.close()
    }

    private suspend fun loadFromNeo() {
        val sessionCfg = sessionConfig()
        withContext(Dispatchers.IO) {
            driver?.session(sessionCfg)?.use { session ->
                val result = session.runLogged("MATCH (n:$label) RETURN n.id AS id, n.data_json AS data_json")
                result.list().forEach { record ->
                    val id = record["id"].asString()
                    val rawJson = record["data_json"].asString(null)
                    if (!rawJson.isNullOrBlank()) {
                        val parsed = json.decodeFromString<JsonObject>(rawJson)
                        val map =
                            (parsed.toAny() as? Map<*, *>)?.entries?.associate { (k, v) ->
                                k.toString() to (v as Any)
                            } ?: emptyMap()
                        docs[id] = mapToStatus(map, docs[id])
                    }
                }
                logger.info { "Loaded ${docs.size} doc statuses for '$namespace/$workspace' from Neo4j" }
            }
        }
    }

    /**
     * Callback for when indexing is done.
     */
    override suspend fun indexDoneCallback() {
        // writes are eager
    }

    /**
     * Gets a document by its ID.
     * @param id The ID of the document to get.
     * @return A map representing the document.
     */
    override suspend fun getById(id: String): Map<String, Any>? = docs[id]?.toMap(id)

    /**
     * Gets documents by their IDs.
     * @param ids The IDs of the documents to get.
     * @return A list of maps representing the documents.
     */
    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> = ids.mapNotNull { getById(it) }

    /**
     * Filters keys from the storage.
     * @param keys The keys to filter.
     * @return A set of the filtered keys.
     */
    override suspend fun filterKeys(keys: Set<String>): Set<String> = keys.filter { !docs.containsKey(it) }.toSet()

    /**
     * Upserts data into the storage.
     * @param data The data to upsert.
     */
    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        if (data.isEmpty()) return
        val sessionCfg = sessionConfig()
        val updates =
            data.map { (id, map) ->
                val existing = docs[id]
                val status = mapToStatus(map, existing)
                docs[id] = status
                val jsonStr =
                    json.encodeToString(
                        status.toMap(id).toJsonElement(),
                    )
                id to jsonStr
            }
        withContext(Dispatchers.IO) {
            driver?.session(sessionCfg)?.use { session ->
                updates.forEach { (id, payload) ->
                    session
                        .runLogged(
                            "MERGE (n:$label {id: \$id}) SET n.data_json = \$data_json",
                            Values.parameters("id", id, "data_json", payload),
                        ).consume()
                }
            }
        }
    }

    /**
     * Deletes items by their IDs.
     * @param ids The IDs of the items to delete.
     */
    override suspend fun delete(ids: List<String>) {
        ids.forEach { docs.remove(it) }
        if (ids.isEmpty()) return
        val sessionCfg = sessionConfig()
        withContext(Dispatchers.IO) {
            driver?.session(sessionCfg)?.use { session ->
                session
                    .runLogged(
                        "MATCH (n:$label) WHERE n.id IN \$ids DETACH DELETE n",
                        Values.parameters("ids", ids),
                    ).consume()
            }
        }
    }

    /**
     * Drops the storage.
     * @return A map with the status of the operation.
     */
    override suspend fun drop(): Map<String, String> {
        docs.clear()
        val sessionCfg = sessionConfig()
        withContext(Dispatchers.IO) {
            driver?.session(sessionCfg)?.use { session ->
                session.runLogged("MATCH (n:$label) DETACH DELETE n").consume()
            }
        }
        return mapOf("status" to "success", "message" to "Doc status data dropped for $label")
    }

    /**
     * Checks if the storage is empty.
     * @return True if the storage is empty, false otherwise.
     */
    override suspend fun isEmpty(): Boolean = docs.isEmpty()

    /**
     * Gets the status counts.
     * @return A map of status counts.
     */
    override suspend fun getStatusCounts(): Map<String, Int> = docs.values.groupingBy { it.status.value }.eachCount()

    /**
     * Gets documents by their status.
     * @param status The status of the documents to get.
     * @return A map of document IDs to document processing statuses.
     */
    override suspend fun getDocsByStatus(status: DocStatus): Map<String, DocProcessingStatus> = docs.filterValues { it.status == status }

    /**
     * Gets documents by their track ID.
     * @param trackId The track ID of the documents to get.
     * @return A map of document IDs to document processing statuses.
     */
    override suspend fun getDocsByTrackId(trackId: String): Map<String, DocProcessingStatus> = docs.filterValues { it.trackId == trackId }

    /**
     * Gets documents with pagination.
     * @param statusFilter The status to filter by.
     * @param page The page number.
     * @param pageSize The size of the page.
     * @param sortField The field to sort by.
     * @param sortDirection The direction to sort by.
     * @return A pair of the list of documents and the total number of documents.
     */
    override suspend fun getDocsPaginated(
        statusFilter: DocStatus?,
        page: Int,
        pageSize: Int,
        sortField: String,
        sortDirection: String,
    ): Pair<List<Pair<String, DocProcessingStatus>>, Int> {
        var filtered =
            if (statusFilter != null) {
                docs.filterValues { it.status == statusFilter }.toList()
            } else {
                docs.toList()
            }

        val total = filtered.size
        filtered =
            when (sortField) {
                "updated_at" -> filtered.sortedBy { it.second.updatedAt }
                "created_at" -> filtered.sortedBy { it.second.createdAt }
                else -> filtered.sortedBy { it.first }
            }
        if (sortDirection == "desc") filtered = filtered.reversed()

        val start = (page - 1) * pageSize
        val end = min(start + pageSize, total)
        if (start >= total) return Pair(emptyList(), total)

        return Pair(filtered.subList(start, end), total)
    }

    /**
     * Gets all status counts.
     * @return A map of all status counts.
     */
    override suspend fun getAllStatusCounts(): Map<String, Int> = getStatusCounts()

    /**

     * Gets a document by its file path.
     * @param filePath The file path of the document to get.
     * @return A map representing the document.
     */
    override suspend fun getDocByFilePath(filePath: String): Map<String, Any>? =
        docs.entries.find { it.value.filePath == filePath }?.let { getById(it.key) }

    private fun mapToStatus(
        map: Map<String, Any>,
        existing: DocProcessingStatus?,
    ): DocProcessingStatus {
        val statusStr = map["status"] as? String ?: existing?.status?.value ?: DocStatus.PENDING.value
        val status = DocStatus.values().find { it.value == statusStr } ?: DocStatus.PENDING
        return DocProcessingStatus(
            status = status,
            contentSummary = map["content_summary"] as? String ?: existing?.contentSummary ?: "",
            contentLength = map["content_length"]?.toString()?.toIntOrNull() ?: existing?.contentLength ?: 0,
            createdAt = map["created_at"] as? String ?: existing?.createdAt ?: Instant.now().toString(),
            updatedAt = map["updated_at"] as? String ?: existing?.updatedAt ?: Instant.now().toString(),
            filePath = map["file_path"] as? String ?: existing?.filePath ?: "",
            trackId = map["track_id"] as? String ?: existing?.trackId,
            chunksCount = map["chunks_count"]?.toString()?.toIntOrNull() ?: existing?.chunksCount,
            errorMsg = map["error_msg"] as? String ?: existing?.errorMsg,
        )
    }

    private fun DocProcessingStatus.toMap(id: String): Map<String, Any> =
        mapOf(
            "id" to id,
            "status" to status.value,
            "content_summary" to contentSummary,
            "content_length" to contentLength,
            "created_at" to createdAt,
            "updated_at" to updatedAt,
            "file_path" to filePath,
            "track_id" to (trackId ?: ""),
            "chunks_count" to (chunksCount ?: 0),
            "error_msg" to (errorMsg ?: ""),
        )

    private fun Any?.toJsonElement(): JsonElement =
        when (this) {
            null -> JsonNull
            is Boolean -> JsonPrimitive(this)
            is Number -> JsonPrimitive(this.toString())
            is String -> JsonPrimitive(this)
            is List<*> -> JsonArray(this.map { it.toJsonElement() })
            is Map<*, *> -> JsonObject(this.entries.associate { it.key.toString() to it.value.toJsonElement() })
            else -> JsonPrimitive(this.toString())
        }

    private fun JsonElement.toAny(): Any? =
        when (this) {
            is JsonNull -> {
                null
            }

            is JsonPrimitive -> {
                if (this.isString) {
                    this.content
                } else {
                    // Manually parse the content string to Boolean, Long, or Double
                    this.content.toBooleanStrictOrNull()
                        ?: this.content.toLongOrNull()
                        ?: this.content.toDoubleOrNull()
                        ?: this.content
                }
            }

            is JsonArray -> {
                this.map { it.toAny() }
            }

            is JsonObject -> {
                this.mapValues { it.value.toAny() }
            }
        }
}
