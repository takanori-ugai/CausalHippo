package lightrag.kg.neo4j

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.store.embedding.CosineSimilarity
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import lightrag.core.Neo4jConfig
import lightrag.core.types.BaseVectorStorage
import lightrag.kg.memory.Metadata
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.Result
import org.neo4j.driver.Session
import org.neo4j.driver.SessionConfig
import org.neo4j.driver.Value
import java.util.concurrent.TimeUnit

private val logger = KotlinLogging.logger {}

/**
 * A simple Neo4j-backed vector storage. It keeps an in-memory cache for similarity
 * calculations and persists vectors/metadata as nodes in Neo4j for durability.
 * @property namespace The namespace of the storage.
 * @property workspace The workspace of the storage.
 * @property globalConfig The global configuration for the storage.
 * @property embeddingFunc The embedding model to use.
 * @property cosineThreshold The threshold for cosine similarity.
 */
class Neo4jVectorStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any?> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
    private val cosineThreshold: Double? = null,
) : BaseVectorStorage {
    /**
     * The threshold for cosine similarity.
     */
    override val cosineBetterThanThreshold: Double =
        cosineThreshold ?: (globalConfig["cosine_better_than_threshold"] as? Double ?: 0.2)

    /**
     * The set of meta fields.
     */
    override val metaFields: Set<String> = emptySet()

    private var driver: Driver? = null
    private val label: String = sanitizeLabel("${namespace}_$workspace")

    private val vectors = mutableMapOf<String, List<Float>>()
    private val metadata = mutableMapOf<String, Metadata>()

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

    private fun logQuery(
        query: String,
        params: Map<String, Any?>,
    ) {
        if (logger.isDebugEnabled()) {
            logger.debug { "[$namespace/$workspace][${databaseForLog()}] CYPHER: $query params=$params" }
        }
    }

    private fun Session.runLogged(
        query: String,
        params: Any? = null,
    ): Result =
        when (params) {
            null -> {
                logQuery(query, emptyMap())
                run(query)
            }

            is Value -> {
                logQuery(query, params.asMap())
                run(query, params)
            }

            is Map<*, *> -> {
                val castParams = params.entries.associate { (k, v) -> k.toString() to v }
                logQuery(query, castParams)
                run(query, castParams)
            }

            else -> {
                val wrapped = mapOf("param" to params)
                logQuery(query, wrapped)
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
            logger.error { "Neo4jVectorStorage[$namespace] NEO4J_URI not configured" }
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
                session
                    .runLogged(
                        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:$label) REQUIRE n.id IS UNIQUE",
                    ).consume()
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
                val result =
                    session.runLogged("MATCH (n:$label) RETURN n.id AS id, n.vector AS vector, n.metadata AS metadata")
                while (result.hasNext()) {
                    val record = result.next()
                    val id = record["id"].asString()
                    val vectorValue = record["vector"]
                    if (vectorValue == null || vectorValue.isNull) {
                        logger.warn { "[$namespace/$workspace] Skipping vector '$id' because stored vector is null" }
                        continue
                    }
                    val vec =
                        runCatching { vectorValue.asList { it.asDouble() }.map { it.toFloat() } }
                            .getOrElse {
                                logger.warn(it) { "[$namespace/$workspace] Skipping vector '$id' due to invalid vector format" }
                                emptyList()
                            }
                    if (vec.isEmpty()) continue

                    val rawMeta =
                        record["metadata"]
                            ?.takeIf { !it.isNull }
                            ?.let { value ->
                                // Neo4j stores properties as scalars or lists; older runs persisted metadata as
                                // a list of "key:value" strings. Try to read as a map first, then fall back to list parsing.
                                runCatching { value.asMap { v -> v.asString() } }
                                    .getOrElse { mapEx ->
                                        runCatching {
                                            value
                                                .asList { it.asString() }
                                                .mapNotNull { entry ->
                                                    val sep = entry.indexOf(':')
                                                    if (sep <= 0) return@mapNotNull null
                                                    val key = entry.substring(0, sep)
                                                    val rawVal = entry.substring(sep + 1)
                                                    key to rawVal
                                                }.toMap()
                                        }.getOrElse { listEx ->
                                            logger.warn(mapEx) {
                                                "[$namespace/$workspace] Skipping metadata for vector '$id' due to invalid format"
                                            }
                                            logger.warn(listEx) {
                                                "[$namespace/$workspace] Failed to parse legacy metadata list for vector '$id'"
                                            }
                                            emptyMap()
                                        }
                                    }
                            } ?: emptyMap()
                    vectors[id] = vec
                    metadata[id] = mapToMetadata(rawMeta)
                }
                if (logger.isInfoEnabled()) {
                    logger.info { "[$namespace/$workspace] Loaded ${vectors.size} vectors from Neo4j label '$label'" }
                }
            }
        }
    }

    /**
     * Callback for when indexing is done.
     */
    override suspend fun indexDoneCallback() {
        // no-op; data is persisted on each upsert
    }

    /**
     * Drops the storage.
     * @return A map with the status of the operation.
     */
    override suspend fun drop(): Map<String, String> {
        withContext(Dispatchers.IO) {
            driver?.session(sessionConfig())?.use { session ->
                session.runLogged("MATCH (n:$label) DETACH DELETE n").consume()
            }
        }
        vectors.clear()
        metadata.clear()
        return mapOf("status" to "success", "message" to "data dropped and file removed at $label")
    }

    /**
     * Queries the vector storage.
     * @param query The query string.
     * @param topK The number of top results to return.
     * @param queryEmbedding The query embedding.
     * @return A list of maps representing the results.
     */
    override suspend fun query(
        query: String,
        topK: Int,
        queryEmbedding: List<Float>?,
    ): List<Map<String, Any>> {
        val qVec = queryEmbedding ?: embed(query)
        if (qVec.isEmpty()) return emptyList()

        if (vectors.isEmpty()) {
            logger.warn { "Vector storage '$namespace' is empty during query." }
            return emptyList()
        }

        logger.debug {
            "[$namespace/$workspace] Query='$query', topK=$topK, vectors=${vectors.size}, metadata=${metadata.size}"
        }

        // Calculate cosine similarity for all vectors
        val results =
            vectors
                .mapNotNull { (id, vec) ->
                    val meta = metadata[id]
                    if (meta == null) {
                        logger.warn { "Skipping vector '$id' in '$namespace' due to missing metadata." }
                        return@mapNotNull null
                    }
                    if (vec.isEmpty()) {
                        logger.warn { "Skipping vector '$id' in '$namespace' because it is empty." }
                        return@mapNotNull null
                    }
                    if (vec.size != qVec.size) {
                        logger.warn {
                            "Skipping vector '$id' in '$namespace' due to dimension mismatch: stored=${vec.size}, query=${qVec.size}"
                        }
                        return@mapNotNull null
                    }
                    val similarity =
                        CosineSimilarity.between(
                            Embedding(qVec.toFloatArray()),
                            Embedding(vec.toFloatArray()),
                        )
                    Triple(id, similarity, meta)
                }.filter {
                    it.second >= cosineBetterThanThreshold
                }.sortedByDescending { it.second }
                .take(topK)

        if (results.isEmpty()) {
            logger.warn {
                "No results found for query: '$query' in '$namespace'. Vectors count: ${vectors.size}"
            }
        }

        return results.map { (id, score, meta) ->
            val raw = meta.raw
            raw + mapOf("id" to id, "score" to score, "distance" to score)
        }
    }

    /**
     * Upserts data into the vector storage.
     * @param data The data to upsert.
     */
    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        data.forEach { (id, metaMap) ->
            val meta = mapToMetadata(metaMap)
            metadata[id] = meta
            val vec =
                when {
                    meta.vector != null -> meta.vector
                    meta.content != null -> embed(meta.content)
                    else -> emptyList()
                }
            if (vec.isEmpty()) {
                logger.warn { "[$namespace/$workspace] No vector for id=$id; skipping persist" }
                return@forEach
            }
            vectors[id] = vec
            persistNode(id, vec, meta.raw)
        }
    }

    /**
     * Deletes an entity from the vector storage.
     * @param entityName The name of the entity to delete.
     */
    override suspend fun deleteEntity(entityName: String) {
        val idsToDelete =
            metadata.filter { it.value.entityName == entityName }.keys
        delete(idsToDelete.toList())
    }

    /**
     * Deletes an entity relation from the vector storage.
     * @param entityName The name of the entity relation to delete.
     */
    override suspend fun deleteEntityRelation(entityName: String) {
        val idsToDelete =
            metadata.filter { it.value.srcId == entityName || it.value.tgtId == entityName }.keys
        delete(idsToDelete.toList())
    }

    /**
     * Gets an item by its ID.
     * @param id The ID of the item to get.
     * @return A map representing the item.
     */
    override suspend fun getById(id: String): Map<String, Any>? {
        val meta = metadata[id]?.raw ?: return null
        return meta + mapOf("id" to id, "vector" to (vectors[id] ?: emptyList<Float>()))
    }

    /**
     * Gets items by their IDs.
     * @param ids The IDs of the items to get.
     * @return A list of maps representing the items.
     */
    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> = ids.mapNotNull { getById(it) }

    /**
     * Deletes items by their IDs.
     * @param ids The IDs of the items to delete.
     */
    override suspend fun delete(ids: List<String>) {
        if (ids.isEmpty()) return
        withContext(Dispatchers.IO) {
            driver?.session(sessionConfig())?.use { session ->
                session
                    .runLogged(
                        "MATCH (n:$label) WHERE n.id IN \$ids DETACH DELETE n",
                        mapOf("ids" to ids),
                    ).consume()
            }
        }
        ids.forEach {
            vectors.remove(it)
            metadata.remove(it)
        }
    }

    /**
     * Gets vectors by their IDs.
     * @param ids The IDs of the vectors to get.
     * @return A map of IDs to vectors.
     */
    override suspend fun getVectorsByIds(ids: List<String>): Map<String, List<Float>> =
        ids
            .mapNotNull { id ->
                vectors[id]?.let { id to it }
            }.toMap()

    private fun persistNode(
        id: String,
        vector: List<Float>,
        rawMeta: Map<String, Any>,
    ) {
        val metaList =
            rawMeta.entries.map { (k, v) ->
                val value =
                    when (v) {
                        is List<*> -> v.filterNotNull().joinToString(",")
                        else -> v.toString()
                    }
                "$k:$value"
            }
        val params =
            mapOf(
                "id" to id,
                "vector" to vector.map { it.toDouble() },
                "metadata" to metaList,
            )
        driver?.session(sessionConfig())?.use { session ->
            session
                .runLogged(
                    "MERGE (n:$label {id: \$id}) SET n.vector = \$vector, n.metadata = \$metadata",
                    params,
                ).consume()
        }
    }

    private fun embed(text: String): List<Float> =
        try {
            val response = embeddingFunc.embed(text)
            val content = response.content()
            content.vector().toList()
        } catch (e: IllegalStateException) {
            logger.error(e) { "Illegal state embedding text for Neo4jVectorStorage" }
            emptyList()
        } catch (e: IllegalArgumentException) {
            logger.error(e) { "Invalid input embedding text for Neo4jVectorStorage" }
            emptyList()
        }

    private fun mapToMetadata(meta: Map<String, Any>): Metadata {
        val content = meta["content"] as? String

        val vector = (meta["vector"] as? List<*>)?.mapNotNull { (it as? Number)?.toFloat() }
        val entityName = meta["entity_name"] as? String
        val srcId = meta["src_id"] as? String
        val tgtId = meta["tgt_id"] as? String
        return Metadata(
            content = content,
            vector = vector,
            entityName = entityName,
            srcId = srcId,
            tgtId = tgtId,
            raw = meta,
        )
    }
}
