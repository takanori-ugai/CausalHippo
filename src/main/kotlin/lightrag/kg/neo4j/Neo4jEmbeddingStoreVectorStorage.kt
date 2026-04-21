package lightrag.kg.neo4j

import dev.langchain4j.community.store.embedding.neo4j.Neo4jEmbeddingStore
import dev.langchain4j.data.document.Metadata
import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.store.embedding.EmbeddingSearchRequest
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import lightrag.core.Neo4jConfig
import lightrag.core.types.BaseVectorStorage
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
 * Vector storage backed by langchain4j's Neo4jEmbeddingStore.
 * It uses Neo4j vector indexes for similarity search and persists metadata alongside embeddings.
 */
class Neo4jEmbeddingStoreVectorStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any?> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
    private val cosineThreshold: Double? = null,
) : BaseVectorStorage {
    override val cosineBetterThanThreshold: Double =
        cosineThreshold ?: (globalConfig["cosine_better_than_threshold"] as? Double ?: 0.2)
    override val metaFields: Set<String> = emptySet()

    private var driver: Driver? = null
    private var store: Neo4jEmbeddingStore? = null

    private val label: String = sanitizeLabel("${namespace}_$workspace")
    private val indexName: String = sanitizeLabel("vector_${namespace}_$workspace")
    private val idProperty: String = "id"
    private val embeddingProperty: String = "embedding"
    private val textProperty: String = "text"
    private val metadataPrefix: String =
        (globalConfig["neo4j_metadata_prefix"] as? String)
            ?: ((globalConfig["neo4j"] as? Map<*, *>)?.get("metadata_prefix") as? String)
            ?: System.getenv("NEO4J_METADATA_PREFIX")
            ?: "meta_"
    private val awaitIndexTimeoutSeconds: Long =
        System.getenv("NEO4J_AWAIT_INDEX_TIMEOUT")?.toLongOrNull()
            ?: ((globalConfig["neo4j"] as? Map<*, *>)?.get("await_index_timeout") as? Number)?.toLong()
            ?: 60L
    private var embeddingDimension: Int = 0

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

    private fun resolveDimension(): Int {
        val configured =
            System.getenv("NEO4J_EMBEDDING_DIMENSION")?.toIntOrNull()
                ?: ((globalConfig["embedding_dimension"] as? Number)?.toInt())
                ?: ((globalConfig["neo4j"] as? Map<*, *>)?.get("embedding_dimension") as? Number)?.toInt()
        if (configured != null) {
            return configured
        }

        return try {
            val vector = embeddingFunc.embed("dimension probe").content().vector()
            vector.size
        } catch (e: IllegalStateException) {
            val fallbackDimension = 1536
            logger.warn(e) {
                "Unable to infer embedding dimension for Neo4jEmbeddingStore; defaulting to $fallbackDimension"
            }
            fallbackDimension
        } catch (e: IllegalArgumentException) {
            val fallbackDimension = 1536
            logger.warn(e) {
                "Invalid input inferring embedding dimension for Neo4jEmbeddingStore; defaulting to $fallbackDimension"
            }
            fallbackDimension
        }
    }

    override suspend fun initialize() {
        val cfg = parseNeo4jConfig()
        val uri = System.getenv("NEO4J_URI") ?: cfg?.uri
        val username = System.getenv("NEO4J_USERNAME") ?: cfg?.username
        val password = System.getenv("NEO4J_PASSWORD") ?: cfg?.password

        if (uri == null) {
            logger.error { "Neo4jEmbeddingStoreVectorStorage[$namespace] NEO4J_URI not configured" }
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

        val driverConfig =
            org.neo4j.driver.Config
                .builder()
                .withMaxConnectionPoolSize(maxPool)
                .withConnectionTimeout(timeoutMs, TimeUnit.MILLISECONDS)
                .withMaxConnectionLifetime(maxLifetimeMs, TimeUnit.MILLISECONDS)
                .build()

        driver = GraphDatabase.driver(uri, auth, driverConfig)
        embeddingDimension = resolveDimension()

        store =
            Neo4jEmbeddingStore
                .builder()
                .driver(driver!!)
                .config(sessionConfig())
                .label(label)
                .indexName(indexName)
                .idProperty(idProperty)
                .embeddingProperty(embeddingProperty)
                .textProperty(textProperty)
                .metadataPrefix(metadataPrefix)
                .dimension(embeddingDimension)
                .awaitIndexTimeout(awaitIndexTimeoutSeconds)
                .initializeSchema(true)
                .build()

        logger.info {
            "Initialized Neo4jEmbeddingStoreVectorStorage label=$label index=$indexName dim=$embeddingDimension " +
                "metadataPrefix=$metadataPrefix"
        }
    }

    override suspend fun finalize() {
        driver?.close()
    }

    override suspend fun indexDoneCallback() {
        // Indexes are created eagerly on initialization.
    }

    override suspend fun drop(): Map<String, String> {
        store?.removeAll()
        return mapOf("status" to "success", "message" to "Deleted all nodes for label $label using Neo4jEmbeddingStore")
    }

    override suspend fun query(
        query: String,
        topK: Int,
        queryEmbedding: List<Float>?,
    ): List<Map<String, Any>> {
        val store = store ?: return emptyList()
        val qVec = queryEmbedding ?: embed(query)
        if (qVec.isEmpty()) return emptyList()

        val request =
            EmbeddingSearchRequest
                .builder()
                .queryEmbedding(Embedding.from(qVec.toFloatArray()))
                .maxResults(topK)
                .minScore(cosineBetterThanThreshold)
                .build()

        val matches = withContext(Dispatchers.IO) { store.search(request).matches() }
        return matches.map { match ->
            val meta =
                match
                    .embedded()
                    ?.metadata()
                    ?.toMap()
                    ?.toMutableMap() ?: mutableMapOf()
            match.embedded()?.text()?.let { meta["content"] = it }
            meta["id"] = match.embeddingId()
            meta["score"] = match.score()
            meta["distance"] = match.score()
            meta.toMap()
        }
    }

    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        val store =
            store ?: run {
                logger.error { "Neo4jEmbeddingStoreVectorStorage is not initialized" }
                return
            }
        if (data.isEmpty()) return

        val ids = mutableListOf<String>()
        val embeddings = mutableListOf<Embedding>()
        val segments = mutableListOf<TextSegment>()

        data.forEach { (id, metaMap) ->
            val meta = metaMap.toMutableMap()
            val content = meta["content"] as? String
            val providedVector =
                (meta["vector"] as? List<*>)?.mapNotNull { (it as? Number)?.toFloat() }
            meta.remove("content")
            meta.remove("vector")

            val metadata =
                Metadata(
                    meta.mapValues { (_, value) -> value.toString() },
                )

            val vector =
                when {
                    providedVector != null -> providedVector
                    content != null -> embed(content)
                    else -> emptyList()
                }

            if (vector.isEmpty()) {
                logger.warn { "[$namespace/$workspace] Skipping upsert for id=$id: no vector content." }
                return@forEach
            }

            ids.add(id)
            embeddings.add(Embedding.from(vector.toFloatArray()))
            segments.add(TextSegment.from(content ?: "", metadata))
        }

        withContext(Dispatchers.IO) {
            store.addAll(ids, embeddings, segments)
        }
    }

    override suspend fun deleteEntity(entityName: String) {
        val metaKey = metadataProperty("entity_name")
        val ids =
            findIdsByMetadata(metaKey, entityName) +
                findIdsByMetadata(metadataProperty("entityName"), entityName)
        delete(ids)
    }

    override suspend fun deleteEntityRelation(entityName: String) {
        val ids =
            findIdsByMetadata(metadataProperty("src_id"), entityName) +
                findIdsByMetadata(metadataProperty("tgt_id"), entityName)
        delete(ids)
    }

    override suspend fun getById(id: String): Map<String, Any>? = getByIds(listOf(id)).firstOrNull()

    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> {
        if (ids.isEmpty()) return emptyList()
        val sessionCfg = sessionConfig()
        val embeddingProp = store?.sanitizedEmbeddingProperty ?: embeddingProperty
        val idProp = store?.idProperty ?: idProperty

        return withContext(Dispatchers.IO) {
            driver?.session(sessionCfg)?.use { session ->
                val result =
                    session.runLogged(
                        "MATCH (n:${labelName()}) WHERE n.$idProp IN \$ids RETURN properties(n) AS props, n.$embeddingProp AS embedding",
                        mapOf("ids" to ids),
                    )
                result.list().mapNotNull { record ->
                    val props = record["props"].asMap()
                    val vector =
                        record["embedding"]?.let { value ->
                            value.asList { (it as Number).toFloat() }
                        } ?: emptyList()
                    val meta = mapPropertiesToRaw(props)
                    val entityId = props[idProp]?.toString() ?: return@mapNotNull null
                    meta + mapOf("id" to entityId, "vector" to vector)
                }
            } ?: emptyList()
        }
    }

    override suspend fun delete(ids: List<String>) {
        if (ids.isEmpty()) return
        withContext(Dispatchers.IO) {
            store?.removeAll(ids)
        }
    }

    override suspend fun getVectorsByIds(ids: List<String>): Map<String, List<Float>> {
        if (ids.isEmpty()) return emptyMap()
        val sessionCfg = sessionConfig()
        val embeddingProp = store?.sanitizedEmbeddingProperty ?: embeddingProperty
        val idProp = store?.idProperty ?: idProperty

        return withContext(Dispatchers.IO) {
            driver?.session(sessionCfg)?.use { session ->
                val result =
                    session.runLogged(
                        "MATCH (n:${labelName()}) WHERE n.$idProp IN \$ids RETURN n.$idProp AS id, n.$embeddingProp AS embedding",
                        mapOf("ids" to ids),
                    )
                result
                    .list()
                    .mapNotNull { record ->
                        val id = record["id"]?.asString() ?: return@mapNotNull null
                        val vec =
                            record["embedding"]?.asList { (it as Number).toFloat() } ?: emptyList()
                        id to vec
                    }.toMap()
            } ?: emptyMap()
        }
    }

    private fun mapPropertiesToRaw(props: Map<String, Any?>): Map<String, Any> {
        val notMeta = store?.notMetaKeys ?: setOf(idProperty, embeddingProperty, textProperty)
        val prefix = store?.metadataPrefix ?: metadataPrefix
        val meta = mutableMapOf<String, Any>()
        props.forEach { (key, value) ->
            if (!notMeta.contains(key)) {
                val cleanedKey = if (prefix.isNotEmpty() && key.startsWith(prefix)) key.removePrefix(prefix) else key
                meta[cleanedKey] = value?.toString() ?: ""
            } else if (key == textProperty && value != null) {
                meta["content"] = value.toString()
            }
        }
        return meta
    }

    private fun metadataProperty(key: String): String = metadataPrefix + key

    private suspend fun findIdsByMetadata(
        metaKey: String,
        value: String,
    ): List<String> {
        val idProp = store?.idProperty ?: idProperty
        val sessionCfg = sessionConfig()
        return withContext(Dispatchers.IO) {
            driver?.session(sessionCfg)?.use { session ->
                val result =
                    session.runLogged(
                        "MATCH (n:${labelName()}) WHERE n.$metaKey = \$value RETURN n.$idProp AS id",
                        mapOf("value" to value),
                    )
                result.list { record -> record["id"].asString() }
            } ?: emptyList()
        }
    }

    private fun labelName(): String = store?.sanitizedLabel ?: label

    private fun embed(text: String): List<Float> =
        try {
            val response = embeddingFunc.embed(text)
            response.content().vector().toList()
        } catch (e: IllegalStateException) {
            logger.error(e) { "Illegal state embedding text for Neo4jEmbeddingStoreVectorStorage" }
            emptyList()
        } catch (e: IllegalArgumentException) {
            logger.error(e) { "Invalid input embedding text for Neo4jEmbeddingStoreVectorStorage" }
            emptyList()
        }
}
