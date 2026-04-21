package lightrag.kg.neo4j

import dev.langchain4j.model.embedding.EmbeddingModel
import io.github.oshai.kotlinlogging.KLogger
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
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.longOrNull
import lightrag.core.Neo4jConfig
import lightrag.core.types.BaseKVStorage
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.Result
import org.neo4j.driver.Session
import org.neo4j.driver.SessionConfig
import org.neo4j.driver.Value
import org.neo4j.driver.Values
import java.util.concurrent.TimeUnit

private val logger = KotlinLogging.logger {}

/**
 * Simple Neo4j-backed KV storage.
 *
 * Values are stored under a label derived from namespace/workspace with properties:
 *  - id   : String (primary key)
 *  - data : Map<String, Any> (payload)
 *
 * A lightweight in-memory cache is kept for fast reads and to satisfy BaseKVStorage contract.
 * @property namespace The namespace of the storage.
 * @property workspace The workspace of the storage.
 * @property globalConfig The global configuration for the storage.
 * @property embeddingFunc The embedding model to use.
 */
class Neo4jKVStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any?> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
) : BaseKVStorage {
    private var driver: Driver? = null
    private val label: String = Neo4jKVHelpers.sanitizeLabel("${namespace}_${workspace}_kv")
    private val data = mutableMapOf<String, Map<String, Any>>()
    private val json =
        Json {
            ignoreUnknownKeys = true
            prettyPrint = true
        }
    private val helperContext = Neo4jKVContext(logger, namespace, workspace, globalConfig)

    /**
     * Initializes the storage by creating a Neo4j driver and creating a constraint on the label.
     */
    override suspend fun initialize() {
        val cfg = Neo4jKVHelpers.parseNeo4jConfig(helperContext)
        val uri = System.getenv("NEO4J_URI") ?: cfg?.uri
        val username = System.getenv("NEO4J_USERNAME") ?: cfg?.username
        val password = System.getenv("NEO4J_PASSWORD") ?: cfg?.password

        if (uri == null) {
            logger.error { "Neo4jKVStorage[$namespace] NEO4J_URI not configured" }
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
            driver
                ?.session(Neo4jKVHelpers.sessionConfig(helperContext))
                ?.use { session ->
                    Neo4jKVHelpers
                        .runLogged(
                            session = session,
                            context = helperContext,
                            query = "CREATE CONSTRAINT IF NOT EXISTS FOR (n:$label) REQUIRE n.id IS UNIQUE",
                        ).consume()
                }
        }

        Neo4jKVHelpers.loadFromNeo(
            driver = driver,
            label = label,
            json = json,
            context = helperContext,
            data = data,
        )
    }

    /**
     * Closes the Neo4j driver.
     */
    override suspend fun finalize() {
        driver?.close()
    }

    /**
     * Callback for when indexing is done.
     */
    override suspend fun indexDoneCallback() {
        // Writes happen eagerly in upsert/delete; nothing extra to do.
    }

    /**
     * Gets an item by its ID.
     * @param id The ID of the item to get.
     * @return A map representing the item.
     */
    override suspend fun getById(id: String): Map<String, Any>? = data[id]

    /**
     * Gets items by their IDs.
     * @param ids The IDs of the items to get.
     * @return A list of maps representing the items.
     */
    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> = ids.mapNotNull { data[it] }

    /**
     * Filters keys from the storage.
     * @param keys The keys to filter.
     * @return A set of the filtered keys.
     */
    override suspend fun filterKeys(keys: Set<String>): Set<String> = keys.filter { !data.containsKey(it) }.toSet()

    /**
     * Upserts data into the storage.
     * @param data The data to upsert.
     */
    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        if (data.isEmpty()) return
        this.data.putAll(data)

        val rows =
            data.map { (id, payload) ->
                mapOf("id" to id, "data" to payload)
            }
        val sessionCfg = Neo4jKVHelpers.sessionConfig(helperContext)
        withContext(Dispatchers.IO) {
            driver
                ?.session(sessionCfg)
                ?.use { session ->
                    rows.forEach { row ->
                        val jsonStr =
                            json.encodeToString(
                                Neo4jKVHelpers.toJsonElement(row["data"] ?: emptyMap<String, Any>()),
                            )
                        Neo4jKVHelpers
                            .runLogged(
                                session = session,
                                context = helperContext,
                                query = "MERGE (n:$label {id: \$id}) SET n.data_json = \$data_json",
                                params = Values.parameters("id", row["id"], "data_json", jsonStr),
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
        ids.forEach { data.remove(it) }
        if (ids.isEmpty()) return
        val sessionCfg = Neo4jKVHelpers.sessionConfig(helperContext)
        withContext(Dispatchers.IO) {
            driver
                ?.session(sessionCfg)
                ?.use { session ->
                    val cypher = "MATCH (n:$label) WHERE n.id IN " + "$" + "ids DETACH DELETE n"
                    Neo4jKVHelpers
                        .runLogged(
                            session = session,
                            context = helperContext,
                            query = cypher,
                            params = Values.parameters("ids", ids),
                        ).consume()
                }
        }
    }

    /**
     * Drops the storage.
     * @return A map with the status of the operation.
     */
    override suspend fun drop(): Map<String, String> {
        data.clear()
        val sessionCfg = Neo4jKVHelpers.sessionConfig(helperContext)
        withContext(Dispatchers.IO) {
            driver
                ?.session(sessionCfg)
                ?.use { session ->
                    Neo4jKVHelpers
                        .runLogged(
                            session = session,
                            context = helperContext,
                            query = "MATCH (n:$label) DETACH DELETE n",
                        ).consume()
                }
        }
        return mapOf("status" to "success", "message" to "Neo4j KV data dropped for $label")
    }

    /**
     * Checks if the storage is empty.
     * @return True if the storage is empty, false otherwise.
     */
    override suspend fun isEmpty(): Boolean = data.isEmpty()
}

private data class Neo4jKVContext(
    val logger: KLogger,
    val namespace: String,
    val workspace: String,
    val globalConfig: Map<String, Any?>,
)

private object Neo4jKVHelpers {
    fun sanitizeLabel(raw: String): String {
        var result = raw.replace(Regex("[^A-Za-z0-9_]+"), "_").trim('_')
        if (result.isEmpty()) result = "base"
        if (!result.first().isLetter() && result.first() != '_') {
            result = "l_$result"
        }
        return result
    }

    fun parseNeo4jConfig(context: Neo4jKVContext): Neo4jConfig? =
        when (val cfg = context.globalConfig["neo4j"]) {
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

    fun sessionConfig(context: Neo4jKVContext): SessionConfig {
        val database = System.getenv("NEO4J_DATABASE") ?: parseNeo4jConfig(context)?.database
        return database?.let { SessionConfig.forDatabase(it) } ?: SessionConfig.defaultConfig()
    }

    private fun databaseForLog(context: Neo4jKVContext): String =
        System.getenv("NEO4J_DATABASE") ?: parseNeo4jConfig(context)?.database ?: "default"

    private fun logQuery(
        context: Neo4jKVContext,
        query: String,
        params: Map<String, Any?>,
    ) {
        if (context.logger.isDebugEnabled()) {
            context.logger.debug {
                "[${context.namespace}/${context.workspace}][${databaseForLog(context)}] CYPHER: $query params=$params"
            }
        }
    }

    fun runLogged(
        session: Session,
        context: Neo4jKVContext,
        query: String,
        params: Any? = null,
    ): Result =
        when (params) {
            null -> {
                logQuery(context, query, emptyMap())
                session.run(query)
            }

            is Value -> {
                logQuery(context, query, params.asMap())
                session.run(query, params)
            }

            is Map<*, *> -> {
                val castParams = params.entries.associate { (k, v) -> k.toString() to v }
                logQuery(context, query, castParams)
                session.run(query, castParams)
            }

            else -> {
                val wrapped = mapOf("param" to params)
                logQuery(context, query, wrapped)
                session.run(query, wrapped)
            }
        }

    suspend fun loadFromNeo(
        driver: Driver?,
        label: String,
        json: Json,
        context: Neo4jKVContext,
        data: MutableMap<String, Map<String, Any>>,
    ) {
        val sessionCfg = sessionConfig(context)
        withContext(Dispatchers.IO) {
            driver?.session(sessionCfg)?.use { session ->
                val result =
                    runLogged(
                        session = session,
                        context = context,
                        query = "MATCH (n:$label) RETURN n.id AS id, n.data_json AS data_json",
                    )
                result.list().forEach { record ->
                    val id = record["id"].asString()
                    val rawJson = record["data_json"].asString(null)
                    val kv =
                        if (rawJson != null && rawJson.isNotBlank()) {
                            val parsed = json.decodeFromString<JsonObject>(rawJson)
                            (toAny(parsed) as? Map<*, *>)?.entries?.associate { (k, v) ->
                                k.toString() to (v as Any)
                            } ?: emptyMap()
                        } else {
                            emptyMap()
                        }
                    data[id] = kv
                }
                context.logger.info { "Loaded ${data.size} KV records for '${context.namespace}/${context.workspace}' from Neo4j" }
            }
        }
    }

    fun toJsonElement(value: Any?): JsonElement =
        when (value) {
            null -> {
                JsonNull
            }

            is Boolean -> {
                JsonPrimitive(value)
            }

            is Number -> {
                JsonPrimitive(value.toString())
            }

            is String -> {
                JsonPrimitive(value)
            }

            is List<*> -> {
                JsonArray(value.map { toJsonElement(it) })
            }

            is Map<*, *> -> {
                JsonObject(value.entries.associate { it.key.toString() to toJsonElement(it.value) })
            }

            else -> {
                JsonPrimitive(value.toString())
            }
        }

    fun toAny(element: JsonElement): Any? =
        when (element) {
            is JsonNull -> {
                null
            }

            is JsonPrimitive -> {
                if (element.isString) {
                    element.content
                } else {
                    element.booleanOrNull ?: element.longOrNull ?: element.doubleOrNull ?: element.content
                }
            }

            is JsonArray -> {
                element.map { toAny(it) }
            }

            is JsonObject -> {
                element.mapValues { toAny(it.value) }
            }
        }
}
