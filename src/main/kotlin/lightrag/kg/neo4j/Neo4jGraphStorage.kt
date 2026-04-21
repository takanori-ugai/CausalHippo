package lightrag.kg.neo4j

import dev.langchain4j.model.embedding.EmbeddingModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import lightrag.core.Neo4jConfig
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.KnowledgeGraph
import org.neo4j.driver.AuthToken
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.Result
import org.neo4j.driver.Session
import org.neo4j.driver.SessionConfig
import org.neo4j.driver.Transaction
import org.neo4j.driver.TransactionContext
import org.neo4j.driver.Value
import org.neo4j.driver.exceptions.ClientException
import org.neo4j.driver.exceptions.Neo4jException
import org.neo4j.driver.exceptions.ServiceUnavailableException
import org.neo4j.driver.types.Node
import java.util.concurrent.TimeUnit
import kotlin.math.min

private val logger = KotlinLogging.logger {}

/**
 * A Neo4j-backed graph storage implementation.
 * @property namespace The namespace of the storage.
 * @property globalConfig The global configuration for the storage.
 * @property embeddingFunc The embedding model to use.
 */
class Neo4jGraphStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
    override val embeddingFunc: EmbeddingModel,
) : BaseGraphStorage {
    /**
     * The workspace of the storage.
     */
    override val workspace: String

    private var driver: Driver? = null
    private val providedConfig: Neo4jConfig? = parseNeo4jConfig()
    private val database: String?

    companion object {
        private const val DEFAULT_POOL_SIZE = 100
        private const val DEFAULT_TIMEOUT_MS = 30000L
        private const val DEFAULT_MAX_LIFETIME_MS = 300000L
        private const val DEFAULT_URI = "bolt://localhost:7687"
        private const val DEFAULT_USERNAME = "neo4j"
        private const val SCORE_EXACT_MATCH = 1000
        private const val SCORE_STARTS_WITH = 500
        private const val SCORE_CONTAINS = 500
        private const val SCORE_PARTIAL = 50
        private const val DEFAULT_MAX_NODES = 1000
        private const val FALLBACK_SCORE_BASE = 100
    }

    init {
        // Read env and override the arg if present
        val neo4jWorkspace = System.getenv("NEO4J_WORKSPACE")
        val originalWorkspace = globalConfig["workspace"] as? String
        val resolvedWorkspace =
            when {
                !neo4jWorkspace.isNullOrBlank() -> neo4jWorkspace
                originalWorkspace.isNullOrBlank() -> "base"
                else -> originalWorkspace
            }
        workspace = resolvedWorkspace

        if (!neo4jWorkspace.isNullOrBlank()) {
            logger.info {
                "Using NEO4J_WORKSPACE environment variable: '$neo4jWorkspace' " +
                    "(overriding '$originalWorkspace/$namespace')"
            }
        }

        database =
            System.getenv("NEO4J_DATABASE")
                ?: providedConfig?.database
    }

    private fun sessionConfig(): SessionConfig = database?.let { SessionConfig.forDatabase(it) } ?: SessionConfig.defaultConfig()

    private fun databaseForLog(): String = database ?: "default"

    private fun logQuery(
        query: String,
        params: Map<String, Any?>,
    ) {
        if (logger.isDebugEnabled()) {
            logger.debug { "[$workspace][${databaseForLog()}] CYPHER: $query params=$params" }
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
                val castParams = params.entries.associate { it.key.toString() to it.value }
                logQuery(query, castParams)
                run(query, castParams)
            }

            else -> {
                val wrapped = mapOf("param" to params)
                logQuery(query, wrapped)
                run(query, wrapped)
            }
        }

    private fun Transaction.runLogged(
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
                val castParams = params.entries.associate { it.key.toString() to it.value }
                logQuery(query, castParams)
                run(query, castParams)
            }

            else -> {
                val wrapped = mapOf("param" to params)
                logQuery(query, wrapped)
                run(query, wrapped)
            }
        }

    private fun TransactionContext.runLogged(
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
                val castParams = params.entries.associate { it.key.toString() to it.value }
                logQuery(query, castParams)
                run(query, castParams)
            }

            else -> {
                val wrapped = mapOf("param" to params)
                logQuery(query, wrapped)
                run(query, wrapped)
            }
        }

    private fun isDebugEnabled(): Boolean {
        val flagFromConfig = (globalConfig["neo4j"] as? Map<*, *>)?.get("debug")?.toString()
        val debugFlag = System.getenv("NEO4J_DEBUG") ?: flagFromConfig
        return debugFlag?.lowercase() in setOf("1", "true", "yes", "on", "debug")
    }

    private fun logDebugInfo(
        uri: String,
        username: String?,
        maxConnectionPoolSize: Int,
        connectionTimeout: Long,
        maxConnectionLifetime: Long,
    ) {
        if (!isDebugEnabled()) return
        logger.debug {
            "[$workspace] Neo4j debug -> uri=$uri, database=${databaseForLog()}, user=${username ?: "N/A"}, " +
                "workspace_label=${getWorkspaceLabel()}, pool_size=$maxConnectionPoolSize, " +
                "connection_timeout_ms=$connectionTimeout, max_connection_lifetime_ms=$maxConnectionLifetime"
        }
    }

    private fun getWorkspaceLabel(): String = workspace

    private fun normalizeIndexSuffix(workspaceLabel: String): String {
        var normalized = workspaceLabel.replace(Regex("[^A-Za-z0-9_]+"), "_").trim('_')
        if (normalized.isEmpty()) {
            normalized = "base"
        }
        if (!normalized[0].isLetter() && normalized[0] != '_') {
            normalized = "ws_$normalized"
        }
        return normalized
    }

    private fun getFulltextIndexName(workspaceLabel: String): String {
        val suffix = normalizeIndexSuffix(workspaceLabel)
        return "entity_id_fulltext_idx_$suffix"
    }

    private fun isChineseText(text: String): Boolean {
        val cjkPattern = Regex("[\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff]|[�-�][�-�]")
        return cjkPattern.containsMatchIn(text)
    }

    private fun parseNeo4jConfig(): Neo4jConfig? =
        when (val config = globalConfig["neo4j"]) {
            is Neo4jConfig -> {
                config
            }

            is Map<*, *> -> {
                Neo4jConfig(
                    uri = config["uri"] as? String,
                    username = config["username"] as? String,
                    password = config["password"] as? String,
                    database = config["database"] as? String,
                )
            }

            else -> {
                null
            }
        }

    /**
     * Initializes the storage by creating a Neo4j driver and creating indexes.
     */
    override suspend fun initialize() {
        val (uri, username, password) = resolveCredentials() ?: return
        driver = GraphDatabase.driver(uri, buildAuthToken(username, password), buildDriverConfig())

        withContext(Dispatchers.IO) {
            val sessionCfg = sessionConfig()
            validateConnection(uri, sessionCfg)
            ensureBTreeIndex(sessionCfg)
            createFulltextIndex(driver!!, getWorkspaceLabel())
        }
    }

    private fun resolveCredentials(): Triple<String, String, String?>? {
        val uri = System.getenv("NEO4J_URI") ?: providedConfig?.uri ?: DEFAULT_URI
        val username = System.getenv("NEO4J_USERNAME") ?: providedConfig?.username ?: DEFAULT_USERNAME
        val password = System.getenv("NEO4J_PASSWORD") ?: providedConfig?.password

        if (uri.isBlank()) {
            logger.error { "NEO4J_URI is not set and no default available" }
            return null
        }
        return Triple(uri, username, password)
    }

    private fun buildAuthToken(
        username: String,
        password: String?,
    ): AuthToken = password?.let { AuthTokens.basic(username, it) } ?: AuthTokens.none()

    private fun buildDriverConfig(): org.neo4j.driver.Config {
        val maxConnectionPoolSize =
            (System.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE")?.toIntOrNull() ?: DEFAULT_POOL_SIZE)
        val connectionTimeout =
            (System.getenv("NEO4J_CONNECTION_TIMEOUT")?.toLongOrNull() ?: DEFAULT_TIMEOUT_MS)
        val maxConnectionLifetime =
            (System.getenv("NEO4J_MAX_CONNECTION_LIFETIME")?.toLongOrNull() ?: DEFAULT_MAX_LIFETIME_MS)

        logDebugInfo(
            uri = System.getenv("NEO4J_URI") ?: providedConfig?.uri ?: DEFAULT_URI,
            username = System.getenv("NEO4J_USERNAME") ?: providedConfig?.username ?: DEFAULT_USERNAME,
            maxConnectionPoolSize = maxConnectionPoolSize,
            connectionTimeout = connectionTimeout,
            maxConnectionLifetime = maxConnectionLifetime,
        )

        return org.neo4j.driver.Config
            .builder()
            .withMaxConnectionPoolSize(maxConnectionPoolSize)
            .withConnectionTimeout(connectionTimeout, TimeUnit.MILLISECONDS)
            .withMaxConnectionLifetime(maxConnectionLifetime, TimeUnit.MILLISECONDS)
            .build()
    }

    private suspend fun validateConnection(
        uri: String,
        sessionCfg: SessionConfig,
    ) {
        try {
            driver!!.session(sessionCfg).use { neoSession: Session ->
                neoSession.runLogged("MATCH (n) RETURN n LIMIT 0").consume()
            }
            logger.info { "[$workspace] Connected to ${databaseForLog()} at $uri" }
        } catch (e: ServiceUnavailableException) {
            logger.error { "[$workspace] Database ${databaseForLog()} at $uri is not available" }
            throw e
        } catch (e: ClientException) {
            logger.error {
                "[$workspace] Failed to access database ${databaseForLog()} at $uri: ${e.message}"
            }
            throw e
        } catch (e: Neo4jException) {
            logger.error { "[$workspace] Authentication or connection failed: ${e.message}" }
            throw e
        }
    }

    private fun ensureBTreeIndex(sessionCfg: SessionConfig) {
        val workspaceLabel = getWorkspaceLabel()
        try {
            driver!!.session(sessionCfg).use { neoSession: Session ->
                neoSession
                    .runLogged(
                        "CREATE INDEX IF NOT EXISTS FOR (n:`$workspaceLabel`) ON (n.entity_id)",
                    ).consume()
                logger.info {
                    "[$workspace] Ensured B-Tree index on entity_id for $workspaceLabel in " +
                        databaseForLog()
                }
            }
        } catch (e: Neo4jException) {
            logger.warn { "[$workspace] Failed to create B-Tree index: ${e.message}" }
        }
    }

    private fun createFulltextIndex(
        driver: Driver,
        workspaceLabel: String,
    ) {
        val indexName = getFulltextIndexName(workspaceLabel)
        val legacyIndexName = "entity_id_fulltext_idx"

        try {
            driver.session(sessionConfig()).use { neoSession: Session ->
                val (existingIndex, legacyIndex) = findIndexes(neoSession, indexName, legacyIndexName)
                migrateLegacyIndex(neoSession, legacyIndexName, legacyIndex, existingIndex, indexName)
                if (existingIndexIsOnline(existingIndex, indexName)) return

                dropExistingIndexIfNeeded(neoSession, indexName, existingIndex)
                createIndexWithFallback(neoSession, indexName, workspaceLabel)
            }
        } catch (e: Neo4jException) {
            if (e.message?.contains("Unknown command") == true ||
                e.message?.lowercase()?.contains("invalid syntax") == true
            ) {
                logger.warn {
                    "[$workspace] Could not create or verify full-text index '$indexName'. " +
                        "Use Neo4j version that supports it."
                }
            } else {
                logger.error {
                    "[$workspace] Failed to create or verify full-text index '$indexName': ${e.message}"
                }
            }
        }
    }

    private fun findIndexes(
        neoSession: Session,
        indexName: String,
        legacyIndexName: String,
    ): Pair<Map<String, Any>?, Map<String, Any>?> {
        val indexes = neoSession.runLogged("SHOW FULLTEXT INDEXES").list().map { it.asMap() }
        var existingIndex: Map<String, Any>? = null
        var legacyIndex: Map<String, Any>? = null

        for (idx in indexes) {
            if (idx["name"] == indexName) {
                existingIndex = idx
            } else if (idx["name"] == legacyIndexName) {
                legacyIndex = idx
            }
        }
        return existingIndex to legacyIndex
    }

    private fun migrateLegacyIndex(
        neoSession: Session,
        legacyIndexName: String,
        legacyIndex: Map<String, Any>?,
        existingIndex: Map<String, Any>?,
        targetIndexName: String,
    ) {
        if (legacyIndex == null || existingIndex != null) return
        logger.info { "[$workspace] Found legacy index '$legacyIndexName'. Migrating to '$targetIndexName'." }
        try {
            neoSession.runLogged("DROP INDEX $legacyIndexName IF EXISTS").consume()
            logger.info { "[$workspace] Dropped legacy index '$legacyIndexName'" }
        } catch (e: Neo4jException) {
            logger.warn { "[$workspace] Failed to drop legacy index: ${e.message}" }
        }
    }

    private fun existingIndexIsOnline(
        existingIndex: Map<String, Any>?,
        indexName: String,
    ): Boolean {
        if (existingIndex == null) {
            logger.info { "[$workspace] No existing index '$indexName' found. Creating new index." }
            return false
        }
        val indexState = existingIndex["state"] as? String ?: "UNKNOWN"
        logger.info { "[$workspace] Found existing index '$indexName' with state: $indexState" }
        return if (indexState == "ONLINE") {
            logger.info {
                "[$workspace] Full-text index '$indexName' already exists and is online. " +
                    "Skipping recreation."
            }
            true
        } else {
            logger.warn {
                "[$workspace] Existing index '$indexName' is not online (state: $indexState). Will recreate."
            }
            false
        }
    }

    private fun dropExistingIndexIfNeeded(
        neoSession: Session,
        indexName: String,
        existingIndex: Map<String, Any>?,
    ) {
        val needsRecreation = existingIndex != null && existingIndex["state"] != "ONLINE"
        if (!needsRecreation) return

        try {
            neoSession.runLogged("DROP INDEX $indexName IF EXISTS").consume()
            logger.info { "[$workspace] Dropped existing index '$indexName'" }
        } catch (e: Neo4jException) {
            logger.warn { "[$workspace] Failed to drop existing index: ${e.message}" }
        }
    }

    private fun createIndexWithFallback(
        neoSession: Session,
        indexName: String,
        workspaceLabel: String,
    ) {
        logger.info { "[$workspace] Creating full-text index '$indexName' with Chinese tokenizer support." }
        try {
            val createIndexQuery =
                """
                CREATE FULLTEXT INDEX $indexName
                FOR (n:`$workspaceLabel`) ON EACH [n.entity_id]
                OPTIONS {
                    indexConfig: {
                        `fulltext.analyzer`: 'cjk',
                        `fulltext.eventually_consistent`: true
                    }
                }
                """.trimIndent()
            neoSession.runLogged(createIndexQuery).consume()
            logger.info { "[$workspace] Successfully created full-text index '$indexName' with CJK analyzer." }
        } catch (cjkError: Neo4jException) {
            logger.warn {
                "[$workspace] CJK analyzer not supported: ${cjkError.message}. Falling back to standard analyzer."
            }
            val createIndexQuery =
                """
                CREATE FULLTEXT INDEX $indexName
                FOR (n:`$workspaceLabel`) ON EACH [n.entity_id]
                """.trimIndent()
            neoSession.runLogged(createIndexQuery).consume()
            logger.info { "[$workspace] Successfully created full-text index '$indexName' with standard analyzer." }
        }
    }

    /**
     * Closes the Neo4j driver.
     */
    override suspend fun finalize() {
        driver?.close()
        driver = null
    }

    /**
     * Callback for when indexing is done.
     */
    override suspend fun indexDoneCallback() {
        // Neo4j handles persistence
    }

    /**
     * Drops the storage.
     * @return A map with the status of the operation.
     */
    override suspend fun drop(): Map<String, String> {
        val workspaceLabel = getWorkspaceLabel()
        return withContext(Dispatchers.IO) {
            try {
                driver!!.session(sessionConfig()).use { neoSession: Session ->
                    val query = "MATCH (n:`$workspaceLabel`) DETACH DELETE n"
                    neoSession.runLogged(query).consume()
                }
                mapOf("status" to "success", "message" to "workspace '$workspaceLabel' data dropped")
            } catch (e: Neo4jException) {
                logger.error { "[$workspace] Error dropping Neo4j workspace '$workspaceLabel': ${e.message}" }
                mapOf("status" to "error", "message" to e.message.toString())
            }
        }
    }

    /**
     * Checks if a node exists.
     * @param nodeId The ID of the node to check.
     * @return True if the node exists, false otherwise.
     */
    override suspend fun hasNode(nodeId: String): Boolean =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                val query =
                    "MATCH (n:`$workspaceLabel` {entity_id: \$entity_id}) RETURN count(n) > 0 AS node_exists"
                val params = mapOf("entity_id" to nodeId)
                val result = neoSession.runLogged(query, params)
                if (result.hasNext()) {
                    result.single().get("node_exists").asBoolean()
                } else {
                    false
                }
            }
        }

    /**
     * Checks if an edge exists.
     * @param sourceNodeId The ID of the source node.
     * @param targetNodeId The ID of the target node.
     * @return True if the edge exists, false otherwise.
     */
    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (a:`$workspaceLabel` {entity_id: ${'$'}source_entity_id})-[r]-(b:`$workspaceLabel` {entity_id: ${'$'}target_entity_id})
                    RETURN COUNT(r) > 0 AS edgeExists
                    """.trimIndent()
                val params =
                    mapOf(
                        "source_entity_id" to sourceNodeId,
                        "target_entity_id" to targetNodeId,
                    )
                val result = neoSession.runLogged(query, params)
                if (result.hasNext()) {
                    result.single().get("edgeExists").asBoolean()
                } else {
                    false
                }
            }
        }

    /**
     * Gets the degree of a node.
     * @param nodeId The ID of the node.
     * @return The degree of the node.
     */
    override suspend fun nodeDegree(nodeId: String): Int =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (n:`$workspaceLabel` {entity_id: ${'$'}entity_id})
                    OPTIONAL MATCH (n)-[r]-() 
                    RETURN COUNT(r) AS degree
                    """.trimIndent()
                val params = mapOf("entity_id" to nodeId)
                val result = neoSession.runLogged(query, params)
                if (result.hasNext()) {
                    result.single().get("degree").asInt()
                } else {
                    0
                }
            }
        }

    /**
     * Gets the degree of an edge.
     * @param srcId The ID of the source node.
     * @param tgtId The ID of the target node.
     * @return The degree of the edge.
     */
    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int {
        val srcDegree = nodeDegree(srcId)
        val tgtDegree = nodeDegree(tgtId)
        return srcDegree + tgtDegree
    }

    /**
     * Gets a node by its ID.
     * @param nodeId The ID of the node to get.
     * @return A map representing the node.
     */
    override suspend fun getNode(nodeId: String): Map<String, String>? =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                val query = "MATCH (n:`$workspaceLabel` {entity_id: ${'$'}entity_id}) RETURN n"
                val params = mapOf("entity_id" to nodeId)
                val result = neoSession.runLogged(query, params)
                if (result.hasNext()) {
                    val record = result.next()
                    val node = record.get("n").asNode()
                    val nodeMap = node.asMap().toMutableMap()
                    if (nodeMap.containsKey("labels")) {
                        val labels = nodeMap["labels"] as? List<*>
                        if (labels != null) {
                            nodeMap["labels"] = labels.filter { it != workspaceLabel }
                        }
                    }
                    nodeMap.entries.associate { it.key to it.value.toString() }
                } else {
                    null
                }
            }
        }

    /**
     * Gets an edge by its source and target node IDs.
     * @param sourceNodeId The ID of the source node.
     * @param targetNodeId The ID of the target node.
     * @return A map representing the edge.
     */
    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, String>? =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (start:`$workspaceLabel` {entity_id: ${'$'}source_entity_id})-[r]-(end:`$workspaceLabel` {entity_id: ${'$'}target_entity_id})
                    RETURN properties(r) as edge_properties
                    """.trimIndent()
                val params =
                    mapOf(
                        "source_entity_id" to sourceNodeId,
                        "target_entity_id" to targetNodeId,
                    )
                val result = neoSession.runLogged(query, params)

                if (result.hasNext()) {
                    val record = result.next()
                    val edgeProps = record.get("edge_properties").asMap().toMutableMap()

                    val requiredKeys =
                        mapOf(
                            "weight" to 1.0,
                            "source_id" to null,
                            "description" to null,
                            "keywords" to null,
                        )
                    requiredKeys.forEach { (key, defaultVal) ->
                        if (!edgeProps.containsKey(key)) {
                            edgeProps[key] = defaultVal
                        }
                    }
                    edgeProps.entries.associate { it.key to (it.value?.toString() ?: "") }
                } else {
                    null
                }
            }
        }

    /**
     * Gets the edges of a node.
     * @param sourceNodeId The ID of the source node.
     * @return A list of pairs representing the edges.
     */
    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>>? =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (n:`$workspaceLabel` {entity_id: ${'$'}entity_id})
                    OPTIONAL MATCH (n)-[r]-(connected:`$workspaceLabel`)
                    WHERE connected.entity_id IS NOT NULL
                    RETURN n, r, connected
                    """.trimIndent()
                val params = mapOf("entity_id" to sourceNodeId)
                val result = neoSession.runLogged(query, params)
                val edges = mutableListOf<Pair<String, String>>()
                while (result.hasNext()) {
                    val record = result.next()
                    val sourceNode = record.get("n").asNode()
                    val connectedNode = record.get("connected")

                    if (!connectedNode.isNull) {
                        val cNode = connectedNode.asNode()
                        val sourceLabel = sourceNode.get("entity_id").asString()
                        val targetLabel = cNode.get("entity_id").asString()
                        if (sourceLabel != null && targetLabel != null) {
                            edges.add(sourceLabel to targetLabel)
                        }
                    }
                }
                edges
            }
        }

    /**
     * Upserts a node.
     * @param nodeId The ID of the node.
     * @param nodeData The data of the node.
     */
    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, String>,
    ) = withContext(Dispatchers.IO) {
        val workspaceLabel = getWorkspaceLabel()
        val properties = nodeData.toMutableMap()
        val entityType = properties["entity_type"] ?: "UNKNOWN"
        require(properties.containsKey("entity_id")) { "Neo4j: node properties must contain an 'entity_id' field" }

        driver!!.session(sessionConfig()).use { neoSession: Session ->
            neoSession.executeWrite { tx ->
                val query =
                    """
                    MERGE (n:`$workspaceLabel` {entity_id: ${'$'}entity_id})
                    SET n += ${'$'}properties
                    SET n:`$entityType`
                    """.trimIndent()
                tx.runLogged(query, mapOf("entity_id" to nodeId, "properties" to properties)).consume()
                Unit
            }
        }
        Unit
    }

    /**
     * Upserts an edge.
     * @param sourceNodeId The ID of the source node.
     * @param targetNodeId The ID of the target node.
     * @param edgeData The data of the edge.
     */
    override suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, String>,
    ) = withContext(Dispatchers.IO) {
        val workspaceLabel = getWorkspaceLabel()
        driver!!.session(sessionConfig()).use { neoSession: Session ->
            neoSession.executeWrite { tx ->
                val query =
                    """
                    MATCH (source:`$workspaceLabel` {entity_id: ${'$'}source_entity_id})
                    WITH source
                    MATCH (target:`$workspaceLabel` {entity_id: ${'$'}target_entity_id})
                    MERGE (source)-[r:DIRECTED]-(target)
                    SET r += ${'$'}properties
                    RETURN r
                    """.trimIndent()
                tx
                    .runLogged(
                        query,
                        mapOf(
                            "source_entity_id" to sourceNodeId,
                            "target_entity_id" to targetNodeId,
                            "properties" to edgeData,
                        ),
                    ).consume()
                Unit
            }
        }
        Unit
    }

    /**
     * Deletes a node.
     * @param nodeId The ID of the node to delete.
     */
    override suspend fun deleteNode(nodeId: String) =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                neoSession.executeWrite { tx ->
                    val query =
                        """
                        MATCH (n:`$workspaceLabel` {entity_id: ${'$'}entity_id})
                        DETACH DELETE n
                        """.trimIndent()
                    tx.runLogged(query, mapOf("entity_id" to nodeId)).consume()
                    Unit
                }
            }
            Unit
        }

    /**
     * Removes nodes.
     * @param nodes The nodes to remove.
     */
    override suspend fun removeNodes(nodes: List<String>) {
        nodes.forEach { deleteNode(it) }
    }

    /**
     * Removes edges.
     * @param edges The edges to remove.
     */
    override suspend fun removeEdges(edges: List<Pair<String, String>>) =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                neoSession.executeWrite { tx ->
                    val query =
                        """
                        MATCH (source:`$workspaceLabel` {entity_id: ${'$'}source_entity_id})-[r]-(target:`$workspaceLabel` {entity_id: ${'$'}target_entity_id})
                        DELETE r
                        """.trimIndent()
                    edges.forEach { (src, tgt) ->
                        tx.runLogged(query, mapOf("source_entity_id" to src, "target_entity_id" to tgt)).consume()
                    }
                    Unit
                }
            }
            Unit
        }

    /**
     * Gets all labels.
     * @return A list of all labels.
     */
    override suspend fun getAllLabels(): List<String> =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (n:`$workspaceLabel`)
                    WHERE n.entity_id IS NOT NULL
                    RETURN DISTINCT n.entity_id AS label
                    ORDER BY label
                    """.trimIndent()
                val result = neoSession.runLogged(query)
                result.list().map { it.get("label").asString() }
            }
        }

    /**
     * Gets the knowledge graph.
     * @param nodeLabel The label of the node.
     * @param maxDepth The maximum depth to traverse.
     * @param maxNodes The maximum number of nodes to return.
     * @return The knowledge graph.
     */
    override suspend fun getKnowledgeGraph(
        nodeLabel: String,
        maxDepth: Int,
        maxNodes: Int,
    ): KnowledgeGraph =
        withContext(Dispatchers.IO) {
            // Check maxNodes global config if needed, here we assume argument is correct
            var effectiveMaxNodes = maxNodes
            val maxGraphNodes = (globalConfig["max_graph_nodes"] as? Number)?.toInt() ?: DEFAULT_MAX_NODES
            effectiveMaxNodes = min(effectiveMaxNodes, maxGraphNodes)

            val workspaceLabel = getWorkspaceLabel()

            driver!!.session(sessionConfig()).use { neoSession: Session ->
                if (nodeLabel == "*") {
                    // Wildcard case
                    val countQuery = "MATCH (n:`$workspaceLabel`) RETURN count(n) as total"
                    val countResult = neoSession.runLogged(countQuery).single()
                    val total = countResult.get("total").asInt()

                    if (total > effectiveMaxNodes) {
                        logger.info {
                            "[$workspace] Graph truncated: $total nodes found, limited to $effectiveMaxNodes"
                        }
                    }
                }
            }

            if (nodeLabel == "*") {
                return@withContext getKnowledgeGraphWildcard(effectiveMaxNodes)
            }

            // Implementation using Robust BFS (Python's _robust_fallback)
            robustFallback(nodeLabel, maxDepth, effectiveMaxNodes)
        }

    private suspend fun robustFallback(
        nodeLabel: String,
        maxDepth: Int,
        maxNodes: Int,
    ): KnowledgeGraph {
        val nodes = mutableListOf<Map<String, Any>>()
        val edges = mutableListOf<Map<String, Any>>()
        var isTruncated = false
        val visitedNodes = mutableSetOf<String>() // entity_id
        val visitedEdges = mutableSetOf<String>() // edge elementId
        val visitedEdgePairs = mutableSetOf<Set<String>>()

        val queue = ArrayDeque<Triple<Map<String, Any>, Map<String, Any>?, Int>>() // Node, Edge?, Depth

        val workspaceLabel = getWorkspaceLabel()

        if (nodeLabel == "*") {
            return getKnowledgeGraphWildcard(maxNodes)
        }

        withContext(Dispatchers.IO) {
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                val query = "MATCH (n:`$workspaceLabel` {entity_id: ${'$'}entity_id}) RETURN n"
                val result = neoSession.runLogged(query, mapOf("entity_id" to nodeLabel))
                if (result.hasNext()) {
                    val node = result.next().get("n").asNode()
                    val props = node.asMap().toMutableMap()
                    val entityId = props["entity_id"]?.toString() ?: return@use
                    val nodeMap =
                        mapOf(
                            "id" to entityId,
                            "labels" to listOf(entityId),
                            "properties" to props,
                        )
                    queue.add(Triple(nodeMap, null, 0))
                }
            }
        }

        while (queue.isNotEmpty() && visitedNodes.size < maxNodes) {
            val (currentNode, currentEdge, currentDepth) = queue.removeFirst()
            val currentEntityId = currentNode["id"].toString()

            if (shouldSkipNode(currentEntityId, currentDepth, maxDepth, visitedNodes)) continue

            nodes.add(currentNode)
            visitedNodes.add(currentEntityId)

            if (currentEdge != null) {
                edges.add(currentEdge)
            }

            if (visitedNodes.size >= maxNodes) {
                isTruncated = true
                break
            }

            processNeighbors(
                currentEntityId,
                currentDepth,
                maxDepth,
                visitedNodes,
                visitedEdges,
                visitedEdgePairs,
                edges,
                queue,
                workspaceLabel,
            )
        }

        return KnowledgeGraph(nodes, edges, isTruncated)
    }

    private fun shouldSkipNode(
        entityId: String,
        depth: Int,
        maxDepth: Int,
        visitedNodes: Set<String>,
    ): Boolean = visitedNodes.contains(entityId) || depth > maxDepth

    private fun processNeighbors(
        currentEntityId: String,
        currentDepth: Int,
        maxDepth: Int,
        visitedNodes: Set<String>,
        visitedEdges: MutableSet<String>,
        visitedEdgePairs: MutableSet<Set<String>>,
        edges: MutableList<Map<String, Any>>,
        queue: ArrayDeque<Triple<Map<String, Any>, Map<String, Any>?, Int>>,
        workspaceLabel: String,
    ) {
        driver!!.session(sessionConfig()).use { neoSession: Session ->
            val query =
                """
                MATCH (a:`$workspaceLabel` {entity_id: ${'$'}entity_id})-[r]-(b)
                WITH r, b
                RETURN r, b
                """.trimIndent()
            val result = neoSession.runLogged(query, mapOf("entity_id" to currentEntityId))
            val records = result.list()

            for (record in records) {
                val r = record.get("r").asRelationship()
                val b = record.get("b").asNode()
                val bProps = b.asMap().toMutableMap()
                val bEntityId = bProps["entity_id"]?.toString() ?: continue
                val rProps = r.asMap().toMutableMap()
                val edgeId = r.elementId()
                val edgeMap =
                    mutableMapOf<String, Any>(
                        "id" to edgeId,
                        "type" to r.type(),
                        "source" to currentEntityId,
                        "target" to bEntityId,
                        "properties" to rProps,
                    )

                if (visitedEdges.contains(edgeId)) continue

                val pair = setOf(currentEntityId, bEntityId)

                if (pair !in visitedEdgePairs) {
                    if (visitedNodes.contains(bEntityId) || currentDepth < maxDepth) {
                        edges.add(edgeMap)
                        visitedEdges.add(edgeId)
                        visitedEdgePairs.add(pair)
                    }
                }

                if (!visitedNodes.contains(bEntityId)) {
                    if (currentDepth < maxDepth) {
                        val neighborNode =
                            mapOf(
                                "id" to bEntityId,
                                "labels" to listOf(bEntityId),
                                "properties" to bProps,
                            )
                        queue.add(Triple(neighborNode, null, currentDepth + 1))
                    }
                }
            }
        }
    }

    private data class WildcardData(
        val nodeInfos: List<Map<String, Any>>,
        val relationships: List<org.neo4j.driver.types.Relationship>,
    )

    private suspend fun getKnowledgeGraphWildcard(maxNodes: Int): KnowledgeGraph =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            val (nodes, edges, isTruncated) =
                driver!!.session(sessionConfig()).use { neoSession ->
                    val total = countNodes(neoSession, workspaceLabel)
                    val wildcardData = fetchWildcardData(neoSession, workspaceLabel, maxNodes)
                    buildWildcardGraph(wildcardData, maxNodes, total)
                }
            KnowledgeGraph(nodes, edges, isTruncated)
        }

    private fun countNodes(
        neoSession: Session,
        workspaceLabel: String,
    ): Int =
        neoSession
            .runLogged("MATCH (n:`$workspaceLabel`) RETURN count(n) as total")
            .single()
            .get("total")
            .asInt()

    private fun fetchWildcardData(
        neoSession: Session,
        workspaceLabel: String,
        maxNodes: Int,
    ): WildcardData {
        val pyQuery =
            """
            MATCH (n:`$workspaceLabel`)
            OPTIONAL MATCH (n)-[r]-() 
            WITH n, COALESCE(count(r), 0) AS degree
            ORDER BY degree DESC
            LIMIT ${'$'}max_nodes
            WITH collect({node: n}) AS filtered_nodes
            UNWIND filtered_nodes AS node_info
            WITH collect(node_info.node) AS kept_nodes, filtered_nodes
            OPTIONAL MATCH (a)-[r]-(b)
            WHERE a IN kept_nodes AND b IN kept_nodes
            RETURN filtered_nodes AS node_info,
                   collect(DISTINCT r) AS relationships
            """.trimIndent()

        val result = neoSession.runLogged(pyQuery, mapOf("max_nodes" to maxNodes))
        if (!result.hasNext()) return WildcardData(emptyList(), emptyList())

        val record = result.single()
        return WildcardData(
            nodeInfos = record.get("node_info").asList { it.asMap() },
            relationships = record.get("relationships").asList { it.asRelationship() },
        )
    }

    private fun buildWildcardGraph(
        wildcardData: WildcardData,
        maxNodes: Int,
        totalNodes: Int,
    ): Triple<List<Map<String, Any>>, List<Map<String, Any>>, Boolean> {
        val nodes = mutableListOf<Map<String, Any>>()
        val edges = mutableListOf<Map<String, Any>>()
        val idToEntityId = mutableMapOf<String, String>()

        wildcardData.nodeInfos.forEach { info ->
            val node = info["node"] as? Node ?: return@forEach
            val props = node.asMap().toMutableMap()
            val entityId = props["entity_id"] as? String ?: return@forEach
            idToEntityId[node.elementId()] = entityId
            nodes.add(
                mapOf(
                    "id" to entityId,
                    "labels" to listOf(entityId),
                    "properties" to props,
                ),
            )
        }

        wildcardData.relationships.forEach { rel ->
            val srcEntityId = idToEntityId[rel.startNodeElementId()]
            val tgtEntityId = idToEntityId[rel.endNodeElementId()]

            if (srcEntityId != null && tgtEntityId != null) {
                val edgeProps = rel.asMap().toMutableMap()
                edges.add(
                    mapOf(
                        "id" to rel.elementId(),
                        "type" to rel.type(),
                        "source" to srcEntityId,
                        "target" to tgtEntityId,
                        "properties" to edgeProps,
                    ),
                )
            }
        }

        val isTruncated = totalNodes > maxNodes
        return Triple(nodes, edges, isTruncated)
    }

    /**
     * Gets all nodes.
     * @return A list of all nodes.
     */
    override suspend fun getAllNodes(): List<Map<String, Any>> =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                val query = "MATCH (n:`$workspaceLabel`) RETURN n"
                val result = neoSession.runLogged(query)
                result.list().map { record ->
                    val node = record.get("n").asNode()
                    val map = node.asMap().toMutableMap()
                    map["id"] = map["entity_id"] ?: ""
                    map
                }
            }
        }

    /**
     * Gets all edges.
     * @return A list of all edges.
     */
    override suspend fun getAllEdges(): List<Map<String, Any>> =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (a:`$workspaceLabel`)-[r]-(b:`$workspaceLabel`)
                    RETURN DISTINCT a.entity_id AS source, b.entity_id AS target, properties(r) AS properties
                    """.trimIndent()
                val result = neoSession.runLogged(query)
                result.list().map { record ->
                    val props = record.get("properties").asMap().toMutableMap()
                    props["source"] = record.get("source").asString()
                    props["target"] = record.get("target").asString()
                    props
                }
            }
        }

    /**
     * Gets popular labels.
     * @param limit The maximum number of labels to return.
     * @return A list of popular labels.
     */
    override suspend fun getPopularLabels(limit: Int): List<String> =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(sessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (n:`$workspaceLabel`)
                    WHERE n.entity_id IS NOT NULL
                    OPTIONAL MATCH (n)-[r]-() 
                    WITH n.entity_id AS label, count(r) AS degree
                    ORDER BY degree DESC, label ASC
                    LIMIT ${'$'}limit
                    RETURN label
                    """.trimIndent()
                val result = neoSession.runLogged(query, mapOf("limit" to limit))
                result.list().map { it.get("label").asString() }
            }
        }

    /**
     * Searches for labels.
     * @param query The query to search for.
     * @param limit The maximum number of labels to return.
     * @return A list of labels.
     */
    override suspend fun searchLabels(
        query: String,
        limit: Int,
    ): List<String> =
        withContext(Dispatchers.IO) {
            val queryStrip = query.trim()
            if (queryStrip.isEmpty()) return@withContext emptyList<String>()

            val context =
                LabelSearchContext(
                    workspaceLabel = getWorkspaceLabel(),
                    queryStrip = queryStrip,
                    queryLower = queryStrip.lowercase(),
                    isChinese = isChineseText(queryStrip),
                    indexName = getFulltextIndexName(getWorkspaceLabel()),
                    limit = limit,
                )

            driver!!.session(sessionConfig()).use { neoSession: Session ->
                try {
                    val (cypherQuery, params) = buildFullTextQuery(context)
                    runLabelQuery(neoSession, cypherQuery, params)
                } catch (e: Neo4jException) {
                    logger.warn {
                        "[$workspace] Full-text search failed: ${e.message}. Falling back to standard search."
                    }
                    val (fallbackQuery, fallbackParams) = buildFallbackQuery(context)
                    runLabelQuery(neoSession, fallbackQuery, fallbackParams)
                }
            }
        }

    private data class LabelSearchContext(
        val workspaceLabel: String,
        val queryStrip: String,
        val queryLower: String,
        val isChinese: Boolean,
        val indexName: String,
        val limit: Int,
    )

    private fun buildFullTextQuery(context: LabelSearchContext): Pair<String, MutableMap<String, Any>> {
        val params: MutableMap<String, Any> =
            mutableMapOf(
                "index_name" to context.indexName,
                "query_strip" to context.queryStrip,
                "query_lower" to context.queryLower,
                "limit" to context.limit,
            )
        val cypherQuery =
            if (context.isChinese) {
                params["search_query"] = context.queryStrip
                buildChineseFulltextQuery(context.workspaceLabel)
            } else {
                params["search_query"] = "${context.queryStrip}*"
                buildNonChineseFulltextQuery(context.workspaceLabel)
            }
        return cypherQuery to params
    }

    private fun buildChineseFulltextQuery(workspaceLabel: String): String =
        """
        CALL db.index.fulltext.queryNodes(${'$'}index_name, ${'$'}search_query) YIELD node, score
        WITH node, score
        WHERE node:`$workspaceLabel`
        WITH node.entity_id AS label, score
        WITH label, score,
             CASE
                 WHEN label = ${'$'}query_strip THEN score + $SCORE_EXACT_MATCH
                 WHEN label CONTAINS ${'$'}query_strip THEN score + $SCORE_CONTAINS
                 ELSE score
             END AS final_score
        RETURN label
        ORDER BY final_score DESC, label ASC
        LIMIT ${'$'}limit
        """.trimIndent()

    private fun buildNonChineseFulltextQuery(workspaceLabel: String): String =
        """
        CALL db.index.fulltext.queryNodes(${'$'}index_name, ${'$'}search_query) YIELD node, score
        WITH node, score
        WHERE node:`$workspaceLabel`
        WITH node.entity_id AS label, toLower(node.entity_id) AS label_lower, score
        WITH label, label_lower, score,
             CASE
                 WHEN label_lower = ${'$'}query_lower THEN score + $SCORE_EXACT_MATCH
                 WHEN label_lower STARTS WITH ${'$'}query_lower THEN score + $SCORE_STARTS_WITH
                 WHEN label_lower CONTAINS ' ' + ${'$'}query_lower OR label_lower CONTAINS '_' + ${'$'}query_lower THEN score + $SCORE_PARTIAL
                 ELSE score
             END AS final_score
        RETURN label
        ORDER BY final_score DESC, label ASC
        LIMIT ${'$'}limit
        """.trimIndent()

    private fun buildFallbackQuery(context: LabelSearchContext): Pair<String, MutableMap<String, Any>> {
        val params: MutableMap<String, Any> = mutableMapOf("limit" to context.limit)
        val cypherQuery =
            if (context.isChinese) {
                params["query_strip"] = context.queryStrip
                """
                MATCH (n:`${context.workspaceLabel}`)
                WHERE n.entity_id IS NOT NULL
                WITH n.entity_id AS label
                WHERE label CONTAINS ${'$'}query_strip
                WITH label,
                     CASE
                         WHEN label = ${'$'}query_strip THEN $SCORE_EXACT_MATCH
                         WHEN label STARTS WITH ${'$'}query_strip THEN $SCORE_STARTS_WITH
                         ELSE $FALLBACK_SCORE_BASE - size(label)
                     END AS score
                ORDER BY score DESC, label ASC
                LIMIT ${'$'}limit
                RETURN label
                """.trimIndent()
            } else {
                params["query_lower"] = context.queryLower
                """
                MATCH (n:`${context.workspaceLabel}`)
                WHERE n.entity_id IS NOT NULL
                WITH n.entity_id AS label, toLower(n.entity_id) AS label_lower
                WHERE label_lower CONTAINS ${'$'}query_lower
                WITH label, label_lower,
                     CASE
                         WHEN label_lower = ${'$'}query_lower THEN $SCORE_EXACT_MATCH
                         WHEN label_lower STARTS WITH ${'$'}query_lower THEN $SCORE_STARTS_WITH
                         ELSE $FALLBACK_SCORE_BASE - size(label)
                     END AS score
                ORDER BY score DESC, label ASC
                LIMIT ${'$'}limit
                RETURN label
                """.trimIndent()
            }
        return cypherQuery to params
    }

    private fun runLabelQuery(
        neoSession: Session,
        cypherQuery: String,
        params: MutableMap<String, Any>,
    ): List<String> = neoSession.runLogged(cypherQuery, params).list().map { it.get("label").asString() }
}
