package lightrag.core

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import dev.langchain4j.model.embedding.EmbeddingModel
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager
import lightrag.utils.computeMd5
import shared.rag.CommonRag
import shared.rag.spi.persistence.GraphEdgeRecord
import shared.rag.spi.persistence.GraphNodeRecord
import shared.rag.spi.persistence.GraphSnapshot
import shared.rag.spi.persistence.KvSnapshot
import shared.rag.spi.persistence.PersistenceSession
import shared.rag.spi.persistence.VectorRecord
import shared.rag.spi.persistence.VectorSnapshot
import java.io.File
import java.time.Instant

private val logger =
    io.github.oshai.kotlinlogging.KotlinLogging
        .logger {}
private val wordRegex = Regex("\\w+")

/**
 * Parameters for a query.
 * @property mode The query mode.
 * @property onlyNeedContext Whether to return only the context.
 * @property onlyNeedPrompt Whether to return only the prompt.
 * @property responseType The desired response type.
 * @property stream Whether to stream the response.
 * @property topK The number of top results to return.
 * @property chunkTopK The number of top chunks to return.
 * @property maxEntityTokens The maximum number of tokens for entities.
 * @property maxRelationTokens The maximum number of tokens for relations.
 * @property maxTotalTokens The maximum total number of tokens.
 * @property hlKeywords Keywords to highlight.
 * @property llKeywords Keywords to lowlight.
 * @property conversationHistory The conversation history.
 * @property userPrompt The user prompt.
 * @property enableRerank Whether to enable reranking.
 * @property includeReferences Whether to include references.
 */
@Serializable
data class QueryParam(
    val mode: String = "global",
    @SerialName("only_need_context")
    val onlyNeedContext: Boolean = false,
    @SerialName("only_need_prompt")
    val onlyNeedPrompt: Boolean = false,
    @SerialName("response_type")
    val responseType: String? = "Multiple Paragraphs",
    val stream: Boolean = false,
    @SerialName("top_k")
    val topK: Int = 40,
    @SerialName("chunk_top_k")
    val chunkTopK: Int = 20,
    @SerialName("max_entity_tokens")
    val maxEntityTokens: Int = 6000,
    @SerialName("max_relation_tokens")
    val maxRelationTokens: Int = 8000,
    @SerialName("max_total_tokens")
    val maxTotalTokens: Int = 30000,
    @SerialName("hl_keywords")
    var hlKeywords: List<String> = emptyList(),
    @SerialName("ll_keywords")
    var llKeywords: List<String> = emptyList(),
    @SerialName("conversation_history")
    val conversationHistory: List<Map<String, String>> = emptyList(),
    @SerialName("user_prompt")
    val userPrompt: String? = null,
    @SerialName("enable_rerank")
    val enableRerank: Boolean = true,
    @SerialName("include_references")
    val includeReferences: Boolean = false,
)

class LightRAG(
    private val ingestionService: IngestionService,
    private val queryService: QueryService,
    val storageManager: StorageManager,
    private val embeddingModelForSpi: EmbeddingModel? = null,
    private val persistenceSession: PersistenceSession? = null,
    private val useUnifiedSpiForRetrievalAndIndex: Boolean = false,
    private val persistenceNamespacePrefix: String = "lightrag",
) : CommonRag<QueryParam, QueryResult?> {
    private val objectMapper = jacksonObjectMapper()

    private val unifiedSpiSession = persistenceSession
    private val unifiedSpiEnabled = useUnifiedSpiForRetrievalAndIndex && unifiedSpiSession != null
    private val unifiedSpiNamespacePrefix = persistenceNamespacePrefix.trim().ifEmpty { "lightrag" }
    private val unifiedSpiGraphNamespace = "${unifiedSpiNamespacePrefix}_graph"
    private val unifiedSpiVectorNamespace = "${unifiedSpiNamespacePrefix}_vector"
    private val unifiedSpiChunksNamespace = "${unifiedSpiNamespacePrefix}_chunks"
    private val unifiedSpiMetadataNamespace = "${unifiedSpiNamespacePrefix}_metadata"
    private var unifiedSpiInitialized = false

    init {
        if (unifiedSpiEnabled) {
            runBlocking { ensureUnifiedSpiIndexInitializedInternal() }
        }
    }

    /**
     * Inserts a single document.
     * @param input The document to insert.
     * @param fileSource The source of the file.
     * @return A track ID for the insertion.
     */
    suspend fun insert(
        input: String,
        fileSource: String? = null,
    ): String = ingestionService.insert(input, fileSource)

    /**
     * Inserts multiple documents.
     * @param input The documents to insert.
     * @param fileSources The sources of the files.
     * @return A track ID for the insertion.
     */
    suspend fun insert(
        input: List<String>,
        fileSources: List<String>? = null,
    ): String = ingestionService.insert(input, fileSources)

    override fun upsert(data: String) = runBlocking { aupsert(data) }

    override fun upsert(data: Collection<String>) = runBlocking { aupsert(data) }

    override suspend fun aupsert(data: String) {
        val doc = data.trim()
        if (doc.isBlank()) {
            logger.warn { "No valid String documents found in upsert payload." }
            return
        }
        insert(doc)
        if (unifiedSpiEnabled) {
            syncUnifiedSpiFromLocal()
            unifiedSpiInitialized = true
        }
    }

    override suspend fun aupsert(data: Collection<String>) {
        val docs =
            data
                .map { it.trim() }
                .filter { it.isNotBlank() }
        if (docs.isEmpty()) {
            logger.warn { "No valid String documents found in upsert payload." }
            return
        }
        insert(docs)
        if (unifiedSpiEnabled) {
            syncUnifiedSpiFromLocal()
            unifiedSpiInitialized = true
        }
    }

    /**
     * Rebuilds the derived storage if it is empty.
     */
    suspend fun rebuildDerivedStorageIfEmpty() {
        ingestionService.rebuildDerivedStorageIfEmpty()
        if (unifiedSpiEnabled) {
            syncUnifiedSpiFromLocal()
            unifiedSpiInitialized = true
        }
    }

    /**
     * Queries the LightRAG system.
     * @param query The query to execute.
     * @param param The query parameters.
     * @return The query result.
     */
    override fun query(
        query: String,
        param: QueryParam,
    ): QueryResult? = runBlocking { aquery(query, param) }

    override suspend fun aquery(
        query: String,
        param: QueryParam,
    ): QueryResult? {
        if (unifiedSpiEnabled) {
            ensureUnifiedSpiIndexInitializedInternal()
            if (param.onlyNeedContext) {
                val contextItems = retrieveContextFromUnifiedSpi(query, param.chunkTopK.coerceAtLeast(1))
                return QueryResult(
                    content = contextItems.joinToString(separator = "\n\n"),
                    rawData = mapOf("data" to mapOf("context" to contextItems)),
                )
            }
        }
        return queryService.query(query, param)
    }

    override fun drop() = runBlocking { adrop() }

    override suspend fun adrop() {
        storageManager.drop()
        if (unifiedSpiEnabled) {
            clearUnifiedSpiState()
            unifiedSpiInitialized = true
        }
    }

    override fun inspectGraph(): Map<String, Any?> = runBlocking { ainspectGraph() }

    override suspend fun ainspectGraph(): Map<String, Any?> =
        if (unifiedSpiEnabled) {
            inspectGraphFromUnifiedSpi()
        } else {
            inspectGraphFromLocal()
        }

    override fun saveGraph(path: String) = runBlocking { asaveGraph(path) }

    @Suppress("TooGenericExceptionCaught")
    override suspend fun asaveGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        if (unifiedSpiEnabled) {
            saveUnifiedSpiCheckpoint(path)
            return
        }
        saveLocalSnapshot(path)
    }

    override fun loadGraph(path: String) = runBlocking { aloadGraph(path) }

    @Suppress("TooGenericExceptionCaught", "UNCHECKED_CAST")
    override suspend fun aloadGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        if (unifiedSpiEnabled) {
            loadUnifiedSpiCheckpoint(path)
            return
        }
        loadLocalSnapshot(path)
    }

    /**
     * Gets the processing status of the documents.
     * @return A map of the status counts.
     */
    suspend fun getProcessingStatus(): Map<String, Int> = ingestionService.getProcessingStatus()

    /**
     * Deletes a document by its ID.
     * @param docId The ID of the document to delete.
     * @return A map of the status.
     */
    suspend fun deleteByDocId(docId: String): Map<String, String> = ingestionService.deleteByDocId(docId)

    private suspend fun saveLocalSnapshot(path: String) {
        val payload = inspectGraphFromLocal()
        val metadata = payload["metadata"] as? Map<*, *> ?: emptyMap<Any?, Any?>()
        val nodeCount = metadata["nodeCount"] as? Int ?: 0
        val edgeCount = metadata["edgeCount"] as? Int ?: 0

        val output = File(path)
        output.parentFile?.mkdirs()
        try {
            output.writeText(objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(payload))
            logger.info {
                "Saved LightRAG graph snapshot to '$path' with $nodeCount nodes and $edgeCount edges."
            }
        } catch (e: Exception) {
            logger.error(e) { "Failed to save LightRAG graph snapshot to '$path'." }
            throw e
        }
    }

    @Suppress("UNCHECKED_CAST")
    private suspend fun loadLocalSnapshot(path: String) {
        val input = File(path)
        require(input.exists()) { "Graph snapshot file not found: $path" }

        val payloadType = object : TypeReference<Map<String, Any?>>() {}
        val payload =
            try {
                objectMapper.readValue(input, payloadType)
            } catch (e: Exception) {
                logger.error(e) { "Failed to parse LightRAG graph snapshot from '$path'." }
                throw e
            }

        val nodesRaw = payload["nodes"] as? List<*> ?: emptyList<Any?>()
        val edgesRaw = payload["edges"] as? List<*> ?: emptyList<Any?>()

        resetLocalDerivedState()

        nodesRaw.forEach { rawNode ->
            val nodeMap = asStringMap(rawNode)
            val nodeId = nodeIdentifier(nodeMap) ?: return@forEach
            val nodeData = nodeStorageData(nodeMap, nodeId)

            storageManager.chunkEntityRelationGraph.upsertNode(nodeId, nodeData)
            storageManager.fullEntities.upsert(mapOf(nodeId to nodeData.mapValues { it.value as Any }))

            val entityDescription = nodeData["description"].orEmpty()
            val entityVdbData =
                mapOf(
                    computeMd5(nodeId) to
                        mapOf(
                            "content" to "$nodeId\n$entityDescription",
                            "entity_name" to nodeId,
                        ),
                )
            storageManager.entitiesVdb.upsert(entityVdbData)
        }

        edgesRaw.forEach { rawEdge ->
            val edgeMap = asStringMap(rawEdge)
            val source = edgeSource(edgeMap) ?: return@forEach
            val target = edgeTarget(edgeMap) ?: return@forEach
            val edgeData = edgeStorageData(edgeMap, source, target)

            storageManager.chunkEntityRelationGraph.upsertEdge(source, target, edgeData)

            val edgeKey = "$source#$target"
            storageManager.fullRelations.upsert(mapOf(edgeKey to edgeData.mapValues { it.value as Any }))
            val relContent =
                "${edgeData["keywords"].orEmpty()}\t$source\n$target\n${edgeData["description"].orEmpty()}"
            storageManager.relationshipsVdb.upsert(
                mapOf(
                    computeMd5(edgeKey) to
                        mapOf(
                            "content" to relContent,
                            "src_id" to source,
                            "tgt_id" to target,
                        ),
                ),
            )
        }

        logger.info {
            "Loaded LightRAG graph snapshot from '$path' with ${nodesRaw.size} nodes and ${edgesRaw.size} edges."
        }
    }

    private suspend fun saveUnifiedSpiCheckpoint(path: String) {
        val session = unifiedSpiSession ?: return
        syncUnifiedSpiFromLocal()
        session.checkpoint(path)
        session.kv(unifiedSpiMetadataNamespace).put("lastCheckpointPath", path)
    }

    private suspend fun loadUnifiedSpiCheckpoint(path: String) {
        val session = unifiedSpiSession ?: return
        session.restore(path)
        hydrateLocalIndexFromUnifiedSpi()
        unifiedSpiInitialized = true
    }

    private suspend fun ensureUnifiedSpiIndexInitializedInternal() {
        if (!unifiedSpiEnabled || unifiedSpiInitialized) return
        val session = unifiedSpiSession ?: return
        val graphSnapshot = session.graph(unifiedSpiGraphNamespace).snapshot()
        val chunksSnapshot = session.kv(unifiedSpiChunksNamespace).snapshot()
        val vectorSnapshot = session.vector(unifiedSpiVectorNamespace).snapshot()
        val hasSpiState =
            graphSnapshot.nodes.isNotEmpty() ||
                graphSnapshot.edges.isNotEmpty() ||
                chunksSnapshot.entries.isNotEmpty() ||
                vectorSnapshot.records.isNotEmpty()
        if (hasSpiState) {
            hydrateLocalIndexFromUnifiedSpi()
            unifiedSpiInitialized = true
            return
        }

        val localHasGraph = storageManager.chunkEntityRelationGraph.getAllNodes().isNotEmpty()
        val localHasChunks = !storageManager.textChunks.isEmpty()
        if (localHasGraph || localHasChunks) {
            syncUnifiedSpiFromLocal()
        }
        unifiedSpiInitialized = true
    }

    private fun clearUnifiedSpiState() {
        val session = unifiedSpiSession ?: return
        session.graph(unifiedSpiGraphNamespace).restore(GraphSnapshot())
        session.vector(unifiedSpiVectorNamespace).restore(VectorSnapshot(metric = "cosine"))
        session.kv(unifiedSpiChunksNamespace).restore(KvSnapshot())
        session.kv(unifiedSpiMetadataNamespace).restore(KvSnapshot())
    }

    private suspend fun syncUnifiedSpiFromLocal() {
        val session = unifiedSpiSession ?: return

        val localNodes = storageManager.chunkEntityRelationGraph.getAllNodes()
        val localEdges = storageManager.chunkEntityRelationGraph.getAllEdges()

        val graphNodes =
            localNodes.mapNotNull { raw ->
                val row = asStringMap(raw)
                val id = nodeIdentifier(row) ?: return@mapNotNull null
                GraphNodeRecord(id = id, data = row + mapOf("entity_id" to id))
            }
        val graphEdges =
            localEdges.mapNotNull { raw ->
                val row = asStringMap(raw)
                val source = edgeSource(row) ?: return@mapNotNull null
                val target = edgeTarget(row) ?: return@mapNotNull null
                GraphEdgeRecord(
                    source = source,
                    target = target,
                    data =
                        row +
                            mapOf(
                                "src_id" to source,
                                "tgt_id" to target,
                            ),
                )
            }

        session.graph(unifiedSpiGraphNamespace).restore(
            GraphSnapshot(
                nodes = graphNodes,
                edges = graphEdges,
                metadata =
                    mapOf(
                        "nodeCount" to graphNodes.size,
                        "edgeCount" to graphEdges.size,
                    ),
            ),
        )

        val chunkIds = collectChunkIds(graphNodes, graphEdges)
        val chunkEntries =
            buildMap<String, Any?> {
                for (chunkId in chunkIds) {
                    val chunk = storageManager.textChunks.getById(chunkId) ?: continue
                    put(chunkId, chunk)
                }
            }
        session.kv(unifiedSpiChunksNamespace).restore(
            KvSnapshot(
                entries = chunkEntries,
                metadata = mapOf("chunkCount" to chunkEntries.size),
            ),
        )

        val vectorRecords = buildUnifiedSpiVectorRecords(graphNodes, graphEdges, chunkEntries)
        val dimensions = vectorRecords.firstOrNull()?.vector?.size
        session
            .vector(unifiedSpiVectorNamespace, dimensions = dimensions, metric = "cosine")
            .restore(
                VectorSnapshot(
                    dimensions = dimensions,
                    metric = "cosine",
                    records = vectorRecords,
                    metadata = mapOf("recordCount" to vectorRecords.size),
                ),
            )

        session.kv(unifiedSpiMetadataNamespace).putAll(
            mapOf(
                "updatedAt" to Instant.now().toString(),
                "recordCount" to vectorRecords.size,
                "chunkCount" to chunkEntries.size,
                "nodeCount" to graphNodes.size,
                "edgeCount" to graphEdges.size,
            ),
        )
    }

    private suspend fun hydrateLocalIndexFromUnifiedSpi() {
        val session = unifiedSpiSession ?: return
        val graphSnapshot = session.graph(unifiedSpiGraphNamespace).snapshot()
        val chunkSnapshot = session.kv(unifiedSpiChunksNamespace).snapshot()
        val vectorSnapshot = session.vector(unifiedSpiVectorNamespace).snapshot()

        resetLocalDerivedState()

        graphSnapshot.nodes.forEach { node ->
            val nodeData = nodeStorageData(asStringMap(node.data), node.id)
            storageManager.chunkEntityRelationGraph.upsertNode(node.id, nodeData)
            storageManager.fullEntities.upsert(mapOf(node.id to nodeData.mapValues { it.value as Any }))
        }
        graphSnapshot.edges.forEach { edge ->
            val edgeData = edgeStorageData(asStringMap(edge.data), edge.source, edge.target)
            storageManager.chunkEntityRelationGraph.upsertEdge(edge.source, edge.target, edgeData)
            val edgeKey = "${edge.source}#${edge.target}"
            storageManager.fullRelations.upsert(mapOf(edgeKey to edgeData.mapValues { it.value as Any }))
        }

        val chunkUpserts =
            chunkSnapshot.entries
                .mapNotNull { (chunkId, payload) ->
                    val map = asStringMap(payload)
                    if (map.isEmpty()) {
                        null
                    } else {
                        chunkId to map.mapValues { it.value as Any }
                    }
                }.toMap()
        if (chunkUpserts.isNotEmpty()) {
            storageManager.textChunks.upsert(chunkUpserts)
        }

        val entityVectors =
            vectorSnapshot.records
                .filter { it.metadata["kind"]?.toString() == "entity" }
                .mapNotNull { record ->
                    val entityName =
                        record.metadata["entity_name"]?.toString()?.takeIf { it.isNotBlank() } ?: return@mapNotNull null
                    val content = record.metadata["content"]?.toString() ?: "$entityName\n"
                    record.id to
                        mapOf(
                            "content" to content,
                            "entity_name" to entityName,
                            "vector" to record.vector.map { it.toFloat() },
                        )
                }.toMap()
        if (entityVectors.isNotEmpty()) {
            storageManager.entitiesVdb.upsert(entityVectors)
        }

        val relationVectors =
            vectorSnapshot.records
                .filter { it.metadata["kind"]?.toString() == "relation" }
                .mapNotNull { record ->
                    val src = record.metadata["src_id"]?.toString()?.takeIf { it.isNotBlank() } ?: return@mapNotNull null
                    val tgt = record.metadata["tgt_id"]?.toString()?.takeIf { it.isNotBlank() } ?: return@mapNotNull null
                    val content = record.metadata["content"]?.toString() ?: "$src\n$tgt\n"
                    record.id to
                        mapOf(
                            "content" to content,
                            "src_id" to src,
                            "tgt_id" to tgt,
                            "vector" to record.vector.map { it.toFloat() },
                        )
                }.toMap()
        if (relationVectors.isNotEmpty()) {
            storageManager.relationshipsVdb.upsert(relationVectors)
        }

        val chunkVectors =
            vectorSnapshot.records
                .filter { it.metadata["kind"]?.toString() == "chunk" }
                .mapNotNull { record ->
                    val chunkId = record.metadata["chunk_id"]?.toString()?.takeIf { it.isNotBlank() } ?: return@mapNotNull null
                    val content =
                        record.metadata["content"]?.toString()?.takeIf { it.isNotBlank() } ?: return@mapNotNull null
                    chunkId to
                        mapOf(
                            "content" to content,
                            "full_doc_id" to (record.metadata["full_doc_id"]?.toString() ?: ""),
                            "file_path" to (record.metadata["file_path"]?.toString() ?: "unknown_source"),
                            "vector" to record.vector.map { it.toFloat() },
                        )
                }.toMap()
        val resolvedChunkVectors =
            if (chunkVectors.isNotEmpty()) {
                chunkVectors
            } else {
                chunkUpserts.mapValues { (_, row) ->
                    mapOf(
                        "content" to (row["content"]?.toString() ?: ""),
                        "full_doc_id" to (row["full_doc_id"]?.toString() ?: ""),
                        "file_path" to (row["file_path"]?.toString() ?: "unknown_source"),
                    )
                }
            }
        if (resolvedChunkVectors.isNotEmpty()) {
            storageManager.chunksVdb.upsert(resolvedChunkVectors)
        }
    }

    private suspend fun retrieveContextFromUnifiedSpi(
        query: String,
        topK: Int,
    ): List<String> {
        val session = unifiedSpiSession ?: return emptyList()
        val chunkSnapshot = session.kv(unifiedSpiChunksNamespace).snapshot()
        val lexicalCandidates =
            chunkSnapshot.entries
                .mapNotNull { (_, payload) ->
                    asStringMap(payload)["content"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
                }

        if (lexicalCandidates.isNotEmpty()) {
            return lexicalCandidates
                .map { candidate -> candidate to lexicalMatchScore(query, candidate) }
                .sortedByDescending { it.second }
                .take(topK.coerceAtLeast(1))
                .map { it.first }
        }

        val queryEmbedding = embedForSpi(query) ?: return emptyList()
        return try {
            session
                .vector(unifiedSpiVectorNamespace, dimensions = queryEmbedding.size, metric = "cosine")
                .query(queryEmbedding, topK = topK.coerceAtLeast(1))
                .mapNotNull { record ->
                    record.metadata["content"]?.toString()?.trim()?.takeIf { it.isNotBlank() }
                }
        } catch (ex: RuntimeException) {
            logger.warn(ex) { "Unified SPI vector retrieval failed for LightRAG." }
            emptyList()
        }
    }

    private suspend fun resetLocalDerivedState() {
        storageManager.chunkEntityRelationGraph.drop()
        storageManager.entitiesVdb.drop()
        storageManager.relationshipsVdb.drop()
        storageManager.chunksVdb.drop()
        storageManager.fullEntities.drop()
        storageManager.fullRelations.drop()
        storageManager.textChunks.drop()
    }

    private suspend fun inspectGraphFromLocal(): Map<String, Any?> {
        val graph = storageManager.chunkEntityRelationGraph
        val nodes = graph.getAllNodes().mapNotNull { raw -> normalizeNodeForInspect(raw) }
        val edges = graph.getAllEdges().mapNotNull { raw -> normalizeEdgeForInspect(raw) }
        return mapOf(
            "nodes" to nodes,
            "edges" to edges,
            "metadata" to
                mapOf(
                    "nodeCount" to nodes.size,
                    "edgeCount" to edges.size,
                ),
        )
    }

    private fun inspectGraphFromUnifiedSpi(): Map<String, Any?> {
        val session = unifiedSpiSession ?: return emptyMap()
        val snapshot = session.graph(unifiedSpiGraphNamespace).snapshot()
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
        return mapOf(
            "nodes" to nodes,
            "edges" to edges,
            "metadata" to
                mapOf(
                    "nodeCount" to nodes.size,
                    "edgeCount" to edges.size,
                    "persistenceBackend" to session.backendId,
                ),
        )
    }

    private fun buildUnifiedSpiVectorRecords(
        nodes: List<GraphNodeRecord>,
        edges: List<GraphEdgeRecord>,
        chunkEntries: Map<String, Any?>,
    ): List<VectorRecord> {
        val records = mutableListOf<VectorRecord>()

        nodes.forEach { node ->
            val description = node.data["description"]?.toString().orEmpty()
            val content = "${node.id}\n$description".trim()
            val vector = embedForSpi(content) ?: return@forEach
            records +=
                VectorRecord(
                    id = "entity:${node.id}",
                    vector = vector,
                    metadata =
                        mapOf(
                            "kind" to "entity",
                            "entity_name" to node.id,
                            "content" to content,
                        ),
                )
        }

        edges.forEach { edge ->
            val keywords = edge.data["keywords"]?.toString().orEmpty()
            val description = edge.data["description"]?.toString().orEmpty()
            val content = "$keywords\t${edge.source}\n${edge.target}\n$description".trim()
            val vector = embedForSpi(content) ?: return@forEach
            records +=
                VectorRecord(
                    id = "relation:${edge.source}#${edge.target}",
                    vector = vector,
                    metadata =
                        mapOf(
                            "kind" to "relation",
                            "src_id" to edge.source,
                            "tgt_id" to edge.target,
                            "content" to content,
                        ),
                )
        }

        chunkEntries.forEach { (chunkId, payload) ->
            val row = asStringMap(payload)
            val content = row["content"]?.toString()?.trim().orEmpty()
            if (content.isBlank()) return@forEach
            val vector = embedForSpi(content) ?: return@forEach
            records +=
                VectorRecord(
                    id = "chunk:$chunkId",
                    vector = vector,
                    metadata =
                        mapOf(
                            "kind" to "chunk",
                            "chunk_id" to chunkId,
                            "content" to content,
                            "full_doc_id" to row["full_doc_id"]?.toString().orEmpty(),
                            "file_path" to row["file_path"]?.toString().orEmpty(),
                        ),
                )
        }

        return records
    }

    private fun collectChunkIds(
        nodes: List<GraphNodeRecord>,
        edges: List<GraphEdgeRecord>,
    ): Set<String> {
        val ids = mutableSetOf<String>()
        nodes.forEach { node ->
            ids += chunkIdsFromSource(node.data["source_id"]?.toString())
        }
        edges.forEach { edge ->
            ids += chunkIdsFromSource(edge.data["source_id"]?.toString())
        }
        return ids
    }

    private fun chunkIdsFromSource(sourceId: String?): Set<String> =
        sourceId
            ?.split(Constants.GRAPH_FIELD_SEP)
            ?.map { it.trim() }
            ?.filter { it.isNotBlank() }
            ?.toSet()
            ?: emptySet()

    private fun nodeIdentifier(row: Map<String, Any?>): String? =
        listOf("id", "entity_id", "entity_name", "_id")
            .firstNotNullOfOrNull { key ->
                row[key]?.toString()?.trim()?.takeIf { it.isNotBlank() }
            }

    private fun edgeSource(row: Map<String, Any?>): String? =
        listOf("source", "src_id", "source_id")
            .firstNotNullOfOrNull { key ->
                row[key]?.toString()?.trim()?.takeIf { it.isNotBlank() }
            }

    private fun edgeTarget(row: Map<String, Any?>): String? =
        listOf("target", "tgt_id", "target_id")
            .firstNotNullOfOrNull { key ->
                row[key]?.toString()?.trim()?.takeIf { it.isNotBlank() }
            }

    private fun nodeStorageData(
        row: Map<String, Any?>,
        nodeId: String,
    ): MutableMap<String, String> {
        val nodeData =
            row
                .entries
                .associate { (key, value) -> key to value.toString() }
                .toMutableMap()
        nodeData.remove("id")
        nodeData.remove("_id")
        if (nodeData["entity_id"].isNullOrBlank()) {
            nodeData["entity_id"] = nodeId
        }
        return nodeData
    }

    private fun edgeStorageData(
        row: Map<String, Any?>,
        source: String,
        target: String,
    ): MutableMap<String, String> {
        val edgeData =
            row
                .entries
                .associate { (key, value) -> key to value.toString() }
                .toMutableMap()
        edgeData.remove("id")
        edgeData.remove("_id")
        edgeData.remove("type")
        edgeData.remove("source")
        edgeData.remove("target")
        edgeData.remove("source_id")
        edgeData.remove("target_id")
        edgeData.remove("src_id")
        edgeData.remove("tgt_id")
        edgeData["src_id"] = source
        edgeData["tgt_id"] = target
        edgeData.putIfAbsent("source_id", "")
        return edgeData
    }

    private fun normalizeNodeForInspect(raw: Any?): Map<String, Any?>? {
        val row = asStringMap(raw)
        val id = nodeIdentifier(row) ?: return null
        return linkedMapOf<String, Any?>("id" to id).apply { putAll(row) }
    }

    private fun normalizeEdgeForInspect(raw: Any?): Map<String, Any?>? {
        val row = asStringMap(raw)
        val source = edgeSource(row) ?: return null
        val target = edgeTarget(row) ?: return null
        return linkedMapOf<String, Any?>(
            "source" to source,
            "target" to target,
        ).apply { putAll(row) }
    }

    private fun asStringMap(value: Any?): Map<String, Any?> {
        val map = value as? Map<*, *> ?: return emptyMap()
        return map.entries.associate { (k, v) -> k.toString() to v }
    }

    private fun embedForSpi(text: String): List<Double>? {
        val model = embeddingModelForSpi ?: storageManager.chunksVdb.embeddingFunc
        return try {
            model.embed(text).content().vector().map { it.toDouble() }
        } catch (ex: RuntimeException) {
            logger.warn(ex) { "Failed to embed text for LightRAG unified SPI indexing/retrieval." }
            null
        }
    }

    private fun lexicalMatchScore(
        query: String,
        candidate: String,
    ): Double {
        val queryTokens = wordRegex.findAll(query.lowercase()).map { it.value }.toSet()
        if (queryTokens.isEmpty()) return 0.0
        val candidateTokens = wordRegex.findAll(candidate.lowercase()).map { it.value }.toSet()
        if (candidateTokens.isEmpty()) return 0.0
        return queryTokens.intersect(candidateTokens).size.toDouble() / queryTokens.size.toDouble()
    }
}
