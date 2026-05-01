package lightrag.core

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager
import lightrag.utils.computeMd5
import shared.rag.CommonRag
import java.io.File

private val logger =
    io.github.oshai.kotlinlogging.KotlinLogging
        .logger {}

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
) : CommonRag<QueryParam, QueryResult?> {
    private val objectMapper = jacksonObjectMapper()

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
    }

    /**
     * Rebuilds the derived storage if it is empty.
     */
    suspend fun rebuildDerivedStorageIfEmpty() {
        ingestionService.rebuildDerivedStorageIfEmpty()
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
    ): QueryResult? = queryService.query(query, param)

    override fun drop() = runBlocking { adrop() }

    override suspend fun adrop() {
        storageManager.drop()
    }

    override fun inspectGraph(): Map<String, Any?> = runBlocking { ainspectGraph() }

    override suspend fun ainspectGraph(): Map<String, Any?> {
        val graph = storageManager.chunkEntityRelationGraph
        val nodes = graph.getAllNodes()
        val edges = graph.getAllEdges()
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

    override fun saveGraph(path: String) = runBlocking { asaveGraph(path) }

    @Suppress("TooGenericExceptionCaught")
    override suspend fun asaveGraph(path: String) {
        val payload = ainspectGraph()
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

    override fun loadGraph(path: String) = runBlocking { aloadGraph(path) }

    @Suppress("TooGenericExceptionCaught", "UNCHECKED_CAST")
    override suspend fun aloadGraph(path: String) {
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

        storageManager.chunkEntityRelationGraph.drop()
        storageManager.entitiesVdb.drop()
        storageManager.relationshipsVdb.drop()
        storageManager.fullEntities.drop()
        storageManager.fullRelations.drop()

        nodesRaw.forEach { rawNode ->
            val nodeMap = rawNode as? Map<*, *> ?: return@forEach
            val nodeId =
                listOf("id", "entity_id", "entity_name", "_id")
                    .firstNotNullOfOrNull { key -> nodeMap[key]?.toString()?.takeIf { it.isNotBlank() } }
                    ?: return@forEach

            val nodeData =
                nodeMap
                    .entries
                    .associate { (key, value) -> key.toString() to value.toString() }
                    .toMutableMap()
            nodeData.remove("id")
            nodeData.remove("_id")
            if (nodeData["entity_id"].isNullOrBlank()) {
                nodeData["entity_id"] = nodeId
            }

            storageManager.chunkEntityRelationGraph.upsertNode(nodeId, nodeData)
            storageManager.fullEntities.upsert(mapOf(nodeId to nodeData))

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
            val edgeMap = rawEdge as? Map<*, *> ?: return@forEach
            val src =
                listOf("source", "src_id", "source_id")
                    .firstNotNullOfOrNull { key -> edgeMap[key]?.toString()?.takeIf { it.isNotBlank() } }
                    ?: return@forEach
            val tgt =
                listOf("target", "tgt_id", "target_id")
                    .firstNotNullOfOrNull { key -> edgeMap[key]?.toString()?.takeIf { it.isNotBlank() } }
                    ?: return@forEach

            val edgeData =
                edgeMap
                    .entries
                    .associate { (key, value) -> key.toString() to value.toString() }
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
            edgeData["src_id"] = src
            edgeData["tgt_id"] = tgt

            storageManager.chunkEntityRelationGraph.upsertEdge(src, tgt, edgeData)

            val edgeKey = "$src#$tgt"
            storageManager.fullRelations.upsert(mapOf(edgeKey to edgeData))
            val relContent = "${edgeData["keywords"].orEmpty()}\t$src\n$tgt\n${edgeData["description"].orEmpty()}"
            storageManager.relationshipsVdb.upsert(
                mapOf(
                    computeMd5(edgeKey) to
                        mapOf(
                            "content" to relContent,
                            "src_id" to src,
                            "tgt_id" to tgt,
                        ),
                ),
            )
        }

        logger.info {
            "Loaded LightRAG graph snapshot from '$path' with ${nodesRaw.size} nodes and ${edgesRaw.size} edges."
        }
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
}
