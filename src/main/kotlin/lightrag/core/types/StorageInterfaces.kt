package lightrag.core.types

import dev.langchain4j.model.embedding.EmbeddingModel
import kotlinx.serialization.Contextual
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Base interface for storage with a namespace and workspace.
 * @property namespace The namespace of the storage.
 * @property workspace The workspace of the storage.
 * @property globalConfig The global configuration for the storage.
 */
interface StorageNameSpace {
    val namespace: String
    val workspace: String
    val globalConfig: Map<String, Any?>

    /**
     * Initializes the storage.
     */
    suspend fun initialize() {}

    /**
     * Finalizes the storage.
     */
    suspend fun finalize() {}

    /**
     * Callback for when indexing is done.
     */
    suspend fun indexDoneCallback()

    /**
     * Drops the storage.
     * @return A map with the status of the operation.
     */
    suspend fun drop(): Map<String, String>
}

/**
 * Base interface for a vector storage.
 * @property embeddingFunc The embedding model to use.
 * @property cosineBetterThanThreshold The threshold for cosine similarity.
 * @property metaFields The set of meta fields.
 */
interface BaseVectorStorage : StorageNameSpace {
    val embeddingFunc: EmbeddingModel
    val cosineBetterThanThreshold: Double
    val metaFields: Set<String>

    /**
     * Queries the vector storage.
     * @param query The query string.
     * @param topK The number of top results to return.
     * @param queryEmbedding The query embedding.
     * @return A list of maps representing the results.
     */
    suspend fun query(
        query: String,
        topK: Int,
        queryEmbedding: List<Float>? = null,
    ): List<Map<String, Any>>

    /**
     * Upserts data into the vector storage.
     * @param data The data to upsert.
     */
    suspend fun upsert(data: Map<String, Map<String, Any>>)

    /**
     * Deletes an entity from the vector storage.
     * @param entityName The name of the entity to delete.
     */
    suspend fun deleteEntity(entityName: String)

    /**
     * Deletes an entity relation from the vector storage.
     * @param entityName The name of the entity relation to delete.
     */
    suspend fun deleteEntityRelation(entityName: String)

    /**
     * Gets an item by its ID.
     * @param id The ID of the item to get.
     * @return A map representing the item.
     */
    suspend fun getById(id: String): Map<String, Any>?

    /**
     * Gets items by their IDs.
     * @param ids The IDs of the items to get.
     * @return A list of maps representing the items.
     */
    suspend fun getByIds(ids: List<String>): List<Map<String, Any>>

    /**
     * Deletes items by their IDs.
     * @param ids The IDs of the items to delete.
     */
    suspend fun delete(ids: List<String>)

    /**
     * Gets vectors by their IDs.
     * @param ids The IDs of the vectors to get.
     * @return A map of IDs to vectors.
     */
    suspend fun getVectorsByIds(ids: List<String>): Map<String, List<Float>>
}

/**
 * Base interface for a key-value storage.
 * @property embeddingFunc The embedding model to use.
 */
interface BaseKVStorage : StorageNameSpace {
    val embeddingFunc: EmbeddingModel

    /**
     * Gets an item by its ID.
     * @param id The ID of the item to get.
     * @return A map representing the item.
     */
    suspend fun getById(id: String): Map<String, Any>?

    /**
     * Gets items by their IDs.
     * @param ids The IDs of the items to get.
     * @return A list of maps representing the items.
     */
    suspend fun getByIds(ids: List<String>): List<Map<String, Any>>

    /**
     * Filters keys from the storage.
     * @param keys The keys to filter.
     * @return A set of the filtered keys.
     */
    suspend fun filterKeys(keys: Set<String>): Set<String>

    /**
     * Upserts data into the storage.
     * @param data The data to upsert.
     */
    suspend fun upsert(data: Map<String, Map<String, Any>>)

    /**
     * Deletes items by their IDs.
     * @param ids The IDs of the items to delete.
     */
    suspend fun delete(ids: List<String>)

    /**
     * Checks if the storage is empty.
     * @return True if the storage is empty, false otherwise.
     */
    suspend fun isEmpty(): Boolean
}

/**
 * Base interface for a graph storage.
 * @property embeddingFunc The embedding model to use.
 */
interface BaseGraphStorage : StorageNameSpace {
    val embeddingFunc: EmbeddingModel

    /**
     * Checks if a node exists.
     * @param nodeId The ID of the node to check.
     * @return True if the node exists, false otherwise.
     */
    suspend fun hasNode(nodeId: String): Boolean

    /**
     * Checks if an edge exists.
     * @param sourceNodeId The ID of the source node.
     * @param targetNodeId The ID of the target node.
     * @return True if the edge exists, false otherwise.
     */
    suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean

    /**
     * Gets the degree of a node.
     * @param nodeId The ID of the node.
     * @return The degree of the node.
     */
    suspend fun nodeDegree(nodeId: String): Int

    /**
     * Gets the degree of an edge.
     * @param srcId The ID of the source node.
     * @param tgtId The ID of the target node.
     * @return The degree of the edge.
     */
    suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int

    /**
     * Gets a node by its ID.
     * @param nodeId The ID of the node to get.
     * @return A map representing the node.
     */
    suspend fun getNode(nodeId: String): Map<String, String>?

    /**
     * Gets an edge by its source and target node IDs.
     * @param sourceNodeId The ID of the source node.
     * @param targetNodeId The ID of the target node.
     * @return A map representing the edge.
     */
    suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, String>?

    /**
     * Gets the edges of a node.
     * @param sourceNodeId The ID of the source node.
     * @return A list of pairs representing the edges.
     */
    suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>>?

    /**
     * Gets a batch of nodes by their IDs.
     * @param nodeIds The IDs of the nodes to get.
     * @return A map of node IDs to nodes.
     */
    suspend fun getNodesBatch(nodeIds: List<String>): Map<String, Map<String, String>> {
        val result = mutableMapOf<String, Map<String, String>>()
        for (nodeId in nodeIds) {
            getNode(nodeId)?.let { result[nodeId] = it }
        }
        return result
    }

    /**
     * Gets the degrees of a batch of nodes.
     * @param nodeIds The IDs of the nodes.
     * @return A map of node IDs to degrees.
     */
    suspend fun nodeDegreesBatch(nodeIds: List<String>): Map<String, Int> {
        val result = mutableMapOf<String, Int>()
        for (nodeId in nodeIds) {
            result[nodeId] = nodeDegree(nodeId)
        }
        return result
    }

    /**
     * Gets the degrees of a batch of edges.
     * @param edgePairs The pairs of source and target node IDs.
     * @return A map of edge pairs to degrees.
     */
    suspend fun edgeDegreesBatch(edgePairs: List<Pair<String, String>>): Map<Pair<String, String>, Int> {
        val result = mutableMapOf<Pair<String, String>, Int>()
        for ((srcId, tgtId) in edgePairs) {
            result[srcId to tgtId] = edgeDegree(srcId, tgtId)
        }
        return result
    }

    /**
     * Gets a batch of edges.
     * @param pairs The pairs of source and target node IDs.
     * @return A map of edge pairs to edges.
     */
    suspend fun getEdgesBatch(pairs: List<Map<String, String>>): Map<Pair<String, String>, Map<String, String>> {
        val result = mutableMapOf<Pair<String, String>, Map<String, String>>()
        for (pair in pairs) {
            val srcId = pair["src"] ?: continue
            val tgtId = pair["tgt"] ?: continue
            getEdge(srcId, tgtId)?.let { result[srcId to tgtId] = it }
        }
        return result
    }

    /**
     * Gets the edges of a batch of nodes.
     * @param nodeIds The IDs of the nodes.
     * @return A map of node IDs to lists of edges.
     */
    suspend fun getNodesEdgesBatch(nodeIds: List<String>): Map<String, List<Pair<String, String>>> {
        val result = mutableMapOf<String, List<Pair<String, String>>>()
        for (nodeId in nodeIds) {
            getNodeEdges(nodeId)?.let { result[nodeId] = it }
        }
        return result
    }

    /**
     * Upserts a node.
     * @param nodeId The ID of the node.
     * @param nodeData The data of the node.
     */
    suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, String>,
    )

    /**
     * Upserts an edge.
     * @param sourceNodeId The ID of the source node.
     * @param targetNodeId The ID of the target node.
     * @param edgeData The data of the edge.
     */
    suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, String>,
    )

    /**
     * Deletes a node.
     * @param nodeId The ID of the node to delete.
     */
    suspend fun deleteNode(nodeId: String)

    /**
     * Removes nodes.
     * @param nodes The nodes to remove.
     */
    suspend fun removeNodes(nodes: List<String>)

    /**
     * Removes edges.
     * @param edges The edges to remove.
     */
    suspend fun removeEdges(edges: List<Pair<String, String>>)

    /**
     * Gets all labels.
     * @return A list of all labels.
     */
    suspend fun getAllLabels(): List<String>

    /**
     * Gets the knowledge graph.
     * @param nodeLabel The label of the node.
     * @param maxDepth The maximum depth to traverse.
     * @param maxNodes The maximum number of nodes to return.
     * @return The knowledge graph.
     */
    suspend fun getKnowledgeGraph(
        nodeLabel: String,
        maxDepth: Int = 3,
        maxNodes: Int = 1000,
    ): KnowledgeGraph

    /**
     * Gets all nodes.
     * @return A list of all nodes.
     */
    suspend fun getAllNodes(): List<Map<String, Any>>

    /**
     * Gets all edges.
     * @return A list of all edges.
     */
    suspend fun getAllEdges(): List<Map<String, Any>>

    /**
     * Gets popular labels.
     * @param limit The maximum number of labels to return.
     * @return A list of popular labels.
     */
    suspend fun getPopularLabels(limit: Int = 300): List<String>

    /**
     * Searches for labels.
     * @param query The query to search for.
     * @param limit The maximum number of labels to return.
     * @return A list of labels.
     */
    suspend fun searchLabels(
        query: String,
        limit: Int = 50,
    ): List<String>
}

/**
 * Represents a knowledge graph.
 * @property nodes The nodes of the graph.
 * @property edges The edges of the graph.
 * @property isTruncated Whether the graph is truncated.
 */
@Serializable
data class KnowledgeGraph(
    val nodes: List<Map<String, @Contextual Any>>,
    val edges: List<Map<String, @Contextual Any>>,
    val isTruncated: Boolean = false,
)

/**
 * Represents the status of a document.
 */
enum class DocStatus(
    val value: String,
) {
    /**
     * The document is pending processing.
     */
    PENDING("pending"),

    /**
     * The document is currently being processed.
     */
    PROCESSING("processing"),

    /**
     * The document has been preprocessed.
     */
    PREPROCESSED("preprocessed"),

    /**
     * The document has been processed.
     */
    PROCESSED("processed"),

    /**
     * Processing of the document has failed.
     */
    FAILED("failed"),
}

/**
 * Represents the processing status of a document.
 * @property contentSummary The summary of the content.
 * @property contentLength The length of the content.
 * @property filePath The path of the file.
 * @property status The status of the document.
 * @property createdAt The creation date of the document.
 * @property updatedAt The last update date of the document.
 * @property trackId The track ID of the document.
 * @property chunksCount The number of chunks.
 * @property chunksList The list of chunks.
 * @property errorMsg The error message.
 * @property metadata The metadata of the document.
 * @property multimodalProcessed Whether the document has been processed for multimodal content.
 */
@Serializable
data class DocProcessingStatus(
    @SerialName("content_summary")
    val contentSummary: String,
    @SerialName("content_length")
    val contentLength: Int,
    @SerialName("file_path")
    val filePath: String,
    val status: DocStatus,
    @SerialName("created_at")
    val createdAt: String,
    @SerialName("updated_at")
    val updatedAt: String,
    @SerialName("track_id")
    val trackId: String? = null,
    @SerialName("chunks_count")
    val chunksCount: Int? = null,
    @SerialName("chunks_list")
    val chunksList: List<String>? = null,
    @SerialName("error_msg")
    val errorMsg: String? = null,
    // Simplified to String map for now
    val metadata: Map<String, String> = emptyMap(),
    @SerialName("multimodal_processed")
    val multimodalProcessed: Boolean? = null,
)

/**
 * Interface for a document status storage.
 */
interface DocStatusStorage : BaseKVStorage {
    /**
     * Gets the status counts.
     * @return A map of status counts.
     */
    suspend fun getStatusCounts(): Map<String, Int>

    /**
     * Gets documents by their status.
     * @param status The status of the documents to get.
     * @return A map of document IDs to document processing statuses.
     */
    suspend fun getDocsByStatus(status: DocStatus): Map<String, DocProcessingStatus>

    /**
     * Gets documents by their track ID.
     * @param trackId The track ID of the documents to get.
     * @return A map of document IDs to document processing statuses.
     */
    suspend fun getDocsByTrackId(trackId: String): Map<String, DocProcessingStatus>

    /**
     * Gets documents with pagination.
     * @param statusFilter The status to filter by.
     * @param page The page number.
     * @param pageSize The size of the page.
     * @param sortField The field to sort by.
     * @param sortDirection The direction to sort by.
     * @return A pair of the list of documents and the total number of documents.
     */
    suspend fun getDocsPaginated(
        statusFilter: DocStatus? = null,
        page: Int = 1,
        pageSize: Int = 50,
        sortField: String = "updated_at",
        sortDirection: String = "desc",
    ): Pair<List<Pair<String, DocProcessingStatus>>, Int>

    /**
     * Gets all status counts.
     * @return A map of all status counts.
     */
    suspend fun getAllStatusCounts(): Map<String, Int>

    /**
     * Gets a document by its file path.
     * @param filePath The file path of the document to get.
     * @return A map representing the document.
     */
    suspend fun getDocByFilePath(filePath: String): Map<String, Any>?
}
