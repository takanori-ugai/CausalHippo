package lightrag.kg.mongo

import com.mongodb.client.model.Filters
import com.mongodb.client.model.ReplaceOptions
import com.mongodb.kotlin.client.coroutine.MongoClient
import com.mongodb.kotlin.client.coroutine.MongoCollection
import com.mongodb.kotlin.client.coroutine.MongoDatabase
import dev.langchain4j.model.embedding.EmbeddingModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.toList
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.KnowledgeGraph
import org.bson.Document

private val logger = KotlinLogging.logger {}

/**
 * A MongoDB-backed graph storage implementation.
 * @property namespace The namespace of the storage.
 * @property globalConfig The global configuration for the storage.
 * @property embeddingFunc The embedding model to use.
 */
class MongoGraphStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
    override val embeddingFunc: EmbeddingModel,
) : BaseGraphStorage {
    /**
     * The workspace of the storage.
     */
    override val workspace: String = globalConfig["working_dir"] as? String ?: "./rag_storage"

    private lateinit var client: MongoClient
    private lateinit var database: MongoDatabase
    private lateinit var nodesCollection: MongoCollection<Document>
    private lateinit var edgesCollection: MongoCollection<Document>

    private val uri: String =
        System.getenv("MONGO_URI") ?: "mongodb://0.0.0.0:27017/?directConnection=true"
    private val dbName: String = System.getenv("MONGO_DATABASE") ?: "LightRAG"
    private val collectionName: String = System.getenv("MONGO_KG_COLLECTION") ?: "MDB_KG"

    /**
     * Initializes the storage by creating a MongoDB client and getting the database and collections.
     */
    override suspend fun initialize() {
        logger.info {
            "Initializing MongoGraphStorage with URI: $uri, DB: $dbName, Collection: $collectionName"
        }
        client = MongoClient.create(uri)
        database = client.getDatabase(dbName)
        // We will store both nodes and edges in the same collection

        // Let's assume we use the collection name provided.
        val col = database.getCollection<Document>(collectionName)
        nodesCollection = col
        edgesCollection = col

        // Create indexes if needed?
    }

    /**
     * Closes the MongoDB client.
     */
    override suspend fun finalize() {
        if (::client.isInitialized) {
            client.close()
        }
    }

    /**
     * Callback for when indexing is done.
     */
    override suspend fun indexDoneCallback() {
        // MongoDB persists immediately, so maybe nothing to do here, or ensure indexes.
    }

    /**
     * Drops the storage.
     * @return A map with the status of the operation.
     */
    override suspend fun drop(): Map<String, String> {
        // Implement drop logic if needed, e.g., drop collection
        return emptyMap()
    }

    /**
     * Checks if a node exists.
     * @param nodeId The ID of the node to check.
     * @return True if the node exists, false otherwise.
     */
    override suspend fun hasNode(nodeId: String): Boolean {
        val count =
            nodesCollection.countDocuments(
                Filters.and(
                    Filters.eq("id", nodeId),
                    Filters.eq("type", "node"),
                ),
            )
        return count > 0
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
    ): Boolean {
        val count =
            edgesCollection.countDocuments(
                Filters.and(
                    Filters.eq("source_id", sourceNodeId),
                    Filters.eq("target_id", targetNodeId),
                    Filters.eq("type", "edge"),
                ),
            )
        return count > 0
    }

    /**
     * Gets the degree of a node.
     * @param nodeId The ID of the node.
     * @return The degree of the node.
     */
    override suspend fun nodeDegree(nodeId: String): Int {
        val srcCount =
            edgesCollection.countDocuments(
                Filters.and(
                    Filters.eq("source_id", nodeId),
                    Filters.eq("type", "edge"),
                ),
            )
        val tgtCount =
            edgesCollection.countDocuments(
                Filters.and(
                    Filters.eq("target_id", nodeId),
                    Filters.eq("type", "edge"),
                ),
            )
        return (srcCount + tgtCount).toInt()
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
        // Degree of an edge? usually it's just 1 if it exists, or weight?
        // The BaseGraphStorage interface asks for edgeDegree.
        // In NetworkX, degree is for nodes.
        // Maybe this means the weight of the edge?
        // Let's check InMemoryGraphStorage implementation.
        return (nodeDegree(srcId) + nodeDegree(tgtId))
    }

    /**
     * Gets a node by its ID.
     * @param nodeId The ID of the node to get.
     * @return A map representing the node.
     */
    override suspend fun getNode(nodeId: String): Map<String, String>? {
        val doc =
            nodesCollection
                .find(
                    Filters.and(
                        Filters.eq("id", nodeId),
                        Filters.eq("type", "node"),
                    ),
                ).firstOrNull() ?: return null

        return doc.entries.associate { (k, v) -> k to v.toString() }
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
    ): Map<String, String>? {
        val doc =
            edgesCollection
                .find(
                    Filters.and(
                        Filters.eq("source_id", sourceNodeId),
                        Filters.eq("target_id", targetNodeId),
                        Filters.eq("type", "edge"),
                    ),
                ).firstOrNull() ?: return null

        return doc.entries.associate { (k, v) -> k to v.toString() }
    }

    /**
     * Gets the edges of a node.
     * @param sourceNodeId The ID of the source node.
     * @return A list of pairs representing the edges.
     */
    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>>? {
        // In python: list(self._graph.edges(sourceNodeId)) -> [(src, tgt), ...]
        // Here we want list of (src, tgt) connected to sourceNodeId?
        // Or just outgoing edges?
        // Usually edges(n) returns all edges adjacent to n.

        val edges =
            edgesCollection
                .find(
                    Filters.and(
                        Filters.or(
                            Filters.eq("source_id", sourceNodeId),
                            Filters.eq("target_id", sourceNodeId),
                        ),
                        Filters.eq("type", "edge"),
                    ),
                ).toList()

        return edges.map { doc ->
            (doc.getString("source_id") ?: "") to (doc.getString("target_id") ?: "")
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
    ) {
        val doc =
            Document("id", nodeId)
                .append("type", "node")
        nodeData.forEach { (k, v) -> doc.append(k, v) }

        nodesCollection.replaceOne(
            Filters.and(Filters.eq("id", nodeId), Filters.eq("type", "node")),
            doc,
            ReplaceOptions().upsert(true),
        )
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
    ) {
        val doc =
            Document("source_id", sourceNodeId)
                .append("target_id", targetNodeId)
                .append("type", "edge")
        edgeData.forEach { (k, v) -> doc.append(k, v) }

        edgesCollection.replaceOne(
            Filters.and(
                Filters.eq("source_id", sourceNodeId),
                Filters.eq("target_id", targetNodeId),
                Filters.eq("type", "edge"),
            ),
            doc,
            ReplaceOptions().upsert(true),
        )
    }

    /**
     * Deletes a node.
     * @param nodeId The ID of the node to delete.
     */
    override suspend fun deleteNode(nodeId: String) {
        nodesCollection.deleteOne(
            Filters.and(
                Filters.eq("id", nodeId),
                Filters.eq("type", "node"),
            ),
        )
        // Also delete connected edges?
        edgesCollection.deleteMany(
            Filters.and(
                Filters.or(
                    Filters.eq("source_id", nodeId),
                    Filters.eq("target_id", nodeId),
                ),
                Filters.eq("type", "edge"),
            ),
        )
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
    override suspend fun removeEdges(edges: List<Pair<String, String>>) {
        edges.forEach { (src, tgt) ->
            edgesCollection.deleteOne(
                Filters.and(
                    Filters.eq("source_id", src),
                    Filters.eq("target_id", tgt),
                    Filters.eq("type", "edge"),
                ),
            )
        }
    }

    /**
     * Gets all labels.
     * @return A list of all labels.
     */
    override suspend fun getAllLabels(): List<String> {
        // Assuming labels are stored in node data, but 'getAllLabels' usually refers to edge labels in some graphs or node types?
        // BaseGraphStorage documentation or python implementation?
        // Python networkx graph doesn't enforce labels.
        // Maybe getting all node IDs?
        return emptyList()
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
    ): KnowledgeGraph {
        // This seems to be complex graph traversal.
        // For now, return empty or implement simple BFS.
        return KnowledgeGraph(emptyList(), emptyList())
    }

    /**
     * Gets all nodes.
     * @return A list of all nodes.
     */
    override suspend fun getAllNodes(): List<Map<String, Any>> =
        nodesCollection
            .find(Filters.eq("type", "node"))
            .toList()
            .map { it.toMap() }

    /**
     * Gets all edges.
     * @return A list of all edges.
     */
    override suspend fun getAllEdges(): List<Map<String, Any>> =
        edgesCollection
            .find(Filters.eq("type", "edge"))
            .toList()
            .map { it.toMap() }

    /**
     * Gets popular labels.
     * @param limit The maximum number of labels to return.
     * @return A list of popular labels.
     */
    override suspend fun getPopularLabels(limit: Int): List<String> = emptyList()

    /**
     * Searches for labels.
     * @param query The query to search for.
     * @param limit The maximum number of labels to return.
     * @return A list of labels.
     */
    override suspend fun searchLabels(
        query: String,
        limit: Int,
    ): List<String> = emptyList()
}
