package lightrag.kg.memory

import dev.langchain4j.model.embedding.EmbeddingModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.KnowledgeGraph
import java.io.File

private val logger = KotlinLogging.logger {}

/**
 * An in-memory graph storage implementation.
 * @property namespace The namespace of the storage.
 * @property workspace The workspace of the storage.
 * @property globalConfig The global configuration for the storage.
 * @property embeddingFunc The embedding model to use.
 */
class InMemoryGraphStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
) : BaseGraphStorage {
    private val nodes = mutableMapOf<String, Map<String, String>>()
    private val edges = mutableMapOf<String, MutableMap<String, Map<String, String>>>()
    private val workingDir = File(globalConfig["working_dir"] as? String ?: "./rag_storage")
    private val file = File(workingDir, "graph_$namespace.json")
    private val json =
        Json {
            ignoreUnknownKeys = true
            prettyPrint = true
        }

    /**
     * Initializes the storage by loading data from the JSON file.
     */
    override suspend fun initialize() {
        if (!workingDir.exists()) {
            workingDir.mkdirs()
        }
        if (!file.exists()) return
        runCatching {
            val content = file.readText()
            if (content.isBlank()) return@runCatching
            val parsed = json.decodeFromString<JsonObject>(content)

            val loadedNodes =
                parsed["nodes"]
                    ?.jsonObject
                    ?.mapValues { (_, value) ->
                        value.jsonObject.mapValues { entry -> entry.value.jsonPrimitive.content }
                    } ?: emptyMap()
            val loadedEdges =
                parsed["edges"]
                    ?.jsonObject
                    ?.mapValues { (_, targets) ->
                        targets.jsonObject
                            .mapValues { (_, data) ->
                                data.jsonObject.mapValues { entry -> entry.value.jsonPrimitive.content }
                            }.toMutableMap()
                    }?.toMutableMap() ?: mutableMapOf()
            nodes.putAll(loadedNodes)
            edges.putAll(loadedEdges)
            logger.info {
                "Loaded graph '$namespace' with ${nodes.size} nodes and ${edges.size} edge buckets from ${file.absolutePath}"
            }
        }.onFailure { logger.error(it) { "Error loading graph storage from ${file.absolutePath}" } }
    }

    /**
     * Saves the current state of the storage to the JSON file.
     */
    override suspend fun indexDoneCallback() {
        runCatching {
            if (!workingDir.exists()) {
                workingDir.mkdirs()
            }
            val nodesObj = JsonObject(nodes.mapValues { it.value.toJsonElement() as JsonObject })
            val edgesObj =
                JsonObject(
                    edges.mapValues { (_, targets) ->
                        JsonObject(targets.mapValues { (_, data) -> data.toJsonElement() as JsonObject })
                    },
                )
            val payload =
                JsonObject(
                    mapOf(
                        "nodes" to nodesObj,
                        "edges" to edgesObj,
                    ),
                )
            val content = json.encodeToString(payload)
            file.writeText(content)
            logger.debug { "Persisted graph '$namespace' with ${nodes.size} nodes to ${file.absolutePath}" }
        }.onFailure { logger.error(it) { "Error saving graph storage to ${file.absolutePath}" } }
    }

    /**
     * Drops the storage.
     * @return A map with the status of the operation.
     */
    override suspend fun drop(): Map<String, String> {
        nodes.clear()
        edges.clear()
        if (file.exists()) {
            runCatching { file.delete() }.onFailure {
                logger.warn(it) { "Failed to delete graph file ${file.absolutePath}" }
            }
        }
        return mapOf("status" to "success", "message" to "data dropped")
    }

    /**
     * Checks if a node exists.
     * @param nodeId The ID of the node to check.
     * @return True if the node exists, false otherwise.
     */
    override suspend fun hasNode(nodeId: String): Boolean = nodes.containsKey(nodeId)

    /**
     * Checks if an edge exists.
     * @param sourceNodeId The ID of the source node.
     * @param targetNodeId The ID of the target node.
     * @return True if the edge exists, false otherwise.
     */
    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = edges[sourceNodeId]?.containsKey(targetNodeId) == true

    /**
     * Gets the degree of a node.
     * @param nodeId The ID of the node.
     * @return The degree of the node.
     */
    override suspend fun nodeDegree(nodeId: String): Int {
        return (edges[nodeId]?.size ?: 0) // Simplified directed degree
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
    ): Int = nodeDegree(srcId) + nodeDegree(tgtId)

    /**
     * Gets a node by its ID.
     * @param nodeId The ID of the node to get.
     * @return A map representing the node.
     */
    override suspend fun getNode(nodeId: String): Map<String, String>? = nodes[nodeId]

    /**
     * Gets an edge by its source and target node IDs.
     * @param sourceNodeId The ID of the source node.
     * @param targetNodeId The ID of the target node.
     * @return A map representing the edge.
     */
    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, String>? = edges[sourceNodeId]?.get(targetNodeId)

    /**
     * Gets the edges of a node.
     * @param sourceNodeId The ID of the source node.
     * @return A list of pairs representing the edges.
     */
    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>>? {
        if (!nodes.containsKey(sourceNodeId)) return null
        return edges[sourceNodeId]?.keys?.map { sourceNodeId to it } ?: emptyList()
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
        nodes[nodeId] = nodeData
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
        edges.computeIfAbsent(sourceNodeId) { mutableMapOf() }[targetNodeId] = edgeData
        // Undirected graph usually stores both ways or handles it logic wise.
        // BaseGraphStorage says "All operations related to edges in graph should be undirected."
        // So we should add the reverse edge too.
        edges.computeIfAbsent(targetNodeId) { mutableMapOf() }[sourceNodeId] = edgeData
    }

    /**
     * Deletes a node.
     * @param nodeId The ID of the node to delete.
     */
    override suspend fun deleteNode(nodeId: String) {
        nodes.remove(nodeId)
        edges.remove(nodeId)
        edges.values.forEach { it.remove(nodeId) }
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
            this.edges[src]?.remove(tgt)
            this.edges[tgt]?.remove(src)
        }
    }

    /**
     * Gets all labels.
     * @return A list of all labels.
     */
    override suspend fun getAllLabels(): List<String> = nodes.keys.toList().sorted()

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
        if (maxNodes <= 0) return KnowledgeGraph(emptyList(), emptyList())

        val nodesResult = mutableListOf<Map<String, Any>>()
        val edgesResult = mutableListOf<Map<String, Any>>()
        val visitedNodes = mutableSetOf<String>()
        val queuedNodes = mutableSetOf<String>()
        val visitedEdges = mutableSetOf<Pair<String, String>>() // normalized undirected pairs
        var isTruncated = false

        val queue = ArrayDeque<Pair<String, Int>>() // nodeId, depth

        fun enqueueStart(nodeId: String) {
            if (nodeId !in queuedNodes && queuedNodes.size < maxNodes) {
                queue.add(nodeId to 0)
                queuedNodes.add(nodeId)
            } else if (nodeId !in queuedNodes) {
                isTruncated = true
            }
        }

        if (nodeLabel == "*") {
            nodes.keys.sorted().forEach { enqueueStart(it) }
        } else if (nodes.containsKey(nodeLabel)) {
            enqueueStart(nodeLabel)
        } else {
            return KnowledgeGraph(emptyList(), emptyList())
        }

        while (queue.isNotEmpty() && visitedNodes.size < maxNodes) {
            val (currentId, depth) = queue.removeFirst()
            val currentData = nodes[currentId] ?: emptyMap()

            if (currentId in visitedNodes || depth > maxDepth) continue

            val nodeMap =
                mapOf(
                    "id" to currentId,
                    "labels" to listOf(currentId),
                    "properties" to (if (currentData.containsKey("entity_id")) currentData else currentData + ("entity_id" to currentId)),
                )
            nodesResult.add(nodeMap)
            visitedNodes.add(currentId)

            if (depth == maxDepth || visitedNodes.size >= maxNodes) {
                if (queue.isNotEmpty()) isTruncated = true
                continue
            }

            val neighbors = edges[currentId] ?: emptyMap()
            for ((neighborId, edgeData) in neighbors) {
                val normalized =
                    if (currentId < neighborId) currentId to neighborId else neighborId to currentId

                if (neighborId !in queuedNodes && depth + 1 <= maxDepth) {
                    if (queuedNodes.size < maxNodes) {
                        queue.add(neighborId to depth + 1)
                        queuedNodes.add(neighborId)
                    } else {
                        isTruncated = true
                    }
                }

                if (neighborId in queuedNodes && visitedEdges.add(normalized)) {
                    val edgeType = edgeData["type"] ?: edgeData["relationship"] ?: "RELATED"
                    val edgeMap =
                        mapOf(
                            "id" to "${normalized.first}--${normalized.second}",
                            "type" to edgeType,
                            "source" to currentId,
                            "target" to neighborId,
                            "properties" to edgeData,
                        )
                    edgesResult.add(edgeMap)
                }
            }
        }

        if (visitedNodes.size >= maxNodes && (queue.isNotEmpty() || nodes.size > visitedNodes.size)) {
            isTruncated = true
        }

        return KnowledgeGraph(nodesResult, edgesResult, isTruncated)
    }

    /**
     * Gets all nodes.
     * @return A list of all nodes.
     */
    override suspend fun getAllNodes(): List<Map<String, Any>> = nodes.values.map { it as Map<String, Any> }

    /**
     * Gets all edges.
     * @return A list of all edges.
     */
    override suspend fun getAllEdges(): List<Map<String, Any>> {
        val result = mutableListOf<Map<String, Any>>()
        val seen = mutableSetOf<Pair<String, String>>()

        edges.forEach { (src, targets) ->
            targets.forEach { (tgt, data) ->
                val key = if (src < tgt) src to tgt else tgt to src
                if (key !in seen) {
                    seen.add(key)
                    result.add(data + mapOf("source" to src, "target" to tgt))
                }
            }
        }
        return result
    }

    /**
     * Gets popular labels.
     * @param limit The maximum number of labels to return.
     * @return A list of popular labels.
     */
    override suspend fun getPopularLabels(limit: Int): List<String> {
        // nodeDegree is suspend function, so we need to map first then sort
        val degrees = nodes.keys.associateWith { nodeDegree(it) }
        return nodes.keys.sortedByDescending { degrees[it] }.take(limit)
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
    ): List<String> = nodes.keys.filter { it.contains(query, ignoreCase = true) }.take(limit)

    private fun Any?.toJsonElement(): JsonElement =
        when (this) {
            null -> JsonNull

            is Boolean -> JsonPrimitive(this)

            is Number -> JsonPrimitive(this.toString())

            // Convert number to string for JsonPrimitive
            is String -> JsonPrimitive(this)

            is List<*> -> JsonArray(this.map { it.toJsonElement() })

            is Map<*, *> -> JsonObject(this.entries.associate { it.key.toString() to it.value.toJsonElement() })

            else -> JsonPrimitive(this.toString())
        }
}
