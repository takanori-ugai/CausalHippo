package ragas.testset.graph

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.util.UUID

/** Logical type for nodes stored in a [KnowledgeGraph]. */
enum class NodeType {
    /** Node type is unknown or not assigned yet. */
    UNKNOWN,

    /** Node represents an original source document. */
    DOCUMENT,

    /** Node represents a chunk derived from a document. */
    CHUNK,
}

/**
 * Graph node with normalized, case-insensitive string properties.
 *
 * @property id Stable node identifier.
 * @property properties Mutable key/value properties stored with lowercase keys.
 * @property type Node category used by graph transforms.
 */
@Serializable
data class Node(
    val id: String = UUID.randomUUID().toString(),
    // Mutable by design; data-class copy() is shallow for this map.
    val properties: MutableMap<String, String> = mutableMapOf(),
    val type: NodeType = NodeType.UNKNOWN,
) {
    init {
        if (properties.isNotEmpty()) {
            val normalized =
                properties.entries.groupBy { (key, _) -> key.lowercase() }
            val duplicates = normalized.filterValues { it.size > 1 }.keys
            require(duplicates.isEmpty()) {
                "Duplicate property keys when normalized to lowercase: ${duplicates.joinToString(", ")}"
            }
            properties.clear()
            properties.putAll(normalized.mapValues { (_, entries) -> entries.first().value })
        }
    }

    /**
     * Adds a new property key/value pair, rejecting duplicate keys (case-insensitive).
     *
     * @param key Property key.
     * @param value Property value.
     */
    fun addProperty(
        key: String,
        value: String,
    ) {
        val normalized = key.lowercase()
        require(normalized !in properties) { "Property $key already exists" }
        properties[normalized] = value
    }

    /**
     * Returns a property value by key using case-insensitive lookup.
     *
     * @param key Property key to read.
     * @return Matched property value, or null when the key is absent.
     */
    fun getProperty(key: String): String? = properties[key.lowercase()]
}

/**
 * Directed edge between two nodes.
 *
 * @property id Stable relationship identifier.
 * @property type Relationship label.
 * @property sourceId Source node ID.
 * @property targetId Target node ID.
 * @property bidirectional Whether this edge should be treated as bidirectional.
 * @property properties Optional edge metadata.
 */
@Serializable
data class Relationship(
    val id: String = UUID.randomUUID().toString(),
    val type: String,
    val sourceId: String,
    val targetId: String,
    val bidirectional: Boolean = false,
    val properties: Map<String, String> = emptyMap(),
)

@Serializable
private data class SerializableKnowledgeGraph(
    val nodes: List<Node>,
    val relationships: List<Relationship>,
)

/**
 * In-memory graph container with JSON persistence helpers.
 *
 * @property nodes Node collection.
 * @property relationships Relationship collection.
 */
class KnowledgeGraph(
    val nodes: MutableList<Node> = mutableListOf(),
    val relationships: MutableList<Relationship> = mutableListOf(),
) {
    /**
     * Appends a node to the graph.
     *
     * @param node Node to append.
     */
    fun add(node: Node) {
        nodes += node
    }

    /**
     * Appends a relationship to the graph.
     *
     * @param relationship Relationship to append.
     */
    fun add(relationship: Relationship) {
        relationships += relationship
    }

    /**
     * Returns the first node with the provided ID, or null when no node matches.
     *
     * @param nodeId Node identifier to search.
     * @return Matching node, or null when not found.
     */
    fun getNodeById(nodeId: String): Node? = nodes.firstOrNull { node -> node.id == nodeId }

    /**
     * Persists this graph to [path] as JSON.
     *
     * @param path Destination file path.
     */
    fun save(path: String) {
        val file = File(path)
        file.parentFile?.mkdirs()
        val payload = SerializableKnowledgeGraph(nodes = nodes, relationships = relationships)
        file.writeText(graphJson.encodeToString(payload))
    }

    /** Returns a compact summary including node and relationship counts. */
    override fun toString(): String = "KnowledgeGraph(nodes=${nodes.size}, relationships=${relationships.size})"

    companion object {
        /**
         * Loads a knowledge graph from a JSON file produced by [save].
         *
         * @param path Source file path.
         * @return Deserialized [KnowledgeGraph].
         */
        fun load(path: String): KnowledgeGraph {
            val file = File(path)
            require(file.exists()) { "Knowledge graph file not found: $path" }
            val payload = graphJson.decodeFromString<SerializableKnowledgeGraph>(file.readText())
            return KnowledgeGraph(
                nodes = payload.nodes.toMutableList(),
                relationships = payload.relationships.toMutableList(),
            )
        }
    }
}

private val graphJson =
    Json {
        prettyPrint = true
        ignoreUnknownKeys = true
    }
