package shared.rag.spi.persistence

/**
 * Graph persistence namespace abstraction.
 */
interface GraphStore {
    fun upsertNode(
        id: String,
        data: Map<String, Any?> = emptyMap(),
    )

    fun upsertEdge(
        source: String,
        target: String,
        data: Map<String, Any?> = emptyMap(),
    )

    fun deleteNode(id: String)

    fun deleteEdge(
        source: String,
        target: String,
    )

    fun nodes(): List<GraphNodeRecord>

    fun edges(): List<GraphEdgeRecord>

    fun snapshot(): GraphSnapshot

    fun restore(snapshot: GraphSnapshot)
}

data class GraphNodeRecord(
    val id: String,
    val data: Map<String, Any?> = emptyMap(),
)

data class GraphEdgeRecord(
    val source: String,
    val target: String,
    val data: Map<String, Any?> = emptyMap(),
)

data class GraphSnapshot(
    val nodes: List<GraphNodeRecord> = emptyList(),
    val edges: List<GraphEdgeRecord> = emptyList(),
    val metadata: Map<String, Any?> = emptyMap(),
)
