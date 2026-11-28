package pathrag

import pathrag.base.BaseKVStorage
import shared.rag.spi.persistence.GraphEdgeRecord
import shared.rag.spi.persistence.GraphNodeRecord
import shared.rag.spi.persistence.GraphSnapshot

internal typealias CheckpointRows = Map<String, Map<String, Any>>

internal suspend fun BaseKVStorage<Map<String, Any>>.checkpointRows(): CheckpointRows {
    val keys = allKeys()
    val rows = getByIds(keys)
    check(rows.size == keys.size)
    return keys.zip(rows).associate { (key, row) -> key to checkNotNull(row) { "Missing checkpoint record '$key'" } }
}

internal data class CorpusCheckpoint(
    val chunks: CheckpointRows,
    val documents: CheckpointRows,
    val extraction: CheckpointRows,
)

internal data class GraphCheckpoint(
    val graph: GraphSnapshot,
    val corpus: CorpusCheckpoint?,
)

private fun objectFields(value: Any?): Map<String, Any?> {
    require(value is Map<*, *>) { "Expected a checkpoint object" }
    return value.entries.associate { (key, field) ->
        require(key is String && key.isNotBlank()) { "Invalid checkpoint key" }
        key to field
    }
}

internal fun checkpointRows(value: Any?): CheckpointRows =
    objectFields(value).mapValues { (_, row) -> objectFields(row).mapValues { (_, field) -> requireNotNull(field) } }

internal fun parseGraphCheckpoint(payload: Map<String, Any?>): GraphCheckpoint {
    val nodes = objectFields(payload["nodes"]).map { (id, data) -> GraphNodeRecord(id, objectFields(data)) }
    val edges = payload["edges"]
    require(edges is List<*>) { "Missing checkpoint edges" }
    val graph =
        GraphSnapshot(
            nodes,
            edges.map { row ->
                val fields = objectFields(row)
                val source = fields["source"]
                val target = fields["target"]
                require(source is String && source.isNotBlank() && target is String && target.isNotBlank()) { "Invalid edge endpoints" }
                GraphEdgeRecord(source, target, objectFields(fields["data"]))
            },
        )
    val corpus =
        if ("version" in payload) {
            require(payload["version"] == 2) { "Unsupported PathRAG checkpoint version" }
            CorpusCheckpoint(checkpointRows(payload["chunks"]), checkpointRows(payload["documents"]), checkpointRows(payload["extraction"]))
                .also { records ->
                    require(
                        (records.chunks.values + records.documents.values).all { it["content"] is String },
                    ) { "Missing checkpoint text" }
                }
        } else {
            require(listOf("chunks", "documents", "extraction").none(payload::containsKey)) { "Missing PathRAG checkpoint version" }
            null
        }
    return GraphCheckpoint(graph, corpus)
}
