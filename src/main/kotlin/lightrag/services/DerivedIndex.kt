package lightrag.services

import lightrag.core.Constants
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.utils.computeMd5

/** Captures derived records, including their existing embeddings, without writing to the stores. */
internal class DerivedIndex private constructor(
    private val nodes: Map<String, Map<String, String>>,
    private val edges: Map<Pair<String, String>, Map<String, String>>,
    private val chunks: Map<String, Map<String, Any>>,
    private val entities: Map<String, Map<String, Any>>,
    private val relations: Map<String, Map<String, Any>>,
    private val chunkVectors: Map<String, Map<String, Any>>,
    private val entityVectors: Map<String, Map<String, Any>>,
    private val relationVectors: Map<String, Map<String, Any>>,
) {
    suspend fun restore(stores: StorageManager) {
        stores.dropDerivedStorage()
        for ((id, properties) in nodes) {
            stores.chunkEntityRelationGraph.upsertNode(id, properties)
        }
        for ((endpoints, properties) in edges) {
            stores.chunkEntityRelationGraph.upsertEdge(endpoints.first, endpoints.second, properties)
        }
        stores.textChunks.upsert(chunks)
        stores.fullEntities.upsert(entities)
        stores.fullRelations.upsert(relations)
        restoreVectors(stores.chunksVdb, chunkVectors)
        restoreVectors(stores.entitiesVdb, entityVectors)
        restoreVectors(stores.relationshipsVdb, relationVectors)
    }

    private suspend fun restoreVectors(
        store: BaseVectorStorage,
        rows: Map<String, Map<String, Any>>,
    ) {
        if (rows.isEmpty()) return
        store.upsert(rows)
        val found = store.getVectorsByIds(rows.keys.toList())
        check(rows.keys.all { !found[it].isNullOrEmpty() }) { "Incomplete vector write to ${store.namespace}" }
    }

    companion object {
        suspend fun capture(
            stores: StorageManager,
            knownChunks: Set<String>,
        ): DerivedIndex {
            val graph = stores.chunkEntityRelationGraph
            val nodes =
                graph.getAllNodes().associate { row ->
                    val id = checkNotNull(row["entity_id"] ?: row["id"] ?: row["entity_name"] ?: row["_id"]).toString()
                    id to (row.mapValues { it.value.toString() } + ("entity_id" to id))
                }
            val edges =
                graph.getAllEdges().associate { row ->
                    val source = checkNotNull(row["source"] ?: row["src_id"] ?: row["source_id"]).toString()
                    val target = checkNotNull(row["target"] ?: row["tgt_id"] ?: row["target_id"]).toString()
                    val endpoints = if (source <= target) source to target else target to source
                    endpoints to (row - setOf("source", "target")).mapValues { it.value.toString() }
                }
            val chunkIds =
                knownChunks + stores.textChunks.keys().orEmpty() +
                    (nodes.values + edges.values).flatMap {
                        it["source_id"].orEmpty().split(Constants.GRAPH_FIELD_SEP).filter(String::isNotBlank)
                    }
            val edgeIds = edges.keys.flatMap { (source, target) -> listOf("$source#$target", "$target#$source") }.toSet()
            return DerivedIndex(
                nodes,
                edges,
                readRows(stores.textChunks, chunkIds),
                readRows(stores.fullEntities, nodes.keys),
                readRows(stores.fullRelations, edgeIds),
                readVectors(stores.chunksVdb, chunkIds),
                readVectors(stores.entitiesVdb, nodes.keys.map(::computeMd5).toSet()),
                readVectors(stores.relationshipsVdb, edgeIds.map(::computeMd5).toSet()),
            )
        }

        private suspend fun readRows(
            store: BaseKVStorage,
            ids: Set<String>,
        ): Map<String, Map<String, Any>> =
            buildMap {
                for (id in ids) {
                    store.getById(id)?.let { put(id, it) }
                }
            }

        private suspend fun readVectors(
            store: BaseVectorStorage,
            ids: Set<String>,
        ): Map<String, Map<String, Any>> =
            buildMap {
                for ((id, vector) in store.getVectorsByIds(ids.toList())) {
                    val row = checkNotNull(store.getById(id)) { "Missing metadata for vector $id in ${store.namespace}" }
                    put(id, row + ("vector" to vector))
                }
            }
    }
}
