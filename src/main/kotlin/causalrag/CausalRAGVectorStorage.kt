package causalrag

import causalrag.retriever.VectorStoreRetriever
import io.github.oshai.kotlinlogging.KotlinLogging
import shared.rag.CommonVectorStorage

/**
 * CommonVectorStorage adapter for CausalRAG's in-memory [VectorStoreRetriever].
 */
class CausalRAGVectorStorage(
    private val vectorRetriever: VectorStoreRetriever,
) : CommonVectorStorage<Any?> {
    private val logger = KotlinLogging.logger("CausalRAGVectorStorage")

    override suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>> {
        val safeTopK = topK.coerceAtLeast(1)
        return vectorRetriever
            .searchWithMetadata(query, topK = safeTopK)
            .map { raw ->
                val metadata = (raw["metadata"] as? Map<*, *>)?.entries?.associate { it.key.toString() to it.value } ?: emptyMap()
                mapOf(
                    "content" to (raw["passage"]?.toString().orEmpty()),
                    "score" to (raw["score"] as? Double ?: 0.0),
                    "rank" to (raw["rank"] as? Int ?: 0),
                ) + metadata
            }
    }

    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {
        if (data.isEmpty()) return

        val existing = snapshotEntries().toMutableMap()
        data.forEach { (id, update) ->
            val prior = existing[id]
            val content =
                update["content"]?.toString()
                    ?: update["passage"]?.toString()
                    ?: prior?.content
            if (content.isNullOrBlank()) {
                logger.warn { "Skipping vector upsert row '$id': missing content/passage." }
                return@forEach
            }
            val mergedMeta =
                (prior?.metadata?.toMutableMap() ?: mutableMapOf()).apply {
                    putAll(
                        update
                            .filterKeys { key -> key != "content" && key != "passage" && key != "vector" }
                            .mapNotNull { (key, value) ->
                                value?.let { key to it }
                            }.toMap(),
                    )
                    put("id", id)
                }
            existing[id] = Entry(content = content, metadata = mergedMeta)
        }

        reindexEntries(existing)
    }

    override suspend fun deleteEntity(entityName: String) {
        val filtered =
            snapshotEntries()
                .filterValues { entry ->
                    val entity = entry.metadata["entity_name"]?.toString()
                    val entityId = entry.metadata["entity_id"]?.toString()
                    entity != entityName && entityId != entityName
                }
        reindexEntries(filtered)
    }

    override suspend fun deleteEntityRelation(entityName: String) {
        val filtered =
            snapshotEntries()
                .filterValues { entry ->
                    val src = entry.metadata["src_id"]?.toString() ?: entry.metadata["source_id"]?.toString()
                    val tgt = entry.metadata["tgt_id"]?.toString() ?: entry.metadata["target_id"]?.toString()
                    src != entityName && tgt != entityName
                }
        reindexEntries(filtered)
    }

    private data class Entry(
        val content: String,
        val metadata: Map<String, Any>,
    )

    private fun snapshotEntries(): Map<String, Entry> {
        val passages = vectorRetriever.getPassages()
        val metadata = vectorRetriever.getMetadata()
        val size = minOf(passages.size, metadata.size)
        val entries = LinkedHashMap<String, Entry>(size)
        for (i in 0 until size) {
            val meta = metadata[i].toMutableMap()
            val id = meta["id"]?.toString()?.takeIf { it.isNotBlank() } ?: i.toString()
            meta["id"] = id
            entries[id] = Entry(content = passages[i], metadata = meta)
        }
        return entries
    }

    private fun reindexEntries(entries: Map<String, Entry>) {
        if (entries.isEmpty()) {
            vectorRetriever.clear()
            return
        }
        val ordered = entries.toSortedMap()
        val ids = ordered.keys.toList()
        val texts = ordered.values.map { it.content }
        val metadata = ordered.values.map { it.metadata }
        vectorRetriever.indexCorpus(texts = texts, metadata = metadata, ids = ids)
    }
}
