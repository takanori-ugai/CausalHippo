package com.microsoft.graphrag

import com.microsoft.graphrag.index.EntityEmbedding
import com.microsoft.graphrag.index.LocalVectorStore
import com.microsoft.graphrag.index.TextEmbedding
import com.microsoft.graphrag.index.TextUnit
import dev.langchain4j.model.embedding.EmbeddingModel
import io.github.oshai.kotlinlogging.KotlinLogging
import shared.rag.CommonVectorStorage

/**
 * GraphRAG adapter that exposes [LocalVectorStore] through [CommonVectorStorage].
 *
 * This keeps GraphRAG query engines on LocalVectorStore while allowing a shared vector-storage
 * interface for cross-RAG integrations.
 */
class GraphRAGVectorStorage(
    private val localVectorStore: LocalVectorStore,
    textUnits: List<TextUnit>,
    private val embeddingModel: EmbeddingModel,
) : CommonVectorStorage<Any?> {
    private val logger = KotlinLogging.logger("GraphRAGVectorStorage")
    private val textByChunkId = textUnits.associateBy { it.chunkId }

    fun asLocalVectorStore(): LocalVectorStore = localVectorStore

    override suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>> {
        if (query.isBlank() || topK <= 0) return emptyList()
        val queryEmbedding = embed(query) ?: return emptyList()

        val textHits = localVectorStore.nearestTextChunks(queryEmbedding, topK)
        if (textHits.isNotEmpty()) {
            return textHits.map { (chunkId, distance) ->
                val textUnit = textByChunkId[chunkId]
                mapOf(
                    "id" to chunkId,
                    "chunk_id" to chunkId,
                    "content" to (textUnit?.text ?: ""),
                    "source_id" to (textUnit?.id ?: chunkId),
                    "distance" to distance,
                    "score" to (1.0 / (1.0 + distance)),
                    "kind" to "text",
                )
            }
        }

        return localVectorStore
            .nearestEntities(queryEmbedding, topK)
            .map { (entityId, distance) ->
                mapOf(
                    "id" to entityId,
                    "entity_id" to entityId,
                    "distance" to distance,
                    "score" to (1.0 / (1.0 + distance)),
                    "kind" to "entity",
                )
            }
    }

    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {
        if (data.isEmpty()) return

        val payload = localVectorStore.load()
        val textEmbeddingsById =
            payload
                ?.textEmbeddings
                ?.associateBy { it.chunkId }
                ?.toMutableMap()
                ?: mutableMapOf()
        val entityEmbeddingsById =
            payload
                ?.entityEmbeddings
                ?.associateBy { it.entityId }
                ?.toMutableMap()
                ?: mutableMapOf()

        data.forEach { (id, row) ->
            val kind = row["kind"]?.toString()?.lowercase()
            val vector =
                extractVector(row["vector"])
                    ?: run {
                        val content =
                            row["content"]?.toString()
                                ?: row["text"]?.toString()
                                ?: row["name"]?.toString()
                        content?.takeIf { it.isNotBlank() }?.let { embed(it) }
                    }

            if (vector.isNullOrEmpty()) {
                logger.warn { "Skipping vector upsert row '$id': no usable vector or embeddable text." }
                return@forEach
            }

            val isEntity = kind == "entity" || row.containsKey("entity_id")
            if (isEntity) {
                val entityId = row["entity_id"]?.toString()?.takeIf { it.isNotBlank() } ?: id
                entityEmbeddingsById[entityId] = EntityEmbedding(entityId = entityId, vector = vector)
            } else {
                val chunkId = row["chunk_id"]?.toString()?.takeIf { it.isNotBlank() } ?: id
                textEmbeddingsById[chunkId] = TextEmbedding(chunkId = chunkId, vector = vector)
            }
        }

        localVectorStore.save(
            textEmbeddings = textEmbeddingsById.values.toList(),
            entityEmbeddings = entityEmbeddingsById.values.toList(),
        )
    }

    override suspend fun deleteEntity(entityName: String) {
        val payload = localVectorStore.load() ?: return
        val nextEntities = payload.entityEmbeddings.filterNot { it.entityId == entityName }
        localVectorStore.save(payload.textEmbeddings, nextEntities)
    }

    override suspend fun deleteEntityRelation(entityName: String) {
        // GraphRAG's LocalVectorStore keeps text/entity embeddings only; no dedicated relation vectors.
        logger.debug { "deleteEntityRelation('$entityName') is a no-op for GraphRAGVectorStorage." }
    }

    private fun extractVector(raw: Any?): List<Double>? =
        (raw as? List<*>)
            ?.mapNotNull { value -> (value as? Number)?.toDouble() }
            ?.takeIf { it.isNotEmpty() }

    private fun embed(text: String): List<Double>? =
        runCatching {
            embeddingModel
                .embed(text)
                .content()
                .vector()
                .asList()
                .map { it.toDouble() }
        }.getOrElse { ex ->
            logger.warn(ex) { "Failed to embed text for GraphRAGVectorStorage query/upsert." }
            null
        }
}
