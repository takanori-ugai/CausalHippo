package com.microsoft.graphrag.index

import com.microsoft.graphrag.logger.ProgressHandler
import com.microsoft.graphrag.logger.progressTicker
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Generates embeddings for document chunks and entities.
 *
 * @property embeddingModel Embedding model used for vector generation.
 */
class EmbedWorkflow(
    private val embeddingModel: EmbeddingModel,
) {
    /**
     * Produce vector embeddings for the given document chunks using the workflow's embedding model.
     *
     * Processes each chunk in an IO dispatcher, reporting progress via the optional handler and
     * associating each successfully computed vector with its chunk id.
     *
     * @param chunks The document chunks to embed.
     * @param progress Optional progress handler that receives periodic updates during embedding.
     * @param description Description passed to the progress ticker for display or logging.
     * @return A list of `TextEmbedding` objects, one per chunk.
     * @throws IllegalStateException if any chunk fails to embed.
     */
    suspend fun embedChunks(
        chunks: List<DocumentChunk>,
        progress: ProgressHandler? = null,
        description: String = "embed chunks progress: ",
    ): List<TextEmbedding> =
        withContext(Dispatchers.IO) {
            val ticker = progressTicker(progress, chunks.size, description)
            val results =
                chunks.map { chunk ->
                    val vector = embed(chunk.text, "chunk ${chunk.id}")
                    ticker()
                    TextEmbedding(chunkId = chunk.id, vector = vector)
                }
            ticker.done()
            results
        }

    /**
     * Produces vector embeddings for the provided entities' names.
     *
     * @param entities The entities to embed; each entity's `name` is used to compute its embedding.
     * @param progress Optional progress handler called as each embedding completes.
     * @param description Human-readable description forwarded to the progress ticker.
     * @return A list of EntityEmbedding objects pairing each `entityId` with its embedding vector.
     * @throws IllegalStateException if any entity fails to embed.
     */
    suspend fun embedEntities(
        entities: List<Entity>,
        progress: ProgressHandler? = null,
        description: String = "embed entities progress: ",
    ): List<EntityEmbedding> =
        withContext(Dispatchers.IO) {
            val ticker = progressTicker(progress, entities.size, description)
            val results =
                entities.map { entity ->
                    val vector = embed(entity.name, "entity ${entity.id}")
                    ticker()
                    EntityEmbedding(entityId = entity.id, vector = vector)
                }
            ticker.done()
            results
        }

    private fun embed(
        text: String,
        item: String,
    ): List<Double> =
        try {
            embeddingModel
                .embed(text)
                .content()
                .vector()
                .map { it.toDouble() }
        } catch (error: CancellationException) {
            throw error
        } catch (error: RuntimeException) {
            throw IllegalStateException("GraphRAG embedding failed for $item", error)
        }
}

/**
 * Creates an OpenAI-backed EmbeddingModel configured with the given API key and model name.
 *
 * @param apiKey The OpenAI API key used to authenticate requests.
 * @param modelName The embedding model identifier to use (default: "text-embedding-3-small").
 * @return An EmbeddingModel instance that sends embedding requests to the specified OpenAI model.
 */
fun defaultEmbeddingModel(
    apiKey: String,
    modelName: String = "text-embedding-3-small",
): EmbeddingModel =
    OpenAiEmbeddingModel
        .builder()
        .apiKey(apiKey)
        .modelName(modelName)
        .build()
