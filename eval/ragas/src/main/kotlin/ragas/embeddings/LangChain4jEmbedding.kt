package ragas.embeddings

import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.embedding.EmbeddingModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Embedding adapter backed by a LangChain4j embedding model.
 *
 * @property model Underlying model instance.
 */
class LangChain4jEmbedding(
    private val model: EmbeddingModel,
) : BaseRagasEmbedding {
    /**
     * Returns an embedding vector for one input text.
     */
    override suspend fun embedText(text: String): List<Float> =
        withContext(Dispatchers.IO) {
            model.embed(text).content().vectorAsList()
        }

    /**
     * Returns embedding vectors for multiple input texts.
     */
    override suspend fun embedTexts(texts: List<String>): List<List<Float>> =
        withContext(Dispatchers.IO) {
            if (texts.isEmpty()) {
                return@withContext emptyList()
            }
            val segments = texts.map { text -> TextSegment.from(text) }
            model.embedAll(segments).content().map { embedding ->
                embedding.vectorAsList()
            }
        }
}
