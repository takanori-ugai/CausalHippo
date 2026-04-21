package hipporag.embeddingmodel

import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.embedding.EmbeddingModel
import hipporag.utils.normalizeVector

/**
 * Adapter that exposes a LangChain4j [EmbeddingModel] as a [BaseEmbeddingModel].
 */
class LangChainEmbeddingModel(
    private val model: EmbeddingModel,
) : BaseEmbeddingModel {
    /**
     * Encodes [texts] using the underlying LangChain4j model.
     */
    override fun batchEncode(
        texts: List<String>,
        instruction: String?,
        norm: Boolean,
    ): Array<DoubleArray> {
        if (texts.isEmpty()) return emptyArray()
        val segments =
            texts.map { text ->
                val prefixed = if (instruction != null) "$instruction $text" else text
                TextSegment.from(prefixed)
            }
        val result = model.embedAll(segments)
        val embeddings =
            result.content().map { embedding ->
                val vector = embedding.vector()
                val asDouble = DoubleArray(vector.size) { idx -> vector[idx].toDouble() }
                if (norm) normalizeVector(asDouble) else asDouble
            }
        return embeddings.toTypedArray()
    }
}
