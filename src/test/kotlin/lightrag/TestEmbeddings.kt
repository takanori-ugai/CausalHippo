package lightrag

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.output.Response
import io.mockk.every
import io.mockk.mockk

/** Helpers to build mock embedding models for tests. */
object TestEmbeddings {
    /**
     * Builds a mock [EmbeddingModel] that returns deterministic vectors for provided inputs.
     * @param vectors map of text to embedding vectors to return
     * @param default default vector to return when text not in [vectors]
     */
    fun mockEmbeddingModel(
        vectors: Map<String, List<Float>> = emptyMap(),
        default: List<Float> = listOf(0.0f, 0.0f),
    ): EmbeddingModel {
        val model = mockk<EmbeddingModel>()

        every { model.embed(any<String>()) } answers {
            val text = this.args.first() as String
            val vector = vectors[text] ?: default
            Response.from(Embedding(vector.toFloatArray()))
        }
        return model
    }
}
