package ragas.embeddings

/**
 * Contract for embedding providers used by metrics and retrieval evaluators.
 */
interface BaseRagasEmbedding {
    /**
     * Returns an embedding vector for one input text.
     *
     * @param text Input text string.
     */
    suspend fun embedText(text: String): List<Float>

    /**
     * Returns embedding vectors for multiple input texts.
     *
     * @param texts Input text collection.
     */
    suspend fun embedTexts(texts: List<String>): List<List<Float>> =
        texts.map { text ->
            embedText(text)
        }
}
