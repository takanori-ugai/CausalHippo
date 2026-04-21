package hipporag.embeddingmodel

import hipporag.config.BaseConfig

/**
 * Abstraction for embedding models used in HippoRAG.
 */
interface BaseEmbeddingModel {
    /**
     * Encodes [texts] into dense vectors.
     *
     * @param instruction optional instruction prefix for the embedding model.
     * @param norm whether to L2-normalize each output vector.
     */
    fun batchEncode(
        texts: List<String>,
        instruction: String? = null,
        norm: Boolean = false,
    ): Array<DoubleArray>
}

/**
 * Factory for creating [BaseEmbeddingModel] instances.
 */
interface EmbeddingModelFactory {
    /**
     * Creates an embedding model from configuration and a model name.
     */
    fun create(
        globalConfig: BaseConfig,
        embeddingModelName: String,
    ): BaseEmbeddingModel
}

/**
 * Returns the default embedding model factory.
 */
fun getEmbeddingModel(): EmbeddingModelFactory = LangChainEmbeddingFactory()
