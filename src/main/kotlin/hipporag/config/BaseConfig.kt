package hipporag.config

/**
 * Runtime configuration for the HippoRAG pipeline.
 *
 * @property saveDir output directory for artifacts.
 * @property llmName LLM model name.
 * @property embeddingModelName embedding model name.
 * @property llmBaseUrl optional base URL for LLM provider.
 * @property embeddingBaseUrl optional base URL for embedding provider.
 * @property llmProvider optional provider hint ("openai" or "ollama").
 * @property openAiApiKey OpenAI API key override.
 * @property ollamaBaseUrl Ollama base URL.
 * @property ollamaModelName Ollama chat model name.
 * @property embeddingProvider optional embedding provider hint.
 * @property ollamaEmbeddingModelName Ollama embedding model name.
 * @property rerankDspyFilePath optional path to DSPy rerank prompt JSON.
 * @property maxNewTokens maximum tokens to generate.
 * @property numGenChoices number of generation choices.
 * @property seed optional random seed.
 * @property temperature sampling temperature.
 * @property maxRetryAttempts retry attempts for LLM calls.
 * @property responseFormat optional response format override.
 * @property openieMode OpenIE mode ("online", "offline", or "transformers-offline").
 * @property informationExtractionModelName information extraction model name.
 * @property skipGraph whether to skip graph construction.
 * @property forceIndexFromScratch whether to rebuild index even if cached.
 * @property forceOpenieFromScratch whether to rerun OpenIE even if cached.
 * @property isDirectedGraph whether the graph is directed.
 * @property embeddingBatchSize embedding batch size.
 * @property embeddingReturnAsNormalized whether embeddings should be normalized.
 * @property embeddingMaxSeqLen max sequence length for embeddings.
 * @property embeddingModelDtype embedding model dtype string.
 * @property retrievalTopK retrieval top-k for documents.
 * @property linkingTopK top-k entities for graph linking.
 * @property passageNodeWeight weight for passage nodes in graph search.
 * @property dataset dataset name for prompt selection.
 * @property maxQaSteps max QA steps (reserved for future use).
 * @property qaTopK top-k passages used for QA prompts.
 * @property synonymyEdgeTopK top-k for synonymy edge creation.
 * @property synonymyEdgeQueryBatchSize batch size for synonymy queries.
 * @property synonymyEdgeKeyBatchSize batch size for synonymy keys.
 * @property synonymyEdgeSimThreshold similarity threshold for synonymy edges.
 * @property saveOpenie whether to persist OpenIE results.
 * @property damping damping factor for PPR.
 * @property textPreprocessorClassName text preprocessor class name.
 * @property preprocessEncoderName encoder name for preprocessing.
 * @property preprocessChunkOverlapTokenSize overlap size for chunking.
 * @property preprocessChunkMaxTokenSize max chunk token size.
 * @property preprocessChunkFunc chunking strategy identifier.
 */
data class BaseConfig(
    var saveDir: String = "outputs",
    var llmName: String = "gpt-5.4-mini",
    var embeddingModelName: String = "text-embedding-3-small",
    var llmBaseUrl: String? = null,
    var embeddingBaseUrl: String? = null,
    var llmProvider: String? = null,
    var openAiApiKey: String? = null,
    var ollamaBaseUrl: String? = null,
    var ollamaModelName: String? = null,
    var embeddingProvider: String? = null,
    var ollamaEmbeddingModelName: String? = null,
    var rerankDspyFilePath: String? = null,
    var maxNewTokens: Int? = 2048,
    var numGenChoices: Int = 1,
    var seed: Int? = null,
    var temperature: Double = 0.0,
    var maxRetryAttempts: Int = 5,
    var responseFormat: Map<String, String>? = mapOf("type" to "json_object"),
    var openieMode: String = "online",
    var informationExtractionModelName: String = "openie_openai_gpt",
    var skipGraph: Boolean = false,
    var forceIndexFromScratch: Boolean = false,
    var forceOpenieFromScratch: Boolean = false,
    var isDirectedGraph: Boolean = false,
    var embeddingBatchSize: Int = 16,
    var embeddingReturnAsNormalized: Boolean = true,
    var embeddingMaxSeqLen: Int = 2048,
    var embeddingModelDtype: String = "auto",
    var retrievalTopK: Int = 200,
    var linkingTopK: Int = 5,
    var passageNodeWeight: Double = 0.05,
    var dataset: String = "",
    var maxQaSteps: Int = 1,
    var qaTopK: Int = 5,
    var synonymyEdgeTopK: Int = 2047,
    var synonymyEdgeQueryBatchSize: Int = 1000,
    var synonymyEdgeKeyBatchSize: Int = 10000,
    var synonymyEdgeSimThreshold: Double = 0.8,
    var saveOpenie: Boolean = true,
    var damping: Double = 0.5,
    var textPreprocessorClassName: String = "TextPreprocessor",
    var preprocessEncoderName: String = "gpt-4o",
    var preprocessChunkOverlapTokenSize: Int = 128,
    var preprocessChunkMaxTokenSize: Int? = 1200,
    var preprocessChunkFunc: String = "by_token",
) {
    /**
     * Returns a map representation of the configuration, masking API keys.
     */
    fun toMap(): Map<String, Any?> =
        mapOf(
            "saveDir" to saveDir,
            "llmName" to llmName,
            "embeddingModelName" to embeddingModelName,
            "llmBaseUrl" to llmBaseUrl,
            "embeddingBaseUrl" to embeddingBaseUrl,
            "openieMode" to openieMode,
            "llmProvider" to llmProvider,
            "openAiApiKey" to openAiApiKey?.let { "***" },
            "ollamaBaseUrl" to ollamaBaseUrl,
            "ollamaModelName" to ollamaModelName,
            "embeddingProvider" to embeddingProvider,
            "ollamaEmbeddingModelName" to ollamaEmbeddingModelName,
            "rerankDspyFilePath" to rerankDspyFilePath,
            "maxNewTokens" to maxNewTokens,
            "numGenChoices" to numGenChoices,
            "seed" to seed,
            "temperature" to temperature,
            "maxRetryAttempts" to maxRetryAttempts,
            "responseFormat" to responseFormat,
            "forceIndexFromScratch" to forceIndexFromScratch,
            "forceOpenieFromScratch" to forceOpenieFromScratch,
            "isDirectedGraph" to isDirectedGraph,
            "embeddingBatchSize" to embeddingBatchSize,
            "embeddingReturnAsNormalized" to embeddingReturnAsNormalized,
            "embeddingMaxSeqLen" to embeddingMaxSeqLen,
            "embeddingModelDtype" to embeddingModelDtype,
            "retrievalTopK" to retrievalTopK,
            "linkingTopK" to linkingTopK,
            "passageNodeWeight" to passageNodeWeight,
            "dataset" to dataset,
            "maxQaSteps" to maxQaSteps,
            "qaTopK" to qaTopK,
            "synonymyEdgeTopK" to synonymyEdgeTopK,
            "synonymyEdgeQueryBatchSize" to synonymyEdgeQueryBatchSize,
            "synonymyEdgeKeyBatchSize" to synonymyEdgeKeyBatchSize,
            "synonymyEdgeSimThreshold" to synonymyEdgeSimThreshold,
            "saveOpenie" to saveOpenie,
            "damping" to damping,
            "textPreprocessorClassName" to textPreprocessorClassName,
            "preprocessEncoderName" to preprocessEncoderName,
            "preprocessChunkOverlapTokenSize" to preprocessChunkOverlapTokenSize,
            "preprocessChunkMaxTokenSize" to preprocessChunkMaxTokenSize,
            "preprocessChunkFunc" to preprocessChunkFunc,
            "informationExtractionModelName" to informationExtractionModelName,
            "skipGraph" to skipGraph,
        )
}
