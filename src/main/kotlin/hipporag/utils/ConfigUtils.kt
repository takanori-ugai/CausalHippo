package hipporag.utils

import hipporag.config.BaseConfig
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive
import java.io.File

/**
 * Loads a [BaseConfig] from a JSON file at [path].
 */
fun loadConfigFromJson(path: String): BaseConfig {
    val text = File(path).readText()
    val json = jsonWithDefaults { ignoreUnknownKeys = true }
    val element = json.parseToJsonElement(text)
    val config = BaseConfig()
    if (element is JsonObject) {
        applyConfigOverrides(config, element)
    }
    return config
}

/**
 * Applies JSON override values to an existing [config].
 */
fun applyConfigOverrides(
    config: BaseConfig,
    overrides: JsonObject,
): BaseConfig {
    val logger = KotlinLogging.logger {}

    fun str(key: String): String? = overrides[key]?.jsonPrimitive?.contentOrNull

    fun bool(key: String): Boolean? = overrides[key]?.jsonPrimitive?.booleanOrNull

    fun int(key: String): Int? = overrides[key]?.jsonPrimitive?.intOrNull

    fun double(key: String): Double? = overrides[key]?.jsonPrimitive?.doubleOrNull

    val alias =
        mapOf(
            "llm_name" to "llmName",
            "llm_base_url" to "llmBaseUrl",
            "embedding_model_name" to "embeddingModelName",
            "embedding_base_url" to "embeddingBaseUrl",
            "ollama_base_url" to "ollamaBaseUrl",
            "ollama_model_name" to "ollamaModelName",
            "ollama_embedding_model_name" to "ollamaEmbeddingModelName",
            "llm_provider" to "llmProvider",
            "embedding_provider" to "embeddingProvider",
            "max_new_tokens" to "maxNewTokens",
            "num_gen_choices" to "numGenChoices",
            "max_retry_attempts" to "maxRetryAttempts",
            "openie_mode" to "openieMode",
            "information_extraction_model_name" to "informationExtractionModelName",
            "force_index_from_scratch" to "forceIndexFromScratch",
            "force_openie_from_scratch" to "forceOpenieFromScratch",
            "is_directed_graph" to "isDirectedGraph",
            "embedding_batch_size" to "embeddingBatchSize",
            "embedding_return_as_normalized" to "embeddingReturnAsNormalized",
            "embedding_max_seq_len" to "embeddingMaxSeqLen",
            "embedding_model_dtype" to "embeddingModelDtype",
            "retrieval_top_k" to "retrievalTopK",
            "linking_top_k" to "linkingTopK",
            "passage_node_weight" to "passageNodeWeight",
            "max_qa_steps" to "maxQaSteps",
            "qa_top_k" to "qaTopK",
            "synonymy_edge_topk" to "synonymyEdgeTopK",
            "synonymy_edge_query_batch_size" to "synonymyEdgeQueryBatchSize",
            "synonymy_edge_key_batch_size" to "synonymyEdgeKeyBatchSize",
            "synonymy_edge_sim_threshold" to "synonymyEdgeSimThreshold",
            "save_openie" to "saveOpenie",
            "text_preprocessor_class_name" to "textPreprocessorClassName",
            "preprocess_encoder_name" to "preprocessEncoderName",
            "preprocess_chunk_overlap_token_size" to "preprocessChunkOverlapTokenSize",
            "preprocess_chunk_max_token_size" to "preprocessChunkMaxTokenSize",
            "preprocess_chunk_func" to "preprocessChunkFunc",
            "rerank_dspy_file_path" to "rerankDspyFilePath",
            "save_dir" to "saveDir",
        )

    fun resolveKey(key: String): String = alias[key] ?: key

    for (entry in overrides.entries) {
        val key = resolveKey(entry.key)
        when (key) {
            "saveDir" -> str(entry.key)?.let { config.saveDir = it }
            "llmName" -> str(entry.key)?.let { config.llmName = it }
            "embeddingModelName" -> str(entry.key)?.let { config.embeddingModelName = it }
            "llmBaseUrl" -> str(entry.key)?.let { config.llmBaseUrl = it }
            "embeddingBaseUrl" -> str(entry.key)?.let { config.embeddingBaseUrl = it }
            "llmProvider" -> str(entry.key)?.let { config.llmProvider = it }
            "embeddingProvider" -> str(entry.key)?.let { config.embeddingProvider = it }
            "openAiApiKey" -> str(entry.key)?.let { config.openAiApiKey = it }
            "ollamaBaseUrl" -> str(entry.key)?.let { config.ollamaBaseUrl = it }
            "ollamaModelName" -> str(entry.key)?.let { config.ollamaModelName = it }
            "ollamaEmbeddingModelName" -> str(entry.key)?.let { config.ollamaEmbeddingModelName = it }
            "rerankDspyFilePath" -> str(entry.key)?.let { config.rerankDspyFilePath = it }
            "maxNewTokens" -> int(entry.key)?.let { config.maxNewTokens = it }
            "numGenChoices" -> int(entry.key)?.let { config.numGenChoices = it }
            "seed" -> int(entry.key)?.let { config.seed = it }
            "temperature" -> double(entry.key)?.let { config.temperature = it }
            "maxRetryAttempts" -> int(entry.key)?.let { config.maxRetryAttempts = it }
            "openieMode" -> str(entry.key)?.let { config.openieMode = it }
            "informationExtractionModelName" -> str(entry.key)?.let { config.informationExtractionModelName = it }
            "skipGraph" -> bool(entry.key)?.let { config.skipGraph = it }
            "forceIndexFromScratch" -> bool(entry.key)?.let { config.forceIndexFromScratch = it }
            "forceOpenieFromScratch" -> bool(entry.key)?.let { config.forceOpenieFromScratch = it }
            "isDirectedGraph" -> bool(entry.key)?.let { config.isDirectedGraph = it }
            "embeddingBatchSize" -> int(entry.key)?.let { config.embeddingBatchSize = it }
            "embeddingReturnAsNormalized" -> bool(entry.key)?.let { config.embeddingReturnAsNormalized = it }
            "embeddingMaxSeqLen" -> int(entry.key)?.let { config.embeddingMaxSeqLen = it }
            "embeddingModelDtype" -> str(entry.key)?.let { config.embeddingModelDtype = it }
            "retrievalTopK" -> int(entry.key)?.let { config.retrievalTopK = it }
            "linkingTopK" -> int(entry.key)?.let { config.linkingTopK = it }
            "passageNodeWeight" -> double(entry.key)?.let { config.passageNodeWeight = it }
            "dataset" -> str(entry.key)?.let { config.dataset = it }
            "maxQaSteps" -> int(entry.key)?.let { config.maxQaSteps = it }
            "qaTopK" -> int(entry.key)?.let { config.qaTopK = it }
            "synonymyEdgeTopK" -> int(entry.key)?.let { config.synonymyEdgeTopK = it }
            "synonymyEdgeQueryBatchSize" -> int(entry.key)?.let { config.synonymyEdgeQueryBatchSize = it }
            "synonymyEdgeKeyBatchSize" -> int(entry.key)?.let { config.synonymyEdgeKeyBatchSize = it }
            "synonymyEdgeSimThreshold" -> double(entry.key)?.let { config.synonymyEdgeSimThreshold = it }
            "saveOpenie" -> bool(entry.key)?.let { config.saveOpenie = it }
            "damping" -> double(entry.key)?.let { config.damping = it }
            "textPreprocessorClassName" -> str(entry.key)?.let { config.textPreprocessorClassName = it }
            "preprocessEncoderName" -> str(entry.key)?.let { config.preprocessEncoderName = it }
            "preprocessChunkOverlapTokenSize" -> int(entry.key)?.let { config.preprocessChunkOverlapTokenSize = it }
            "preprocessChunkMaxTokenSize" -> int(entry.key)?.let { config.preprocessChunkMaxTokenSize = it }
            "preprocessChunkFunc" -> str(entry.key)?.let { config.preprocessChunkFunc = it }
            else -> logger.debug { "Unknown config key '${entry.key}' ignored." }
        }
    }

    return config
}
