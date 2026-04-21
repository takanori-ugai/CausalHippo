package lightrag

import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager
import java.io.File

/**
 * Build a LightRAG instance for tests with supplied chat/embedding models and a working directory.
 */
fun buildTestLightRag(
    chatModel: ChatModel,
    embeddingModel: EmbeddingModel,
    workingDir: File,
): LightRAG {
    val globalConfig =
        mapOf(
            "llm_model_func" to chatModel,
            "embedding_func" to embeddingModel,
            "chunk_token_size" to 1200,
            "chunk_overlap_token_size" to 100,
            "entity_types" to listOf("Person", "Organization", "Location", "Event", "Concept"),
            "language" to "English",
            "working_dir" to workingDir.absolutePath,
            "enable_llm_cache" to false,
        )

    val tokenizer: (String) -> List<Int> = { text ->
        text.codePoints().toArray().toList()
    }
    val decoder: (List<Int>) -> String = { tokens ->
        buildString {
            tokens.forEach { append(it.toChar()) }
        }
    }

    val storageManager =
        StorageManager(
            workingDir = workingDir.absolutePath,
            embeddingModel = embeddingModel,
            addonConfig = AddonConfig(),
            globalConfig = globalConfig,
        )
    val ingestionService = IngestionService(storageManager, globalConfig, tokenizer, decoder)
    val queryService =
        QueryService(
            storageManager = storageManager,
            chatModel = chatModel,
            hashingKv = null,
            globalConfig = globalConfig,
            tokenizer = tokenizer,
            decoder = decoder,
        )

    return LightRAG(ingestionService, queryService, storageManager)
}
