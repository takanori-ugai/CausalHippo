package lightrag.di

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.StreamingChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.core.LightRagOverrides
import lightrag.llm.DualChatModel
import lightrag.llm.LLMFactory
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager

data class LightRagRuntime(
    val lightRagConfig: LightRagConfig,
    val appConfig: AppConfig,
    val chatModel: ChatModel,
    val embeddingModel: EmbeddingModel,
    val globalConfig: Map<String, Any?>,
    val storageManager: StorageManager,
    val ingestionService: IngestionService,
    val queryService: QueryService,
    val rag: LightRAG,
)

fun createLightRagRuntime(
    configPath: String = resolveLightRagConfigPath(),
    configTransform: (LightRagConfig) -> LightRagConfig = { it },
    chatModelFactory: ((LightRagConfig) -> ChatModel)? = null,
    streamingChatModelFactory: ((LightRagConfig) -> StreamingChatModel)? = null,
    embeddingModelFactory: ((LightRagConfig) -> EmbeddingModel)? = null,
    appConfigTransform: (AppConfig, LightRagConfig) -> AppConfig = { appConfig, _ -> appConfig },
    globalConfigFactory: ((AppConfig, LightRagConfig) -> Map<String, Any?>)? = null,
    storageManagerFactory: ((AppConfig, Map<String, Any?>) -> StorageManager)? = null,
): LightRagRuntime {
    val lightRagConfig = configTransform(loadLightRagConfigFromCommonJson(configPath))
    val chatModel = createChatModel(lightRagConfig, chatModelFactory, streamingChatModelFactory)
    val embeddingModel = embeddingModelFactory?.invoke(lightRagConfig) ?: createEmbeddingModel(lightRagConfig)

    val appConfig =
        appConfigTransform(
            defaultAppConfig(lightRagConfig, chatModel, embeddingModel),
            lightRagConfig,
        )

    val globalConfig =
        globalConfigFactory?.invoke(appConfig, lightRagConfig)
            ?: defaultGlobalConfig(appConfig, lightRagConfig)

    val storageManager =
        storageManagerFactory?.invoke(appConfig, globalConfig)
            ?: defaultStorageManager(appConfig, globalConfig)

    val encoding = Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.CL100K_BASE)
    val tokenizer: (String) -> List<Int> = { text ->
        val encoded = encoding.encode(text)
        val tokens = mutableListOf<Int>()
        for (i in 0 until encoded.size()) {
            tokens += encoded.get(i)
        }
        tokens
    }
    val decoder: (List<Int>) -> String = { tokenIds ->
        val ids = IntArrayList()
        tokenIds.forEach { ids.add(it) }
        encoding.decode(ids)
    }

    val ingestionService =
        IngestionService(
            storageManager = storageManager,
            globalConfig = globalConfig,
            tokenizer = tokenizer,
            decoder = decoder,
        )
    val queryService =
        QueryService(
            storageManager = storageManager,
            chatModel = chatModel,
            hashingKv = appConfig.hashingKv,
            globalConfig = globalConfig,
            tokenizer = tokenizer,
            decoder = decoder,
        )
    val rag =
        LightRAG(
            ingestionService = ingestionService,
            queryService = queryService,
            storageManager = storageManager,
        )

    return LightRagRuntime(
        lightRagConfig = lightRagConfig,
        appConfig = appConfig,
        chatModel = chatModel,
        embeddingModel = embeddingModel,
        globalConfig = globalConfig,
        storageManager = storageManager,
        ingestionService = ingestionService,
        queryService = queryService,
        rag = rag,
    )
}

fun defaultAppConfig(
    lightRagConfig: LightRagConfig,
    chatModel: ChatModel,
    embeddingModel: EmbeddingModel,
): AppConfig {
    val provider = normalizedProvider(lightRagConfig)
    return AppConfig(
        llmBinding = provider,
        embeddingBinding = provider,
        llmModelName = activeChatModelName(lightRagConfig),
        embeddingModelName = activeEmbeddingModelName(lightRagConfig),
        chatModel = chatModel,
        embeddingModel = embeddingModel,
        workingDir = lightRagConfig.storage.workingDir,
        graphStorageName = lightRagConfig.storage.graphStorageName,
        vectorStorageName = lightRagConfig.storage.vectorStorageName,
        addonConfig = defaultAddonConfig(lightRagConfig),
    )
}

fun defaultGlobalConfig(
    appConfig: AppConfig,
    lightRagConfig: LightRagConfig,
): Map<String, Any?> {
    val overrides = appConfig.addonConfig.overrides
    val chunkTokenSize = overrides.chunkTokenSize ?: 1200
    val chunkOverlapTokenSize = overrides.chunkOverlapTokenSize ?: 100
    val entityTypes = overrides.entityTypes ?: listOf("Person", "Organization", "Location", "Event", "Concept")
    val language = overrides.language ?: "English"

    return mapOf(
        "llm_model_func" to appConfig.chatModel,
        "embedding_func" to appConfig.embeddingModel,
        "neo4j" to (appConfig.addonConfig.neo4j ?: lightRagConfig.neo4j),
        "chunk_token_size" to chunkTokenSize,
        "chunk_overlap_token_size" to chunkOverlapTokenSize,
        "entity_types" to entityTypes,
        "language" to language,
        "working_dir" to appConfig.workingDir,
        "enable_llm_cache" to (appConfig.hashingKv != null),
    ) + appConfig.addonConfig.toMap()
}

fun defaultStorageManager(
    appConfig: AppConfig,
    globalConfig: Map<String, Any?>,
): StorageManager =
    StorageManager(
        workingDir = appConfig.workingDir,
        embeddingModel = appConfig.embeddingModel,
        graphStorageName = appConfig.graphStorageName,
        vectorStorageName = appConfig.vectorStorageName,
        addonConfig = appConfig.addonConfig,
        globalConfig = globalConfig,
        docStatusStorageOverride = appConfig.docStatusStorageOverride,
        fullDocsStorageOverride = appConfig.fullDocsStorageOverride,
        textChunksStorageOverride = appConfig.textChunksStorageOverride,
        fullEntitiesStorageOverride = appConfig.fullEntitiesStorageOverride,
        fullRelationsStorageOverride = appConfig.fullRelationsStorageOverride,
    )

fun defaultAddonConfig(lightRagConfig: LightRagConfig): AddonConfig =
    AddonConfig(
        neo4j = lightRagConfig.neo4j,
        overrides =
            LightRagOverrides(
                chunkTokenSize = lightRagConfig.addonConfig.chunkTokenSize,
                chunkOverlapTokenSize = lightRagConfig.addonConfig.chunkOverlapTokenSize,
                entityTypes = lightRagConfig.addonConfig.entityTypes,
                language = lightRagConfig.addonConfig.language,
                cosineBetterThreshold = lightRagConfig.addonConfig.cosineBetterThreshold,
            ),
        cosineBetterThreshold = lightRagConfig.addonConfig.cosineBetterThreshold,
    )

private fun createChatModel(
    lightRagConfig: LightRagConfig,
    chatModelFactory: ((LightRagConfig) -> ChatModel)?,
    streamingChatModelFactory: ((LightRagConfig) -> StreamingChatModel)?,
): ChatModel {
    val baseChatModel = chatModelFactory?.invoke(lightRagConfig) ?: createProviderChatModel(lightRagConfig)
    val streamingChatModel = streamingChatModelFactory?.invoke(lightRagConfig)

    if (streamingChatModel != null) {
        return if (baseChatModel is StreamingChatModel) {
            baseChatModel
        } else {
            DualChatModel(baseChatModel, streamingChatModel)
        }
    }

    if (chatModelFactory == null) {
        return DualChatModel(baseChatModel, createProviderStreamingChatModel(lightRagConfig))
    }

    return baseChatModel
}

private fun createProviderChatModel(lightRagConfig: LightRagConfig): ChatModel =
    LLMFactory.createChatModel(
        binding = normalizedProvider(lightRagConfig),
        modelName = activeChatModelName(lightRagConfig),
        baseUrl = activeBaseUrl(lightRagConfig),
        apiKey = activeApiKey(lightRagConfig),
    )

private fun createProviderStreamingChatModel(lightRagConfig: LightRagConfig): StreamingChatModel =
    LLMFactory.createStreamingChatModel(
        binding = normalizedProvider(lightRagConfig),
        modelName = activeChatModelName(lightRagConfig),
        baseUrl = activeBaseUrl(lightRagConfig),
        apiKey = activeApiKey(lightRagConfig),
    )

private fun createEmbeddingModel(lightRagConfig: LightRagConfig): EmbeddingModel =
    LLMFactory.createEmbeddingModel(
        binding = normalizedProvider(lightRagConfig),
        modelName = activeEmbeddingModelName(lightRagConfig),
        baseUrl = activeBaseUrl(lightRagConfig),
        apiKey = activeApiKey(lightRagConfig),
    )

private fun normalizedProvider(lightRagConfig: LightRagConfig): String = lightRagConfig.provider.trim().lowercase()

private fun activeChatModelName(lightRagConfig: LightRagConfig): String =
    if (normalizedProvider(lightRagConfig) == "ollama") {
        lightRagConfig.ollama.chatModelName
    } else {
        lightRagConfig.openai.chatModelName
    }

private fun activeEmbeddingModelName(lightRagConfig: LightRagConfig): String =
    if (normalizedProvider(lightRagConfig) == "ollama") {
        lightRagConfig.ollama.embeddingModelName
    } else {
        lightRagConfig.openai.embeddingModelName
    }

private fun activeBaseUrl(lightRagConfig: LightRagConfig): String? =
    if (normalizedProvider(lightRagConfig) == "ollama") {
        lightRagConfig.ollama.baseUrl
    } else {
        lightRagConfig.openai.baseUrl
    }

private fun activeApiKey(lightRagConfig: LightRagConfig): String? = lightRagConfig.openai.apiKey.takeIf { it.isNotBlank() }
