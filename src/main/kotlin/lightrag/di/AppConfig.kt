package lightrag.di

import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import lightrag.core.AddonConfig
import lightrag.core.Neo4jConfig
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.DocStatusStorage

data class AppConfig(
    val workingDir: String = "./rag_storage",
    val graphStorageName: String = "InMemoryGraphStorage",
    val vectorStorageName: String = "InMemoryVectorStorage",
    val addonConfig: AddonConfig = AddonConfig(),
    val llmBinding: String = "ollama",
    val llmModelName: String = "llama3",
    val embeddingBinding: String = "ollama",
    val embeddingModelName: String = "all-minilm",
    val chatModel: ChatModel,
    val embeddingModel: EmbeddingModel,
    val hashingKv: BaseKVStorage? = null,
    val docStatusStorageOverride: DocStatusStorage? = null,
    val fullDocsStorageOverride: BaseKVStorage? = null,
    val textChunksStorageOverride: BaseKVStorage? = null,
    val fullEntitiesStorageOverride: BaseKVStorage? = null,
    val fullRelationsStorageOverride: BaseKVStorage? = null,
)

data class LightRagConfig(
    val openai: OpenAiConfig,
    val ollama: OllamaConfig,
    val neo4j: Neo4jConfig,
    val mongodb: MongoDbConfig,
    val storage: StorageConfig,
    val addonConfig: AddonConfigConfig,
    val resetStorage: Boolean,
)

data class OpenAiConfig(
    val apiKey: String,
    val chatModelName: String,
    val embeddingModelName: String,
    val embeddingModelDimensions: Int,
)

data class OllamaConfig(
    val baseUrl: String,
    val chatModelName: String,
    val embeddingModelName: String,
)

data class MongoDbConfig(
    val uri: String,
    val database: String,
)

data class StorageConfig(
    val workingDir: String,
    val graphStorageName: String,
    val vectorStorageName: String,
)

data class AddonConfigConfig(
    val chunkTokenSize: Int,
    val chunkOverlapTokenSize: Int,
    val cosineBetterThreshold: Double,
    val entityTypes: List<String>,
    val language: String,
)
