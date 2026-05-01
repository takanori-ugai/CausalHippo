package lightrag.di

import io.github.oshai.kotlinlogging.KotlinLogging
import lightrag.core.Neo4jConfig
import shared.config.CommonRagConfigLoader
import shared.config.LightRagSettings
import java.nio.file.Files
import java.nio.file.Path

/**
 * Load the LightRAG-specific runtime settings from a common multi-module JSON config file.
 */
fun loadLightRagSettingsFromCommonConfig(path: String): LightRagSettings = CommonRagConfigLoader.load(path).toLightRagSettings()

const val LIGHTRAG_CONFIG_ENV = "LIGHTRAG_CONFIG"
const val DEFAULT_LIGHTRAG_COMMON_CONFIG = "config/common_rag.json"

private const val DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
private const val DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
private const val DEFAULT_OPENAI_EMBEDDING_DIMENSION = 1536
private const val DEFAULT_OLLAMA_CHAT_MODEL = "llama3"
private const val DEFAULT_OLLAMA_EMBEDDING_MODEL = "all-minilm"
private const val DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

private val logger = KotlinLogging.logger {}

fun resolveLightRagConfigPath(explicitPath: String? = null): String {
    explicitPath?.trim()?.takeIf { it.isNotEmpty() }?.let { return it }
    System
        .getenv(LIGHTRAG_CONFIG_ENV)
        ?.trim()
        ?.takeIf { it.isNotEmpty() }
        ?.let { return it }
    return DEFAULT_LIGHTRAG_COMMON_CONFIG
}

/**
 * Build [LightRagConfig] from common JSON config (`config/common_rag.json`) with env fallbacks.
 */
fun loadLightRagConfigFromCommonJson(path: String = resolveLightRagConfigPath()): LightRagConfig {
    val settings =
        if (Files.exists(Path.of(path))) {
            loadLightRagSettingsFromCommonConfig(path)
        } else {
            logger.warn { "Common config not found at '$path'; using LightRAG defaults + environment variables." }
            LightRagSettings(
                provider = null,
                llmModelName = null,
                embeddingModelName = null,
                embeddingModelDimensions = null,
                apiKey = null,
                baseUrl = null,
                workingDir = null,
                graphStorageName = null,
                vectorStorageName = null,
                chunkTokenSize = null,
                chunkOverlapTokenSize = null,
                entityTypes = null,
                language = null,
                cosineBetterThreshold = null,
            )
        }

    val provider = normalizeProvider(settings.provider ?: System.getenv("LLM_PROVIDER") ?: "openai")
    val sharedBaseUrl = firstNonBlankOrNull(settings.baseUrl, System.getenv("LLM_BASE_URL"))

    val openAiApiKey = firstNonBlankOrNull(settings.apiKey, System.getenv("OPENAI_API_KEY")).orEmpty()
    val openAiBaseUrl = firstNonBlankOrNull(sharedBaseUrl, System.getenv("OPENAI_API_BASE"))
    val openAiChatModelName =
        if (provider == "openai") {
            firstNonBlank(settings.llmModelName, System.getenv("LLM_MODEL"), DEFAULT_OPENAI_CHAT_MODEL)
        } else {
            firstNonBlank(System.getenv("OPENAI_MODEL"), DEFAULT_OPENAI_CHAT_MODEL)
        }
    val openAiEmbeddingModelName =
        if (provider == "openai") {
            firstNonBlank(settings.embeddingModelName, System.getenv("EMBEDDING_MODEL"), DEFAULT_OPENAI_EMBEDDING_MODEL)
        } else {
            firstNonBlank(System.getenv("OPENAI_EMBEDDING_MODEL"), DEFAULT_OPENAI_EMBEDDING_MODEL)
        }
    val openAiEmbeddingModelDimensions = resolveEmbeddingDimension(settings, openAiEmbeddingModelName)

    val ollamaBaseUrl = firstNonBlank(sharedBaseUrl, System.getenv("OLLAMA_BASE_URL"), DEFAULT_OLLAMA_BASE_URL)
    val ollamaChatModelName =
        if (provider == "ollama") {
            firstNonBlank(settings.llmModelName, System.getenv("LLM_MODEL"), DEFAULT_OLLAMA_CHAT_MODEL)
        } else {
            firstNonBlank(System.getenv("OLLAMA_MODEL"), DEFAULT_OLLAMA_CHAT_MODEL)
        }
    val ollamaEmbeddingModelName =
        if (provider == "ollama") {
            firstNonBlank(settings.embeddingModelName, System.getenv("EMBEDDING_MODEL"), DEFAULT_OLLAMA_EMBEDDING_MODEL)
        } else {
            firstNonBlank(System.getenv("OLLAMA_EMBED_MODEL"), DEFAULT_OLLAMA_EMBEDDING_MODEL)
        }

    val neo4jConfig =
        Neo4jConfig(
            uri = nonBlankOrNull(System.getenv("NEO4J_URI")),
            username = nonBlankOrNull(System.getenv("NEO4J_USERNAME")),
            password = nonBlankOrNull(System.getenv("NEO4J_PASSWORD")),
            database = nonBlankOrNull(System.getenv("NEO4J_DATABASE")),
        )

    return LightRagConfig(
        provider = provider,
        openai =
            OpenAiConfig(
                apiKey = openAiApiKey,
                chatModelName = openAiChatModelName,
                embeddingModelName = openAiEmbeddingModelName,
                embeddingModelDimensions = openAiEmbeddingModelDimensions,
                baseUrl = openAiBaseUrl,
            ),
        ollama =
            OllamaConfig(
                baseUrl = ollamaBaseUrl,
                chatModelName = ollamaChatModelName,
                embeddingModelName = ollamaEmbeddingModelName,
            ),
        neo4j = neo4jConfig,
        mongodb =
            MongoDbConfig(
                uri = firstNonBlank(System.getenv("MONGO_URI"), "mongodb://localhost:27017"),
                database = firstNonBlank(System.getenv("MONGO_DB"), "lightrag"),
            ),
        storage =
            StorageConfig(
                workingDir = firstNonBlank(settings.workingDir, "./rag_storage"),
                graphStorageName = firstNonBlank(settings.graphStorageName, "InMemoryGraphStorage"),
                vectorStorageName = firstNonBlank(settings.vectorStorageName, "InMemoryVectorStorage"),
            ),
        addonConfig =
            AddonConfigConfig(
                chunkTokenSize = settings.chunkTokenSize ?: 1200,
                chunkOverlapTokenSize = settings.chunkOverlapTokenSize ?: 100,
                cosineBetterThreshold = settings.cosineBetterThreshold ?: 0.2,
                entityTypes =
                    settings.entityTypes
                        ?.takeIf { it.isNotEmpty() }
                        ?: listOf("Person", "Organization", "Location", "Event", "Concept"),
                language = firstNonBlank(settings.language, "English"),
            ),
        resetStorage = parseBoolean(System.getenv("LIGHTRAG_RESET_STORAGE")) ?: false,
    )
}

private fun normalizeProvider(raw: String): String = if (raw.trim().lowercase() == "ollama") "ollama" else "openai"

private fun nonBlankOrNull(value: String?): String? = value?.trim()?.takeIf { it.isNotEmpty() }

private fun firstNonBlankOrNull(vararg values: String?): String? {
    for (value in values) {
        val normalized = value?.trim()
        if (!normalized.isNullOrEmpty()) {
            return normalized
        }
    }
    return null
}

private fun firstNonBlank(vararg values: String?): String {
    for (value in values) {
        val normalized = value?.trim()
        if (!normalized.isNullOrEmpty()) {
            return normalized
        }
    }
    error("Expected at least one non-empty fallback value")
}

private fun parseBoolean(raw: String?): Boolean? =
    when (raw?.trim()?.lowercase()) {
        "1", "true", "yes", "on" -> true
        "0", "false", "no", "off" -> false
        else -> null
    }

private fun inferEmbeddingDimension(modelName: String): Int {
    val normalized = modelName.lowercase()
    val inferred =
        when {
            normalized.contains("text-embedding-3-large") -> 3072
            normalized.contains("text-embedding-3-small") -> 1536
            normalized.contains("text-embedding-ada-002") -> 1536
            normalized.contains("nomic-embed-text") -> 768
            else -> null
        }
    if (inferred != null) {
        return inferred
    }
    logger.warn {
        "Unrecognized embedding model '$modelName'; defaulting dimension to $DEFAULT_OPENAI_EMBEDDING_DIMENSION. " +
            "Set 'embeddingModelDimensions' in common config or OPENAI_EMBEDDING_DIMENSION/EMBEDDING_DIMENSION env vars."
    }
    return DEFAULT_OPENAI_EMBEDDING_DIMENSION
}

private fun resolveEmbeddingDimension(
    settings: LightRagSettings,
    embeddingModelName: String,
): Int =
    settings.embeddingModelDimensions
        ?: System.getenv("OPENAI_EMBEDDING_DIMENSION")?.toIntOrNull()
        ?: System.getenv("EMBEDDING_DIMENSION")?.toIntOrNull()
        ?: inferEmbeddingDimension(embeddingModelName)
