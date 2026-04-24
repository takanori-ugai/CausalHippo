package shared.config

import causalrag.PipelineConfig
import hipporag.config.BaseConfig
import hipporag.utils.applyConfigOverrides
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import java.nio.file.Files
import java.nio.file.Path

/**
 * Canonical multi-module JSON config shape:
 *
 * {
 *   "shared": { ... },
 *   "causalrag": { ... },
 *   "hipporag": { ... },
 *   "pathrag": { ... },
 *   "lightrag": { ... }
 * }
 */
data class CommonRagConfig(
    val shared: JsonObject = JsonObject(emptyMap()),
    val causalrag: JsonObject = JsonObject(emptyMap()),
    val hipporag: JsonObject = JsonObject(emptyMap()),
    val pathrag: JsonObject = JsonObject(emptyMap()),
    val lightrag: JsonObject = JsonObject(emptyMap()),
) {
    fun sharedModelSettings(): SharedModelSettings =
        SharedModelSettings(
            provider = firstString(shared, "llmProvider", "provider"),
            llmModelName = firstString(shared, "modelName", "llmModel", "llmName"),
            embeddingModelName = firstString(shared, "embeddingModel", "embeddingModelName"),
            apiKey = firstString(shared, "llmApiKey", "apiKey", "openAiApiKey"),
            embeddingApiKey = firstString(shared, "embeddingApiKey"),
            baseUrl = firstString(shared, "llmBaseUrl", "baseUrl"),
        )

    /**
     * Resolve causalrag [PipelineConfig] from common config.
     * Section values override shared values.
     */
    fun toPipelineConfig(): PipelineConfig {
        val merged = mutableMapOf<String, JsonElement>()
        merged.setFromFirst("modelName", shared, "modelName", "llmModel", "llmName")
        merged.setFromFirst("embeddingModel", shared, "embeddingModel", "embeddingModelName")
        merged.setFromFirst("llmProvider", shared, "llmProvider", "provider")
        merged.setFromFirst("llmApiKey", shared, "llmApiKey", "apiKey", "openAiApiKey")
        merged.setFromFirst("llmBaseUrl", shared, "llmBaseUrl", "baseUrl")
        merged.setFromFirst("embeddingApiKey", shared, "embeddingApiKey")
        merged.putAll(causalrag)
        return JSON.decodeFromJsonElement(PipelineConfig.serializer(), JsonObject(merged))
    }

    /**
     * Resolve HippoRAG [BaseConfig] from common config.
     * Section values override shared values.
     */
    fun toHippoBaseConfig(base: BaseConfig = BaseConfig()): BaseConfig {
        val merged = mutableMapOf<String, JsonElement>()
        merged.setFromFirst("llmName", shared, "modelName", "llmModel", "llmName")
        merged.setFromFirst("embeddingModelName", shared, "embeddingModel", "embeddingModelName")
        merged.setFromFirst("llmProvider", shared, "llmProvider", "provider")
        merged.setFromFirst("openAiApiKey", shared, "llmApiKey", "apiKey", "openAiApiKey")
        merged.setFromFirst("llmBaseUrl", shared, "llmBaseUrl", "baseUrl")
        merged.setFromFirst("embeddingBaseUrl", shared, "embeddingBaseUrl", "llmBaseUrl", "baseUrl")
        merged.putAll(hipporag)
        return applyConfigOverrides(base, JsonObject(merged))
    }

    /**
     * Resolve PathRAG construction + runtime properties from common config.
     * Section values override shared values.
     */
    fun toPathRagSettings(): PathRagSettings {
        val merged = mutableMapOf<String, JsonElement>()
        merged.setFromFirst("llmProvider", shared, "llmProvider", "provider")
        merged.setFromFirst("llmModelName", shared, "modelName", "llmModel", "llmName")
        merged.setFromFirst("embeddingModelName", shared, "embeddingModel", "embeddingModelName")
        merged.setFromFirst("apiKey", shared, "llmApiKey", "apiKey", "openAiApiKey")
        merged.setFromFirst("baseUrl", shared, "llmBaseUrl", "baseUrl")
        merged.putAll(pathrag)
        val obj = JsonObject(merged)
        return PathRagSettings(
            workingDir = firstString(obj, "workingDir", "working_dir"),
            kvStorage = firstString(obj, "kvStorage", "kv_storage"),
            vectorStorage = firstString(obj, "vectorStorage", "vector_storage"),
            graphStorage = firstString(obj, "graphStorage", "graph_storage"),
            chunkTokenSize = firstInt(obj, "chunkTokenSize", "chunk_token_size"),
            chunkOverlapTokenSize = firstInt(obj, "chunkOverlapTokenSize", "chunk_overlap_token_size"),
            language = firstString(obj, "language"),
            llmProvider = firstString(obj, "llmProvider", "provider"),
            llmModelName = firstString(obj, "llmModelName", "llmModel", "modelName", "openaiModel"),
            embeddingModelName =
                firstString(
                    obj,
                    "embeddingModelName",
                    "embeddingModel",
                    "openaiEmbeddingModel",
                ),
            apiKey = firstString(obj, "apiKey", "llmApiKey", "openAiApiKey"),
            baseUrl = firstString(obj, "baseUrl", "llmBaseUrl", "openAiApiBase"),
            ollamaBaseUrl = firstString(obj, "ollamaBaseUrl", "ollama_base_url"),
            ollamaModelName = firstString(obj, "ollamaModelName", "ollama_model_name"),
            ollamaEmbeddingModelName =
                firstString(
                    obj,
                    "ollamaEmbeddingModelName",
                    "ollama_embedding_model_name",
                ),
        )
    }

    /**
     * Resolve LightRAG runtime settings from common config.
     * Section values override shared values.
     */
    fun toLightRagSettings(): LightRagSettings {
        val merged = mutableMapOf<String, JsonElement>()
        merged.setFromFirst("llmProvider", shared, "llmProvider", "provider")
        merged.setFromFirst("llmModelName", shared, "modelName", "llmModel", "llmName")
        merged.setFromFirst("embeddingModelName", shared, "embeddingModel", "embeddingModelName")
        merged.setFromFirst(
            "embeddingModelDimensions",
            shared,
            "embeddingModelDimensions",
            "embeddingDimensions",
            "embedding_dimension",
        )
        merged.setFromFirst("apiKey", shared, "llmApiKey", "apiKey", "openAiApiKey")
        merged.setFromFirst("baseUrl", shared, "llmBaseUrl", "baseUrl")
        merged.putAll(lightrag)
        val obj = JsonObject(merged)
        return LightRagSettings(
            provider = firstString(obj, "llmProvider", "provider"),
            llmModelName = firstString(obj, "llmModelName", "llmModel", "modelName"),
            embeddingModelName = firstString(obj, "embeddingModelName", "embeddingModel"),
            embeddingModelDimensions = firstInt(obj, "embeddingModelDimensions", "embeddingDimensions", "embedding_dimension"),
            apiKey = firstString(obj, "apiKey", "llmApiKey", "openAiApiKey"),
            baseUrl = firstString(obj, "baseUrl", "llmBaseUrl"),
            workingDir = firstString(obj, "workingDir", "working_dir"),
            graphStorageName = firstString(obj, "graphStorageName", "graph_storage_name"),
            vectorStorageName = firstString(obj, "vectorStorageName", "vector_storage_name"),
            chunkTokenSize = firstInt(obj, "chunkTokenSize", "chunk_token_size"),
            chunkOverlapTokenSize = firstInt(obj, "chunkOverlapTokenSize", "chunk_overlap_token_size"),
            entityTypes = firstStringList(obj, "entityTypes", "entity_types"),
            language = firstString(obj, "language"),
            cosineBetterThreshold = firstDouble(obj, "cosineBetterThreshold", "cosine_better_threshold"),
        )
    }
}

data class SharedModelSettings(
    val provider: String?,
    val llmModelName: String?,
    val embeddingModelName: String?,
    val apiKey: String?,
    val embeddingApiKey: String?,
    val baseUrl: String?,
)

data class PathRagSettings(
    val workingDir: String?,
    val kvStorage: String?,
    val vectorStorage: String?,
    val graphStorage: String?,
    val chunkTokenSize: Int?,
    val chunkOverlapTokenSize: Int?,
    val language: String?,
    val llmProvider: String?,
    val llmModelName: String?,
    val embeddingModelName: String?,
    val apiKey: String?,
    val baseUrl: String?,
    val ollamaBaseUrl: String?,
    val ollamaModelName: String?,
    val ollamaEmbeddingModelName: String?,
) {
    /**
     * Convert PathRAG settings into runtime key-values consumed by PathRAG/LLM helpers.
     */
    fun toRuntimeSettingsMap(): Map<String, String> =
        buildMap {
            putIfNonBlank("LLM_PROVIDER", llmProvider)
            putIfNonBlank("OPENAI_MODEL", llmModelName)
            putIfNonBlank("OPENAI_EMBEDDING_MODEL", embeddingModelName)
            putIfNonBlank("OPENAI_API_KEY", apiKey)
            putIfNonBlank("OPENAI_API_BASE", baseUrl)
            putIfNonBlank("OLLAMA_BASE_URL", ollamaBaseUrl ?: baseUrl)
            putIfNonBlank("OLLAMA_MODEL", ollamaModelName ?: llmModelName)
            putIfNonBlank("OLLAMA_EMBED_MODEL", ollamaEmbeddingModelName ?: embeddingModelName)
            putIfNonBlank("LANGUAGE", language)
        }

    /**
     * Backward-compatible bridge for legacy call sites that still rely on process-wide properties.
     */
    @Deprecated(
        message = "Mutates JVM-global state. Prefer toRuntimeSettingsMap() and pass settings directly to PathRAG.",
        replaceWith = ReplaceWith("toRuntimeSettingsMap()"),
    )
    fun applyAsSystemProperties() {
        toRuntimeSettingsMap().forEach { (key, value) ->
            System.setProperty(key, value)
        }
    }
}

data class LightRagSettings(
    val provider: String?,
    val llmModelName: String?,
    val embeddingModelName: String?,
    val embeddingModelDimensions: Int?,
    val apiKey: String?,
    val baseUrl: String?,
    val workingDir: String?,
    val graphStorageName: String?,
    val vectorStorageName: String?,
    val chunkTokenSize: Int?,
    val chunkOverlapTokenSize: Int?,
    val entityTypes: List<String>?,
    val language: String?,
    val cosineBetterThreshold: Double?,
)

object CommonRagConfigLoader {
    fun parseOrNull(content: String): CommonRagConfig? {
        val root = runCatching { JSON.parseToJsonElement(content) }.getOrNull() as? JsonObject ?: return null
        if (!looksLikeCommonConfig(root)) return null
        return CommonRagConfig(
            shared = root["shared"] as? JsonObject ?: JsonObject(emptyMap()),
            causalrag = root["causalrag"] as? JsonObject ?: JsonObject(emptyMap()),
            hipporag = root["hipporag"] as? JsonObject ?: JsonObject(emptyMap()),
            pathrag = root["pathrag"] as? JsonObject ?: JsonObject(emptyMap()),
            lightrag = root["lightrag"] as? JsonObject ?: JsonObject(emptyMap()),
        )
    }

    fun load(path: Path): CommonRagConfig {
        val text = Files.readString(path)
        return parseOrNull(text) ?: error("Not a common config JSON file: $path")
    }

    fun load(path: String): CommonRagConfig = load(Path.of(path))

    private fun looksLikeCommonConfig(root: JsonObject): Boolean =
        root.containsKey("shared") ||
            root.containsKey("causalrag") ||
            root.containsKey("hipporag") ||
            root.containsKey("pathrag") ||
            root.containsKey("lightrag")
}

private val JSON =
    Json {
        ignoreUnknownKeys = true
    }

private fun MutableMap<String, JsonElement>.setFromFirst(
    targetKey: String,
    source: JsonObject,
    vararg sourceKeys: String,
) {
    val value = firstElement(source, *sourceKeys) ?: return
    this[targetKey] = value
}

private fun firstElement(
    source: JsonObject,
    vararg keys: String,
): JsonElement? {
    for (key in keys) {
        if (source.containsKey(key)) {
            return source[key]
        }
    }
    return null
}

private fun firstString(
    source: JsonObject,
    vararg keys: String,
): String? {
    for (key in keys) {
        val value = source[key] as? JsonPrimitive ?: continue
        value.contentOrNull?.let { return it }
    }
    return null
}

private fun firstInt(
    source: JsonObject,
    vararg keys: String,
): Int? {
    for (key in keys) {
        val value = source[key] as? JsonPrimitive ?: continue
        value.intOrNull?.let { return it }
    }
    return null
}

private fun firstDouble(
    source: JsonObject,
    vararg keys: String,
): Double? {
    for (key in keys) {
        val value = source[key] as? JsonPrimitive ?: continue
        value.doubleOrNull?.let { return it }
    }
    return null
}

private fun firstStringList(
    source: JsonObject,
    vararg keys: String,
): List<String>? {
    for (key in keys) {
        val arr = source[key] as? JsonArray ?: continue
        return arr.mapNotNull { (it as? JsonPrimitive)?.contentOrNull }
    }
    return null
}

private fun MutableMap<String, String>.putIfNonBlank(
    key: String,
    value: String?,
) {
    val normalized = value?.trim()
    if (!normalized.isNullOrEmpty()) {
        this[key] = normalized
    }
}
