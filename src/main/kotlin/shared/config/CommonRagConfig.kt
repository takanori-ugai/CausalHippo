package shared.config

import hipporag.config.BaseConfig
import hipporag.utils.applyConfigOverrides
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.longOrNull
import java.nio.file.Files
import java.nio.file.Path

/**
 * Canonical multi-module JSON config shape:
 *
 * {
 *   "shared": { ... },
 *   "unified": { ... },
 *   "causalrag": { ... },
 *   "hipporag": { ... },
 *   "pathrag": { ... },
 *   "lightrag": { ... }
 * }
 *
 * Each module-specific section overrides equivalent values from [shared].
 *
 * @property shared defaults shared across all modules.
 * @property unified overrides/defaults for unified API + persistence SPI wiring.
 * @property causalrag overrides for CausalRAG [CaualRagConfig] resolution.
 * @property hipporag overrides for HippoRAG [BaseConfig] resolution.
 * @property pathrag overrides for PathRAG runtime/config resolution.
 * @property lightrag overrides for LightRAG runtime/config resolution.
 */
data class CommonRagConfig(
    val shared: JsonObject = JsonObject(emptyMap()),
    val unified: JsonObject = JsonObject(emptyMap()),
    val causalrag: JsonObject = JsonObject(emptyMap()),
    val hipporag: JsonObject = JsonObject(emptyMap()),
    val pathrag: JsonObject = JsonObject(emptyMap()),
    val lightrag: JsonObject = JsonObject(emptyMap()),
) {
    /**
     * Extracts common LLM/embedding settings from the [shared] section only.
     *
     * @return normalized shared model settings.
     */
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
     * Resolves CausalRAG [CaualRagConfig] from this common config.
     *
     * Values from [causalrag] override [shared], including alias forms.
     *
     * @return merged [CaualRagConfig] for CausalRAG.
     */
    fun toCaualRagConfig(): CaualRagConfig {
        val merged = mutableMapOf<String, JsonElement>()
        merged.setFromFirst("modelName", shared, "modelName", "llmModel", "llmName")
        merged.setFromFirst("embeddingModel", shared, "embeddingModel", "embeddingModelName")
        merged.setFromFirst("llmProvider", shared, "llmProvider", "provider")
        merged.setFromFirst("llmApiKey", shared, "llmApiKey", "apiKey", "openAiApiKey")
        merged.setFromFirst("llmBaseUrl", shared, "llmBaseUrl", "baseUrl")
        merged.setFromFirst("embeddingApiKey", shared, "embeddingApiKey")
        merged.setFromFirst(
            "ingestChunkTokenSize",
            shared,
            "ingestChunkTokenSize",
            "chunkTokenSize",
            "chunk_token_size",
        )
        merged.setFromFirst(
            "ingestChunkOverlapTokenSize",
            shared,
            "ingestChunkOverlapTokenSize",
            "chunkOverlapTokenSize",
            "chunk_overlap_token_size",
        )
        merged.setFromFirst(
            "promptContextTokenBudget",
            shared,
            "promptContextTokenBudget",
            "contextTokenBudget",
            "context_token_budget",
            "prompt_context_token_budget",
        )
        merged.putAll(causalrag)
        // Re-apply canonical keys from module section so module aliases override shared values.
        merged.setFromFirst("modelName", causalrag, "modelName", "llmModel", "llmName")
        merged.setFromFirst("embeddingModel", causalrag, "embeddingModel", "embeddingModelName")
        merged.setFromFirst("llmProvider", causalrag, "llmProvider", "provider")
        merged.setFromFirst("llmApiKey", causalrag, "llmApiKey", "apiKey", "openAiApiKey")
        merged.setFromFirst("llmBaseUrl", causalrag, "llmBaseUrl", "baseUrl")
        merged.setFromFirst("embeddingApiKey", causalrag, "embeddingApiKey")
        merged.setFromFirst(
            "ingestChunkTokenSize",
            causalrag,
            "ingestChunkTokenSize",
            "chunkTokenSize",
            "chunk_token_size",
        )
        merged.setFromFirst(
            "ingestChunkOverlapTokenSize",
            causalrag,
            "ingestChunkOverlapTokenSize",
            "chunkOverlapTokenSize",
            "chunk_overlap_token_size",
        )
        merged.setFromFirst(
            "promptContextTokenBudget",
            causalrag,
            "promptContextTokenBudget",
            "contextTokenBudget",
            "context_token_budget",
            "prompt_context_token_budget",
        )
        return JSON.decodeFromJsonElement(CaualRagConfig.serializer(), JsonObject(merged))
    }

    @Deprecated(
        message = "Use toCaualRagConfig()",
        replaceWith = ReplaceWith("toCaualRagConfig()"),
    )
    fun toCausalRagSettings(): CaualRagConfig = toCaualRagConfig()

    @Deprecated(
        message = "Use toCaualRagConfig()",
        replaceWith = ReplaceWith("toCaualRagConfig()"),
    )
    fun toPipelineConfig(): CaualRagConfig = toCaualRagConfig()

    /**
     * Resolves HippoRAG [BaseConfig] from this common config.
     *
     * Values from [hipporag] override [shared].
     *
     * @param base base configuration to apply overrides onto.
     * @return merged HippoRAG configuration.
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
     * Resolves PathRAG config from this common config.
     *
     * Values from [pathrag] override [shared], including alias forms.
     *
     * @return merged PathRAG config.
     */
    fun toPathRagConfig(): PathRagConfig {
        val merged = mutableMapOf<String, JsonElement>()
        merged.setFromFirst("llmProvider", shared, "llmProvider", "provider")
        merged.setFromFirst("llmModelName", shared, "modelName", "llmModel", "llmName")
        merged.setFromFirst("embeddingModelName", shared, "embeddingModel", "embeddingModelName")
        merged.setFromFirst("apiKey", shared, "llmApiKey", "apiKey", "openAiApiKey")
        merged.setFromFirst("baseUrl", shared, "llmBaseUrl", "baseUrl")
        merged.putAll(pathrag)
        // Re-apply canonical keys from module section so module aliases override shared values.
        merged.setFromFirst("llmProvider", pathrag, "llmProvider", "provider")
        merged.setFromFirst("llmModelName", pathrag, "llmModelName", "llmModel", "modelName", "openaiModel")
        merged.setFromFirst(
            "embeddingModelName",
            pathrag,
            "embeddingModelName",
            "embeddingModel",
            "openaiEmbeddingModel",
        )
        merged.setFromFirst("apiKey", pathrag, "apiKey", "llmApiKey", "openAiApiKey")
        merged.setFromFirst("baseUrl", pathrag, "baseUrl", "llmBaseUrl", "openAiApiBase")
        val obj = JsonObject(merged)
        return PathRagConfig(
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

    @Deprecated(
        message = "Use toPathRagConfig()",
        replaceWith = ReplaceWith("toPathRagConfig()"),
    )
    fun toPathRagSettings(): PathRagConfig = toPathRagConfig()

    /**
     * Resolves LightRAG runtime config from this common config.
     *
     * Values from [lightrag] override [shared], including alias forms.
     *
     * @return merged LightRAG config.
     */
    fun toLightRagConfig(): LightRagConfig {
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
        // Re-apply canonical keys from module section so module aliases override shared values.
        merged.setFromFirst("llmProvider", lightrag, "llmProvider", "provider")
        merged.setFromFirst("llmModelName", lightrag, "llmModelName", "llmModel", "modelName")
        merged.setFromFirst("embeddingModelName", lightrag, "embeddingModelName", "embeddingModel")
        merged.setFromFirst(
            "embeddingModelDimensions",
            lightrag,
            "embeddingModelDimensions",
            "embeddingDimensions",
            "embedding_dimension",
        )
        merged.setFromFirst("apiKey", lightrag, "apiKey", "llmApiKey", "openAiApiKey")
        merged.setFromFirst("baseUrl", lightrag, "baseUrl", "llmBaseUrl")
        val obj = JsonObject(merged)
        return LightRagConfig(
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

    @Deprecated(
        message = "Use toLightRagConfig()",
        replaceWith = ReplaceWith("toLightRagConfig()"),
    )
    fun toLightRagSettings(): LightRagConfig = toLightRagConfig()

    /**
     * Resolves unified persistence defaults from the [unified] section.
     *
     * Supported keys:
     * - `useUnifiedPersistence`
     * - `useUnifiedSpiForRetrievalAndIndex`
     * - `persistenceBackend`
     * - `persistenceRootDir`
     * - `persistenceConfig`
     *
     * @return map suitable for merging with unified factory overrides.
     */
    fun unifiedPersistenceOverrides(): Map<String, Any?> {
        val overrides = linkedMapOf<String, Any?>()
        firstBoolean(unified, "useUnifiedPersistence")?.let { overrides["useUnifiedPersistence"] = it }
        firstBoolean(unified, "useUnifiedSpiForRetrievalAndIndex")?.let {
            overrides["useUnifiedSpiForRetrievalAndIndex"] = it
        }
        firstString(unified, "persistenceBackend")?.let { overrides["persistenceBackend"] = it }
        firstString(unified, "persistenceRootDir")?.let { overrides["persistenceRootDir"] = it }
        firstObject(unified, "persistenceConfig")?.let { cfg ->
            val map = cfg.toAnyMap()
            if (map.isNotEmpty()) {
                overrides["persistenceConfig"] = map
            }
        }
        return overrides
    }
}

/**
 * Shared model/credential settings resolved from common configuration.
 *
 * @property provider LLM provider identifier (for example `openai` or `ollama`).
 * @property llmModelName model name used for generation requests.
 * @property embeddingModelName model name used for embedding requests.
 * @property apiKey primary API key for LLM calls.
 * @property embeddingApiKey optional dedicated API key for embeddings.
 * @property baseUrl optional provider base URL override.
 */
data class SharedModelSettings(
    val provider: String?,
    val llmModelName: String?,
    val embeddingModelName: String?,
    val apiKey: String?,
    val embeddingApiKey: String?,
    val baseUrl: String?,
)

/**
 * CausalRAG construction/runtime settings resolved from common configuration.
 *
 * This shared DTO intentionally avoids any dependency on `causalrag` module types.
 */
@Serializable
data class CaualRagConfig(
    val modelName: String? = null,
    val embeddingModel: String? = null,
    val graphPath: String? = null,
    val indexPath: String? = null,
    val llmProvider: String? = null,
    val llmApiKey: String? = null,
    val llmBaseUrl: String? = null,
    val embeddingApiKey: String? = null,
    val templateStyle: String? = null,
    val semanticMode: String? = null,
    val minCausalMatches: Int? = null,
    val ingestChunkTokenSize: Int? = null,
    val ingestChunkOverlapTokenSize: Int? = null,
    val promptContextTokenBudget: Int? = null,
)

/**
 * PathRAG construction and runtime settings resolved from common configuration.
 *
 * @property workingDir working directory used by PathRAG storage.
 * @property kvStorage key-value storage backend name.
 * @property vectorStorage vector storage backend name.
 * @property graphStorage graph storage backend name.
 * @property chunkTokenSize maximum chunk size in tokens.
 * @property chunkOverlapTokenSize overlap size between adjacent chunks in tokens.
 * @property language language hint consumed by PathRAG prompts.
 * @property llmProvider LLM provider identifier.
 * @property llmModelName model name used for generation.
 * @property embeddingModelName model name used for embedding.
 * @property apiKey API key used by the provider.
 * @property baseUrl provider base URL.
 * @property ollamaBaseUrl Ollama endpoint override.
 * @property ollamaModelName Ollama generation model override.
 * @property ollamaEmbeddingModelName Ollama embedding model override.
 */
data class PathRagConfig(
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
     * Converts PathRAG config into runtime key-values consumed by PathRAG/LLM helpers.
     *
     * Empty and blank values are excluded from the returned map.
     *
     * @return runtime settings map keyed by PathRAG environment variable names.
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
     * Backward-compatible bridge for legacy call sites that still rely on JVM system properties.
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

/**
 * LightRAG runtime settings resolved from common configuration.
 *
 * @property provider LLM provider identifier.
 * @property llmModelName model name used for generation.
 * @property embeddingModelName model name used for embedding.
 * @property embeddingModelDimensions embedding vector dimensionality.
 * @property apiKey API key used by the provider.
 * @property baseUrl provider base URL.
 * @property workingDir LightRAG working directory.
 * @property graphStorageName graph storage backend name.
 * @property vectorStorageName vector storage backend name.
 * @property chunkTokenSize maximum chunk size in tokens.
 * @property chunkOverlapTokenSize overlap size between adjacent chunks in tokens.
 * @property entityTypes optional entity type allow-list.
 * @property language language hint used in extraction/query prompts.
 * @property cosineBetterThreshold similarity threshold tuning parameter.
 */
data class LightRagConfig(
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

/**
 * Loader/parsing helpers for the canonical multi-module common config JSON document.
 */
object CommonRagConfigLoader {
    /**
     * Parses [content] into [CommonRagConfig] when it matches the expected multi-module shape.
     *
     * @param content JSON config text.
     * @return parsed config, or `null` when the payload is invalid/non-common-config JSON.
     */
    fun parseOrNull(content: String): CommonRagConfig? {
        val root = runCatching { JSON.parseToJsonElement(content) }.getOrNull() as? JsonObject ?: return null
        if (!looksLikeCommonConfig(root)) return null
        return CommonRagConfig(
            shared = root["shared"] as? JsonObject ?: JsonObject(emptyMap()),
            unified = root["unified"] as? JsonObject ?: JsonObject(emptyMap()),
            causalrag = root["causalrag"] as? JsonObject ?: JsonObject(emptyMap()),
            hipporag = root["hipporag"] as? JsonObject ?: JsonObject(emptyMap()),
            pathrag = root["pathrag"] as? JsonObject ?: JsonObject(emptyMap()),
            lightrag = root["lightrag"] as? JsonObject ?: JsonObject(emptyMap()),
        )
    }

    /**
     * Loads and parses common config JSON from [path].
     *
     * @param path path to a JSON file.
     * @return parsed common config.
     * @throws IllegalStateException when file content is not a valid common config document.
     */
    fun load(path: Path): CommonRagConfig {
        val text = Files.readString(path)
        return parseOrNull(text) ?: error("Not a common config JSON file: $path")
    }

    /**
     * Loads and parses common config JSON from [path].
     *
     * @param path path string to a JSON file.
     * @return parsed common config.
     */
    fun load(path: String): CommonRagConfig = load(Path.of(path))

    private fun looksLikeCommonConfig(root: JsonObject): Boolean =
        root.containsKey("shared") ||
            root.containsKey("unified") ||
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

private fun firstBoolean(
    source: JsonObject,
    vararg keys: String,
): Boolean? {
    for (key in keys) {
        val value = source[key] as? JsonPrimitive ?: continue
        value.booleanOrNull?.let { return it }
    }
    return null
}

private fun firstObject(
    source: JsonObject,
    vararg keys: String,
): JsonObject? {
    for (key in keys) {
        val value = source[key] as? JsonObject ?: continue
        return value
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

private fun JsonObject.toAnyMap(): Map<String, Any?> = this.entries.associate { (k, v) -> k to v.toAnyValue() }

private fun JsonElement.toAnyValue(): Any? =
    when (this) {
        is JsonObject -> this.toAnyMap()
        is JsonArray -> this.map { it.toAnyValue() }
        is JsonPrimitive ->
            when {
                this.toString().startsWith("\"") -> this.contentOrNull
                this.booleanOrNull != null -> this.booleanOrNull
                this.intOrNull != null -> this.intOrNull
                this.longOrNull != null -> this.longOrNull
                this.doubleOrNull != null -> this.doubleOrNull
                else -> this.contentOrNull
            }
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
