package shared.rag.unified

import causalhippo.CausalHippoRAG
import causalrag.CausalRAG
import causalrag.retriever.HippoRagSemanticMode
import com.microsoft.graphrag.GraphRAG
import hipporag.HippoRAG
import hipporag.config.BaseConfig
import lightrag.core.LightRAG
import lightrag.core.Neo4jConfig
import lightrag.di.createLightRagRuntime
import pathrag.PathRAG
import pathrag.base.ExtraConfig
import shared.config.CommonRagConfigLoader
import shared.rag.spi.persistence.PersistenceSession
import java.nio.file.Files
import java.nio.file.Path

/**
 * Factory entrypoint for creating unified RAG adapters.
 */
object UnifiedRagFactory {
    fun create(
        ragId: RagId,
        configPath: String? = null,
        overrides: Map<String, Any?> = emptyMap(),
    ): UnifiedRagHandle {
        val resolvedOverrides = applyUnifiedPersistenceDefaults(configPath, overrides)
        val useUnifiedPersistence = resolvedOverrides.bool("useUnifiedPersistence") ?: false
        val session = if (useUnifiedPersistence) UnifiedPersistenceFactory.openSession(resolvedOverrides) else null
        val effectiveOverrides =
            if (session == null) {
                resolvedOverrides
            } else {
                resolvedOverrides +
                    mapOf(
                        "__persistenceSession" to session,
                        "__useUnifiedSpiForRetrievalAndIndex" to
                            (resolvedOverrides.bool("useUnifiedSpiForRetrievalAndIndex") ?: true),
                    )
            }

        try {
            val baseHandle =
                when (ragId) {
                    RagId.CAUSAL_RAG -> createCausalRag(configPath, effectiveOverrides)
                    RagId.CAUSAL_HIPPO_RAG -> createCausalHippoRag(configPath, effectiveOverrides)
                    RagId.HIPPO_RAG -> createHippoRag(configPath, effectiveOverrides)
                    RagId.PATH_RAG -> createPathRag(configPath, effectiveOverrides)
                    RagId.LIGHT_RAG -> createLightRag(configPath, effectiveOverrides)
                    RagId.GRAPH_RAG -> createGraphRag(effectiveOverrides)
                }
            if (session == null) return baseHandle

            val shouldWrapWithGenericPersistenceAdapter =
                ragId != RagId.CAUSAL_RAG && ragId != RagId.LIGHT_RAG && ragId != RagId.PATH_RAG
            return if (shouldWrapWithGenericPersistenceAdapter) {
                baseHandle.copy(
                    rag =
                        UnifiedPersistenceAdapter(
                            ragId = ragId,
                            delegate = baseHandle.rag,
                            session = session,
                        ),
                    persistence = session,
                )
            } else {
                baseHandle.copy(persistence = session)
            }
        } catch (ex: Throwable) {
            session?.close()
            throw ex
        }
    }

    private fun createCausalRag(
        configPath: String?,
        overrides: Map<String, Any?>,
    ): UnifiedRagHandle {
        val effective = withPersistenceBridgeDefaults(RagId.CAUSAL_RAG, overrides)
        val persistenceSession = effective.persistenceSession("__persistenceSession")
        val useUnifiedSpi = effective.bool("__useUnifiedSpiForRetrievalAndIndex") ?: false
        val rag =
            CausalRAG(
                modelName = effective.string("modelName") ?: "gpt-4",
                embeddingModel = effective.string("embeddingModel") ?: "all-MiniLM-L6-v2",
                graphPath = effective.string("graphPath"),
                indexPath = effective.string("indexPath"),
                configPath = configPath ?: effective.string("configPath"),
                templateStyle = effective.string("templateStyle"),
                embeddingApiKey = effective.string("embeddingApiKey"),
                dynamicWeightingEnabled = effective.bool("dynamicWeightingEnabled") ?: false,
                twoPassAdaptiveEnabled = effective.bool("twoPassAdaptiveEnabled") ?: false,
                confidenceBasedSwitchEnabled = effective.bool("confidenceBasedSwitchEnabled") ?: false,
                persistenceSession = persistenceSession,
                useUnifiedSpiForRetrievalAndIndex = useUnifiedSpi,
                persistenceNamespacePrefix = effective.string("persistenceNamespacePrefix") ?: "causalrag",
            )
        val adapter = CausalRagUnifiedAdapter(rag)
        return UnifiedRagHandle(
            id = RagId.CAUSAL_RAG,
            capabilities = CausalRagUnifiedAdapter.CAPABILITIES,
            rag = adapter,
        )
    }

    private fun createCausalHippoRag(
        configPath: String?,
        overrides: Map<String, Any?>,
    ): UnifiedRagHandle {
        val effective = withPersistenceBridgeDefaults(RagId.CAUSAL_HIPPO_RAG, overrides)
        val effectiveConfigPath = configPath ?: effective.string("configPath")
        val rag =
            CausalHippoRAG(
                modelName = effective.string("modelName") ?: "gpt-4o-mini",
                embeddingModel = effective.string("embeddingModel") ?: "text-embedding-3-small",
                configPath = effectiveConfigPath,
                templateStyle = effective.string("templateStyle"),
                hippoConfig = buildHippoBaseConfig(effectiveConfigPath, effective),
                hippoSemanticMode = parseHippoSemanticMode(effective.string("hippoSemanticMode")),
                semanticWeight = effective.double("semanticWeight") ?: 0.4,
                causalWeight = effective.double("causalWeight") ?: 0.5,
                bm25Weight = effective.double("bm25Weight") ?: 0.1,
                minCausalMatches = effective.int("minCausalMatches") ?: 0,
                dynamicWeightingEnabled = effective.bool("dynamicWeightingEnabled") ?: false,
                twoPassAdaptiveEnabled = effective.bool("twoPassAdaptiveEnabled") ?: false,
                confidenceBasedSwitchEnabled = effective.bool("confidenceBasedSwitchEnabled") ?: false,
            )
        val adapter = CausalHippoRagUnifiedAdapter(rag)
        return UnifiedRagHandle(
            id = RagId.CAUSAL_HIPPO_RAG,
            capabilities = CausalHippoRagUnifiedAdapter.CAPABILITIES,
            rag = adapter,
        )
    }

    private fun createHippoRag(
        configPath: String?,
        overrides: Map<String, Any?>,
    ): UnifiedRagHandle {
        val effective = withPersistenceBridgeDefaults(RagId.HIPPO_RAG, overrides)
        val effectiveConfigPath = configPath ?: effective.string("configPath")
        val rag =
            HippoRAG(
                config = buildHippoBaseConfig(effectiveConfigPath, effective) ?: BaseConfig(),
                saveDir = effective.string("saveDir"),
                llmModelName = effective.string("llmModelName"),
                llmBaseUrl = effective.string("llmBaseUrl"),
                embeddingModelName = effective.string("embeddingModelName"),
                embeddingBaseUrl = effective.string("embeddingBaseUrl"),
            )
        val adapter = HippoRagUnifiedAdapter(rag)
        return UnifiedRagHandle(
            id = RagId.HIPPO_RAG,
            capabilities = HippoRagUnifiedAdapter.CAPABILITIES,
            rag = adapter,
        )
    }

    private fun createPathRag(
        configPath: String?,
        overrides: Map<String, Any?>,
    ): UnifiedRagHandle {
        val effective = withPersistenceBridgeDefaults(RagId.PATH_RAG, overrides)
        val persistenceSession = effective.persistenceSession("__persistenceSession")
        val useUnifiedSpi = effective.bool("__useUnifiedSpiForRetrievalAndIndex") ?: false
        val settings = configPath?.let { CommonRagConfigLoader.load(it).toPathRagConfig() }
        val hasStorageBridgeOverride =
            setOf(
                "kvStorage",
                "vectorStorage",
                "graphStorage",
                "neo4jUri",
                "neo4jUser",
                "neo4jPassword",
                "mongoUri",
                "mongoDatabase",
            ).any { key -> key in effective }
        val rag =
            if (!configPath.isNullOrBlank() && !hasStorageBridgeOverride && persistenceSession == null) {
                PathRAG.fromCommonConfig(
                    configPath = configPath,
                    workingDirOverride = effective.string("workingDir"),
                )
            } else {
                PathRAG(
                    workingDir =
                        effective.string("workingDir")
                            ?: settings?.workingDir
                            ?: defaultPathRagWorkingDir(),
                    kvStorage = effective.string("kvStorage") ?: settings?.kvStorage ?: "JsonKVStorage",
                    vectorStorage = effective.string("vectorStorage") ?: settings?.vectorStorage ?: "NanoVectorDBStorage",
                    graphStorage = effective.string("graphStorage") ?: settings?.graphStorage ?: "NetworkXStorage",
                    chunkTokenSize = effective.int("chunkTokenSize") ?: settings?.chunkTokenSize ?: 1200,
                    chunkOverlapTokenSize = effective.int("chunkOverlapTokenSize") ?: settings?.chunkOverlapTokenSize ?: 100,
                    language = effective.string("language") ?: settings?.language ?: "English",
                    runtimeSettings = settings?.toRuntimeSettingsMap() ?: emptyMap(),
                    extraConfig = buildPathExtraConfig(effective),
                    persistenceSession = persistenceSession,
                    useUnifiedSpiForRetrievalAndIndex = useUnifiedSpi,
                    persistenceNamespacePrefix = effective.string("persistenceNamespacePrefix") ?: "pathrag",
                )
            }
        val adapter = PathRagUnifiedAdapter(rag)
        return UnifiedRagHandle(
            id = RagId.PATH_RAG,
            capabilities = PathRagUnifiedAdapter.CAPABILITIES,
            rag = adapter,
        )
    }

    private fun createLightRag(
        configPath: String?,
        overrides: Map<String, Any?>,
    ): UnifiedRagHandle {
        val effective = withPersistenceBridgeDefaults(RagId.LIGHT_RAG, overrides)
        val persistenceSession = effective.persistenceSession("__persistenceSession")
        val useUnifiedSpi = effective.bool("__useUnifiedSpiForRetrievalAndIndex") ?: false
        val appTransform: (lightrag.di.AppConfig, lightrag.di.LightRagConfig) -> lightrag.di.AppConfig =
            { appConfig, _ ->
                val neo4j = buildLightRagNeo4jConfig(effective, appConfig.addonConfig.neo4j)
                val extras = appConfig.addonConfig.extras + buildLightRagAddonExtras(effective)
                appConfig.copy(
                    workingDir = effective.string("workingDir") ?: appConfig.workingDir,
                    graphStorageName = effective.string("graphStorageName") ?: appConfig.graphStorageName,
                    vectorStorageName = effective.string("vectorStorageName") ?: appConfig.vectorStorageName,
                    addonConfig = appConfig.addonConfig.copy(neo4j = neo4j, extras = extras),
                )
            }
        val runtime =
            if (configPath.isNullOrBlank()) {
                createLightRagRuntime(appConfigTransform = appTransform)
            } else {
                createLightRagRuntime(
                    configPath = configPath,
                    appConfigTransform = appTransform,
                )
            }
        val rag =
            LightRAG(
                ingestionService = runtime.ingestionService,
                queryService = runtime.queryService,
                storageManager = runtime.storageManager,
                embeddingModelForSpi = runtime.embeddingModel,
                persistenceSession = persistenceSession,
                useUnifiedSpiForRetrievalAndIndex = useUnifiedSpi,
                persistenceNamespacePrefix = effective.string("persistenceNamespacePrefix") ?: "lightrag",
            )
        val adapter = LightRagUnifiedAdapter(rag)
        return UnifiedRagHandle(
            id = RagId.LIGHT_RAG,
            capabilities = LightRagUnifiedAdapter.CAPABILITIES,
            rag = adapter,
        )
    }

    private fun createGraphRag(overrides: Map<String, Any?>): UnifiedRagHandle {
        val effective = withPersistenceBridgeDefaults(RagId.GRAPH_RAG, overrides)
        val rootDir = effective.string("rootDir")?.let { Path.of(it) }
        val chatModel = effective.string("chatModelName") ?: "gpt-4o-mini"
        val embeddingModel = effective.string("embeddingModelName") ?: "text-embedding-3-small"

        val rag =
            if (rootDir == null) {
                GraphRAG(
                    defaultChatModelName = chatModel,
                    defaultEmbeddingModelName = embeddingModel,
                )
            } else {
                val inputDir = effective.string("inputDir")?.let { Path.of(it) } ?: rootDir.resolve("input")
                val outputDir = effective.string("outputDir")?.let { Path.of(it) } ?: rootDir.resolve("output")
                val updateOutputDir =
                    effective.string("updateOutputDir")?.let { Path.of(it) }
                        ?: rootDir.resolve("update_output")
                GraphRAG(
                    rootDir = rootDir,
                    inputDir = inputDir,
                    outputDir = outputDir,
                    updateOutputDir = updateOutputDir,
                    defaultChatModelName = chatModel,
                    defaultEmbeddingModelName = embeddingModel,
                )
            }

        val adapter = GraphRagUnifiedAdapter(rag)
        return UnifiedRagHandle(
            id = RagId.GRAPH_RAG,
            capabilities = GraphRagUnifiedAdapter.CAPABILITIES,
            rag = adapter,
        )
    }

    private fun parseHippoSemanticMode(raw: String?): HippoRagSemanticMode =
        when (raw?.trim()?.lowercase()) {
            "dpr" -> HippoRagSemanticMode.DPR
            else -> HippoRagSemanticMode.GRAPH
        }

    private fun buildHippoBaseConfig(
        configPath: String?,
        overrides: Map<String, Any?>,
    ): BaseConfig? {
        val hasHippoOverrides =
            overrides.keys.any { key ->
                key in
                    setOf(
                        "saveDir",
                        "llmModelName",
                        "embeddingModelName",
                        "llmProvider",
                        "embeddingProvider",
                        "llmBaseUrl",
                        "embeddingBaseUrl",
                        "openAiApiKey",
                        "rerankDspyFilePath",
                        "retrievalTopK",
                        "qaTopK",
                    )
            }
        if (configPath.isNullOrBlank() && !hasHippoOverrides) return null

        val base =
            if (!configPath.isNullOrBlank()) {
                CommonRagConfigLoader.load(configPath).toHippoBaseConfig()
            } else {
                BaseConfig()
            }

        return base.apply {
            overrides.string("saveDir")?.let { saveDir = it }
            overrides.string("llmModelName")?.let { llmName = it }
            overrides.string("embeddingModelName")?.let { embeddingModelName = it }
            overrides.string("llmProvider")?.let { llmProvider = it }
            overrides.string("embeddingProvider")?.let { embeddingProvider = it }
            overrides.string("llmBaseUrl")?.let { llmBaseUrl = it }
            overrides.string("embeddingBaseUrl")?.let { embeddingBaseUrl = it }
            overrides.string("openAiApiKey")?.let { openAiApiKey = it }
            overrides.string("rerankDspyFilePath")?.let { rerankDspyFilePath = it }
            overrides.int("retrievalTopK")?.let { retrievalTopK = it.coerceAtLeast(1) }
            overrides.int("qaTopK")?.let { qaTopK = it.coerceAtLeast(1) }
        }
    }

    private fun withPersistenceBridgeDefaults(
        ragId: RagId,
        overrides: Map<String, Any?>,
    ): Map<String, Any?> {
        if (!(overrides.bool("useUnifiedPersistence") ?: false)) return overrides
        val backend = overrides.string("persistenceBackend")?.trim()?.lowercase() ?: return overrides
        val rootDir = overrides.string("persistenceRootDir")?.trim()?.takeIf { it.isNotEmpty() } ?: return overrides
        val config = overrides.anyMap("persistenceConfig")
        val bridged = overrides.toMutableMap()

        when (ragId) {
            RagId.PATH_RAG -> {
                when (backend) {
                    "neo4j" -> {
                        bridged.putIfAbsent("kvStorage", "Neo4jKVStorage")
                        bridged.putIfAbsent("vectorStorage", "Neo4jVectorStorage")
                        bridged.putIfAbsent("graphStorage", "Neo4jStorage")
                        bridged.putIfAbsent("neo4jUri", config.string("uri"))
                        bridged.putIfAbsent("neo4jUser", config.string("username"))
                        bridged.putIfAbsent("neo4jPassword", config.string("password"))
                    }

                    "mongodb", "mongo" -> {
                        bridged.putIfAbsent("kvStorage", "MongoKVStorage")
                        bridged.putIfAbsent("vectorStorage", "MongoVectorStorage")
                        bridged.putIfAbsent("graphStorage", "MongoGraphStorage")
                        bridged.putIfAbsent("mongoUri", config.string("connectionString") ?: config.string("uri"))
                        bridged.putIfAbsent("mongoDatabase", config.string("database"))
                    }
                }
                bridged.putIfAbsent("workingDir", Path.of(rootDir, "pathrag").toString())
            }

            RagId.LIGHT_RAG -> {
                when (backend) {
                    "neo4j" -> {
                        bridged.putIfAbsent("graphStorageName", "Neo4jGraphStorage")
                        bridged.putIfAbsent("vectorStorageName", "Neo4jVectorStorage")
                        bridged.putIfAbsent("neo4jUri", config.string("uri"))
                        bridged.putIfAbsent("neo4jUser", config.string("username"))
                        bridged.putIfAbsent("neo4jPassword", config.string("password"))
                        bridged.putIfAbsent("neo4jDatabase", config.string("database"))
                    }

                    "mongodb", "mongo" -> {
                        bridged.putIfAbsent("graphStorageName", "MongoGraphStorage")
                        bridged.putIfAbsent("mongoUri", config.string("connectionString") ?: config.string("uri"))
                        bridged.putIfAbsent("mongoDatabase", config.string("database"))
                    }
                }
                bridged.putIfAbsent("workingDir", Path.of(rootDir, "lightrag").toString())
            }

            RagId.GRAPH_RAG -> {
                bridged.putIfAbsent("rootDir", Path.of(rootDir, "graphrag").toString())
            }

            RagId.CAUSAL_RAG -> {
                bridged.putIfAbsent("graphPath", Path.of(rootDir, "causalrag", "graph.json").toString())
                bridged.putIfAbsent("indexPath", Path.of(rootDir, "causalrag", "index").toString())
            }

            RagId.HIPPO_RAG -> {
                bridged.putIfAbsent("saveDir", Path.of(rootDir, "hipporag").toString())
            }

            RagId.CAUSAL_HIPPO_RAG -> {
                bridged.putIfAbsent("saveDir", Path.of(rootDir, "causal_hipporag").toString())
            }
        }
        return bridged
    }

    private fun buildPathExtraConfig(overrides: Map<String, Any?>): ExtraConfig {
        val config = overrides.anyMap("persistenceConfig")
        return ExtraConfig(
            neo4jUri = overrides.string("neo4jUri") ?: config.string("uri"),
            neo4jUser = overrides.string("neo4jUser") ?: config.string("username"),
            neo4jPassword = overrides.string("neo4jPassword") ?: config.string("password"),
            mongoUri = overrides.string("mongoUri") ?: config.string("connectionString") ?: config.string("uri"),
            mongoDatabase = overrides.string("mongoDatabase") ?: config.string("database"),
            additional = overrides.anyMap("pathExtraConfig"),
        )
    }

    private fun buildLightRagNeo4jConfig(
        overrides: Map<String, Any?>,
        fallback: Neo4jConfig?,
    ): Neo4jConfig? {
        val config = overrides.anyMap("persistenceConfig")
        val uri = overrides.string("neo4jUri") ?: config.string("uri") ?: fallback?.uri
        val user = overrides.string("neo4jUser") ?: config.string("username") ?: fallback?.username
        val pass = overrides.string("neo4jPassword") ?: config.string("password") ?: fallback?.password
        val database = overrides.string("neo4jDatabase") ?: config.string("database") ?: fallback?.database
        return if (uri.isNullOrBlank() && user.isNullOrBlank() && pass.isNullOrBlank() && database.isNullOrBlank()) {
            fallback
        } else {
            Neo4jConfig(
                uri = uri,
                username = user,
                password = pass,
                database = database,
            )
        }
    }

    private fun buildLightRagAddonExtras(overrides: Map<String, Any?>): Map<String, Any> {
        val config = overrides.anyMap("persistenceConfig")
        val extras = mutableMapOf<String, Any>()
        val mongoUri = overrides.string("mongoUri") ?: config.string("connectionString") ?: config.string("uri")
        val mongoDatabase = overrides.string("mongoDatabase") ?: config.string("database")
        if (!mongoUri.isNullOrBlank()) extras["mongo_uri"] = mongoUri
        if (!mongoDatabase.isNullOrBlank()) extras["mongo_database"] = mongoDatabase
        return extras
    }

    private fun defaultPathRagWorkingDir(): String = "./PathRAG_cache_unified"

    private fun applyUnifiedPersistenceDefaults(
        configPath: String?,
        overrides: Map<String, Any?>,
    ): Map<String, Any?> {
        val defaults = loadUnifiedPersistenceDefaults(configPath)
        if (defaults.isEmpty()) return overrides
        val merged = defaults.toMutableMap()
        merged.putAll(overrides)

        val defaultConfig = defaults.anyMap("persistenceConfig")
        val overrideConfig = overrides.anyMap("persistenceConfig")
        if (defaultConfig.isNotEmpty() || overrideConfig.isNotEmpty()) {
            val mergedConfig = defaultConfig.toMutableMap().apply { putAll(overrideConfig) }
            val defaultMetadata = defaultConfig.anyMap("metadata")
            val overrideMetadata = overrideConfig.anyMap("metadata")
            if (defaultMetadata.isNotEmpty() || overrideMetadata.isNotEmpty()) {
                mergedConfig["metadata"] = defaultMetadata.toMutableMap().apply { putAll(overrideMetadata) }
            }
            merged["persistenceConfig"] = mergedConfig
        }
        return merged
    }

    private fun loadUnifiedPersistenceDefaults(configPath: String?): Map<String, Any?> {
        val defaultPath = Path.of("config", "common_rag.json")
        val candidates =
            buildList {
                configPath?.let { add(Path.of(it)) }
                add(defaultPath)
            }.map { it.toAbsolutePath().normalize() }
                .distinct()

        for (candidate in candidates) {
            if (!Files.exists(candidate)) continue
            val config = runCatching { CommonRagConfigLoader.load(candidate) }.getOrNull() ?: continue
            val defaults = config.unifiedPersistenceOverrides()
            if (defaults.isNotEmpty()) return defaults
        }
        return emptyMap()
    }

    private fun Map<String, Any?>.string(key: String): String? = this[key] as? String

    private fun Map<String, Any?>.anyMap(key: String): Map<String, Any?> {
        val map = this[key] as? Map<*, *> ?: return emptyMap()
        return map.entries.associate { (k, v) -> k.toString() to v }
    }

    private fun Map<String, Any?>.bool(key: String): Boolean? =
        when (val value = this[key]) {
            is Boolean -> value
            is String -> value.toBooleanStrictOrNull()
            else -> null
        }

    private fun Map<String, Any?>.int(key: String): Int? =
        when (val value = this[key]) {
            is Int -> value
            is Number -> value.toInt()
            is String -> value.toIntOrNull()
            else -> null
        }

    private fun Map<String, Any?>.double(key: String): Double? =
        when (val value = this[key]) {
            is Double -> value
            is Number -> value.toDouble()
            is String -> value.toDoubleOrNull()
            else -> null
        }

    private fun Map<String, Any?>.persistenceSession(key: String): PersistenceSession? = this[key] as? PersistenceSession
}
