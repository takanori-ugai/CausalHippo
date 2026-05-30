package com.youtu.graphrag.shared.config

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.PropertyNamingStrategies
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import io.github.oshai.kotlinlogging.KotlinLogging
import java.nio.file.Files
import java.nio.file.Path
import java.util.Locale

class ConfigManager(
    configPath: String? = null,
) {
    private val logger = KotlinLogging.logger {}
    private val jsonMapper: ObjectMapper = createMapper()
    private val yamlMapper: ObjectMapper = createMapper(YAMLFactory())

    val configPath: Path = Path.of(configPath ?: DEFAULT_CONFIG_PATH)
    private var rawConfig: MutableMap<String, Any?> = mutableMapOf()
    private var parsedConfig: AppConfig = AppConfig()

    init {
        loadConfig()
    }

    val datasets: Map<String, DatasetConfig>
        get() = parsedConfig.datasets

    val triggers: TriggersConfig
        get() = parsedConfig.triggers

    val construction: ConstructionConfig
        get() = parsedConfig.construction

    val treeComm: TreeCommConfig
        get() = parsedConfig.construction.treeComm

    val retrieval: RetrievalConfig
        get() = parsedConfig.retrieval

    val embeddings: EmbeddingsConfig
        get() = parsedConfig.embeddings

    val nlp: NlpConfig
        get() = parsedConfig.nlp

    val prompts: Map<String, Map<String, String>>
        get() = parsedConfig.prompts

    val output: OutputConfig
        get() = parsedConfig.output

    val performance: PerformanceConfig
        get() = parsedConfig.performance

    val evaluation: EvaluationConfig
        get() = parsedConfig.evaluation

    fun loadConfig() {
        require(Files.exists(configPath)) { "Configuration file not found: $configPath" }
        val normalizedPath = configPath.toString().lowercase(Locale.ROOT)
        require(
            normalizedPath.endsWith(".json") || normalizedPath.endsWith(".yaml") || normalizedPath.endsWith(".yml"),
        ) {
            "Only JSON or YAML configuration files are supported: $configPath"
        }
        val mapper = if (normalizedPath.endsWith(".yaml") || normalizedPath.endsWith(".yml")) yamlMapper else jsonMapper

        rawConfig =
            Files.newInputStream(configPath).use { stream ->
                mapper.readValue(stream, object : TypeReference<MutableMap<String, Any?>>() {})
            }

        parseConfig()
        validateConfig()

        logger.info { "Configuration loaded successfully from $configPath" }
    }

    fun getDatasetConfig(datasetName: String): DatasetConfig =
        requireNotNull(datasets[datasetName]) {
            "Dataset '$datasetName' not found in configuration"
        }

    fun getPrompt(
        category: String,
        promptType: String,
    ): String =
        requireNotNull(prompts[category]?.get(promptType)) {
            "Prompt not found: $category.$promptType"
        }

    fun getPromptFormatted(
        category: String,
        promptType: String,
        variables: Map<String, Any?>,
    ): String {
        val template = getPrompt(category, promptType)
        val rendered =
            PLACEHOLDER_REGEX.replace(template) { matchResult ->
                val key = matchResult.groupValues[1]
                if (variables.containsKey(key)) {
                    variables[key]?.toString() ?: ""
                } else {
                    matchResult.value
                }
            }

        val missing = PLACEHOLDER_REGEX.find(rendered)?.groupValues?.get(1)
        require(missing == null) {
            "Missing variable '$missing' for prompt $category.$promptType"
        }

        return rendered
    }

    fun overrideConfig(overrides: Map<String, Any?>) {
        deepMerge(rawConfig, overrides)
        parseConfig()
        validateConfig()
    }

    fun saveConfig(outputPath: String) {
        val path = Path.of(outputPath)
        val normalizedPath = path.toString().lowercase(Locale.ROOT)
        require(
            normalizedPath.endsWith(".json") || normalizedPath.endsWith(".yaml") || normalizedPath.endsWith(".yml"),
        ) {
            "Only JSON or YAML configuration files are supported: $path"
        }
        Files.createDirectories(path.parent ?: Path.of("."))
        val mapper = if (normalizedPath.endsWith(".yaml") || normalizedPath.endsWith(".yml")) yamlMapper else jsonMapper

        Files.newOutputStream(path).use { stream ->
            mapper.writerWithDefaultPrettyPrinter().writeValue(stream, rawConfig)
        }
    }

    fun toMap(): Map<String, Any?> =
        jsonMapper.readValue(
            jsonMapper.writeValueAsBytes(rawConfig),
            object : TypeReference<Map<String, Any?>>() {},
        )

    fun createOutputDirectories() {
        val directories = listOf(output.baseDir, output.graphsDir, output.chunksDir, output.logsDir)
        directories.forEach { directory -> Files.createDirectories(Path.of(directory)) }
    }

    private fun parseConfig() {
        parsedConfig = jsonMapper.convertValue(rawConfig, AppConfig::class.java)
    }

    private fun validateConfig() {
        datasets.forEach { (datasetName, datasetConfig) ->
            if (datasetConfig.corpusPath.isBlank() || !Files.exists(Path.of(datasetConfig.corpusPath))) {
                logger.warn { "Corpus path not found for $datasetName: ${datasetConfig.corpusPath}" }
            }
            if (datasetConfig.schemaPath.isBlank() || !Files.exists(Path.of(datasetConfig.schemaPath))) {
                logger.warn { "Schema path not found for $datasetName: ${datasetConfig.schemaPath}" }
            }
        }

        val validModes = setOf("agent", "noagent")
        require(triggers.mode in validModes) {
            "Invalid mode: ${triggers.mode}. Must be one of $validModes"
        }
        require(construction.mode in validModes) {
            "Invalid construction mode: ${construction.mode}"
        }
        require(retrieval.topK > 0) { "top_k must be positive" }
        require(retrieval.strategy.timeoutMs > 0) {
            "retrieval.strategy.timeout_ms must be positive"
        }
        require(retrieval.strategy.maxConcurrency > 0) {
            "retrieval.strategy.max_concurrency must be positive"
        }
        val validRetrievalStrategies =
            setOf(
                "lexical_triple",
                "semantic_triple",
                "path_expand",
                "community_triple",
                "lexical_chunk",
                "semantic_chunk",
                "triple_chunk_bridge",
            )
        val enabledStrategies =
            retrieval.strategy.enabled.map { strategy ->
                strategy.lowercase(Locale.ROOT)
            }
        require(enabledStrategies.isNotEmpty()) {
            "retrieval.strategy.enabled must not be empty"
        }
        require(enabledStrategies.all { strategy -> strategy in validRetrievalStrategies }) {
            "retrieval.strategy.enabled contains unknown strategy"
        }
        retrieval.strategy.weights.forEach { (strategy, weight) ->
            require(strategy.lowercase(Locale.ROOT) in validRetrievalStrategies) {
                "retrieval.strategy.weights[$strategy] is not a supported strategy"
            }
            require(weight >= 0.0) {
                "retrieval.strategy.weights[$strategy] must be non-negative"
            }
        }
        require(treeComm.structWeight in 0.0..1.0) {
            "struct_weight must be between 0 and 1"
        }
        require(treeComm.mergeThreshold in 0.0..1.0) {
            "merge_threshold must be between 0 and 1"
        }
        require(treeComm.maxIterations > 0) {
            "max_iterations must be positive"
        }
        require(treeComm.maxTotalCommunities > 0) {
            "max_total_communities must be positive"
        }
        require(treeComm.summaryMaxWords > 0) {
            "summary_max_words must be positive"
        }
    }

    private fun deepMerge(
        base: MutableMap<String, Any?>,
        overrides: Map<String, Any?>,
    ) {
        overrides.forEach { (key, overrideValue) ->
            val baseValue = base[key]
            if (baseValue is Map<*, *> && overrideValue is Map<*, *>) {
                val mergedChild = toStringAnyMap(baseValue).toMutableMap()
                deepMerge(mergedChild, toStringAnyMap(overrideValue))
                base[key] = mergedChild
            } else {
                base[key] = overrideValue
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun toStringAnyMap(map: Map<*, *>): Map<String, Any?> = map as Map<String, Any?>

    companion object {
        private const val DEFAULT_CONFIG_PATH = "config/base_config.json"
        private val PLACEHOLDER_REGEX = Regex("\\{([^{}]+)}")

        private fun createMapper(factory: com.fasterxml.jackson.core.JsonFactory? = null): ObjectMapper =
            (if (factory == null) ObjectMapper() else ObjectMapper(factory))
                .registerKotlinModule()
                .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
                .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
    }
}
