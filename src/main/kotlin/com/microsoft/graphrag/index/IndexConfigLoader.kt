package com.microsoft.graphrag.index

import com.fasterxml.jackson.annotation.JsonProperty
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import shared.config.CommonRagConfig
import shared.config.CommonRagConfigLoader
import tools.jackson.databind.DeserializationFeature
import tools.jackson.databind.ObjectMapper
import tools.jackson.databind.json.JsonMapper
import tools.jackson.module.kotlin.KotlinModule
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path

/**
 * Resolved index configuration for the application.
 *
 * @property graphConfig GraphRag settings derived from the resolved paths.
 */
data class IndexConfig(
    val graphConfig: GraphRagConfig,
)

/**
 * Loads index configuration from a JSON file under the provided root.
 */
object IndexConfigLoader {
    private const val DEFAULT_COMMON_CONFIG = "config/common_rag.json"
    private val logger = KotlinLogging.logger {}

    private val mapper: ObjectMapper =
        JsonMapper
            .builder()
            .addModule(KotlinModule.Builder().build())
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
            .build()

    /**
     * Loads configuration from `config/common_rag.json` (or an explicit config path) and resolves directories.
     *
     * @param root Project root used to resolve relative paths.
     * @param configPath Optional explicit config file path.
     * @param overrideOutputDir Optional explicit output directory override.
     * @return IndexConfig containing a GraphRagConfig with resolved `rootDir`, `inputDir`, `outputDir`, and `updateOutputDir`.
     */
    fun load(
        root: Path,
        configPath: Path?,
        overrideOutputDir: Path? = null,
    ): IndexConfig {
        val resolvedRoot = root.toAbsolutePath().normalize()
        val settingsPath = (configPath ?: defaultConfigPath(resolvedRoot)).toAbsolutePath().normalize()
        val raw = loadRawConfig(settingsPath, configPath != null)

        val configRoot = raw.rootDir?.let { resolvedRoot.resolve(it).normalize() } ?: resolvedRoot
        val inputDir =
            resolveBaseDir(
                configRoot,
                raw.input?.storage?.baseDir ?: raw.input?.baseDir,
                "input",
            )

        val outputDir =
            overrideOutputDir?.toAbsolutePath()?.normalize()
                ?: resolveBaseDir(
                    configRoot,
                    raw.output?.baseDir,
                    "output",
                )

        val updateOutputDir = outputDir.resolve("update_output")

        return IndexConfig(
            graphConfig =
                GraphRagConfig(
                    rootDir = configRoot,
                    inputDir = inputDir,
                    outputDir = outputDir,
                    updateOutputDir = updateOutputDir,
                ),
        )
    }

    private fun defaultConfigPath(root: Path): Path = root.resolve(DEFAULT_COMMON_CONFIG)

    private fun loadRawConfig(
        settingsPath: Path,
        explicitConfig: Boolean,
    ): RawIndexConfig {
        if (!Files.exists(settingsPath)) {
            if (explicitConfig) {
                logger.warn { "Config file not found: $settingsPath; using defaults." }
            }
            return RawIndexConfig()
        }

        val text =
            runCatching { Files.readString(settingsPath) }.getOrElse { error ->
                logger.warn { "Failed to read $settingsPath ($error); using defaults." }
                return RawIndexConfig()
            }

        val commonFallback = CommonRagConfigLoader.parseOrNull(text)?.toRawIndexConfig()
        val parsed =
            runCatching { mapper.readValue(text, RawIndexConfig::class.java) }.getOrElse { error ->
                if (error is IOException) {
                    logger.warn { "Failed to parse $settingsPath as JSON ($error); using defaults." }
                } else {
                    logger.warn { "Failed to parse $settingsPath ($error); using defaults." }
                }
                RawIndexConfig()
            }
        return parsed.withFallback(commonFallback)
    }

    private fun CommonRagConfig.toRawIndexConfig(): RawIndexConfig {
        val rootDir = jsonString(shared, "rootDir", "root_dir")
        val inputDir = jsonString(shared, "inputDir", "input_dir")
        val outputDir = jsonString(shared, "outputDir", "output_dir")
        return RawIndexConfig(
            rootDir = rootDir,
            input = inputDir?.let { RawInput(baseDir = it) },
            output = outputDir?.let { RawOutput(baseDir = it) },
        )
    }

    private fun RawIndexConfig.withFallback(fallback: RawIndexConfig?): RawIndexConfig {
        if (fallback == null) return this
        return RawIndexConfig(
            rootDir = this.rootDir ?: fallback.rootDir,
            input = (this.input ?: fallback.input)?.withFallback(fallback.input),
            output = (this.output ?: fallback.output)?.withFallback(fallback.output),
        )
    }

    private fun RawInput.withFallback(fallback: RawInput?): RawInput =
        RawInput(
            baseDir = this.baseDir ?: fallback?.baseDir,
            storage = (this.storage ?: fallback?.storage)?.withFallback(fallback?.storage),
        )

    private fun RawOutput.withFallback(fallback: RawOutput?): RawOutput =
        RawOutput(
            baseDir = this.baseDir ?: fallback?.baseDir,
        )

    private fun RawStorage.withFallback(fallback: RawStorage?): RawStorage =
        RawStorage(
            baseDir = this.baseDir ?: fallback?.baseDir,
        )

    private fun jsonString(
        source: JsonObject,
        vararg keys: String,
    ): String? {
        for (key in keys) {
            val value = source[key] as? JsonPrimitive ?: continue
            val str = value.contentOrNull?.trim()
            if (!str.isNullOrBlank()) {
                return str
            }
        }
        return null
    }

    private fun resolveBaseDir(
        root: Path,
        configured: String?,
        fallback: String,
    ): Path {
        val value = configured?.trim().orEmpty()
        val path =
            if (value.isBlank()) {
                root.resolve(fallback)
            } else {
                val candidate = Path.of(value)
                if (candidate.isAbsolute) candidate else root.resolve(candidate)
            }
        return path.normalize()
    }
}

/**
 * Raw JSON-backed configuration container.
 *
 * @property rootDir Optional `root_dir` value; if absent, defaults to the provided root.
 * @property input Optional `input` configuration section.
 * @property output Optional `output` configuration section.
 */
data class RawIndexConfig(
    @param:JsonProperty("root_dir") val rootDir: String? = null,
    @param:JsonProperty("input") val input: RawInput? = null,
    @param:JsonProperty("output") val output: RawOutput? = null,
)

/**
 * Raw input configuration section.
 *
 * @property baseDir Optional `base_dir` override for input resolution.
 * @property storage Optional nested `storage` section.
 */
data class RawInput(
    @param:JsonProperty("base_dir") val baseDir: String? = null,
    @param:JsonProperty("storage") val storage: RawStorage? = null,
)

/**
 * Raw output configuration section.
 *
 * @property baseDir Optional `base_dir` override for output resolution.
 */
data class RawOutput(
    @param:JsonProperty("base_dir") val baseDir: String? = null,
)

/**
 * Raw storage configuration section.
 *
 * @property baseDir Optional `base_dir` override for storage input resolution.
 */
data class RawStorage(
    @param:JsonProperty("base_dir") val baseDir: String? = null,
)
