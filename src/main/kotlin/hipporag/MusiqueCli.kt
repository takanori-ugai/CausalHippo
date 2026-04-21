package hipporag

import hipporag.config.BaseConfig
import hipporag.utils.jsonWithDefaults
import hipporag.utils.loadConfigFromJson
import hipporag.utils.stringToBool
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.io.File
import kotlin.system.exitProcess

internal fun readMusiqueSamples(
    path: String,
    limit: Int?,
): List<MusiqueSample> {
    val json = jsonWithDefaults { ignoreUnknownKeys = true }
    val file = resolveMusiqueFile(path, "input")
    val lines = file.readLines().filter { it.isNotBlank() }
    val capped = if (limit != null) lines.take(limit) else lines
    return capped.map { line -> json.decodeFromString(MusiqueSample.serializer(), line) }
}

internal fun buildMusiqueConfig(
    parsed: MusiqueArgs,
    sampleId: String,
): BaseConfig {
    val config =
        parsed.configPath
            ?.let { loadConfigFromJson(resolveMusiqueFile(it, "config").path) }
            ?: BaseConfig()
    parsed.saveDir?.let { base ->
        val sanitized = sampleId.replace(Regex("[^A-Za-z0-9._-]"), "_")
        config.saveDir = File(base, sanitized).path
    }
    parsed.llmName?.let { config.llmName = it }
    parsed.llmBaseUrl?.let { config.llmBaseUrl = it }
    parsed.embeddingName?.let { config.embeddingModelName = it }
    parsed.openieMode?.let { config.openieMode = it }
    parsed.forceIndexFromScratch?.let { config.forceIndexFromScratch = stringToBool(it) }
    parsed.forceOpenieFromScratch?.let { config.forceOpenieFromScratch = stringToBool(it) }
    parsed.rerankDspyFilePath?.let { config.rerankDspyFilePath = it }
    return config
}

internal fun resolveMusiqueFile(
    path: String,
    label: String,
): File {
    val baseDir = File(".").canonicalFile
    val file = File(path).canonicalFile
    val basePath = baseDir.path + File.separator
    require(file.path.startsWith(basePath)) {
        "Refusing to read $label outside working directory: ${file.path}"
    }
    require(file.isFile) {
        "Missing or invalid $label file: ${file.path}"
    }
    return file
}

@Serializable
internal data class MusiqueParagraph(
    val idx: Int? = null,
    val title: String? = null,
    @SerialName("paragraph_text")
    val paragraphText: String,
    @SerialName("is_supporting")
    val isSupporting: Boolean? = null,
)

@Serializable
internal data class MusiqueSample(
    val id: String,
    val paragraphs: List<MusiqueParagraph>,
    val question: String,
    val answer: String,
    @SerialName("answer_aliases")
    val answerAliases: List<String> = emptyList(),
    val answerable: Boolean? = null,
)

internal data class MusiqueArgs(
    val inputPath: String,
    val limit: Int?,
    val parallelism: Int?,
    val configPath: String?,
    val saveDir: String?,
    val llmBaseUrl: String?,
    val llmName: String?,
    val embeddingName: String?,
    val openieMode: String?,
    val forceIndexFromScratch: String?,
    val forceOpenieFromScratch: String?,
    val rerankDspyFilePath: String?,
) {
    companion object {
        fun parse(args: Array<String>): MusiqueArgs {
            val map = mutableMapOf<String, String>()
            var i = 0
            while (i < args.size) {
                val key = args[i]
                if (!key.startsWith("--") || i + 1 >= args.size) {
                    val message = if (key.startsWith("--")) "Missing value for $key." else "Unexpected argument: $key."
                    usageAndExit(message)
                }
                map[key.removePrefix("--")] = args[i + 1]
                i += 2
            }

            val input = map["input"] ?: usageAndExit("Missing required --input.")
            val limit = map["limit"]?.toIntOrNull()
            val parallelism = map["parallelism"]?.toIntOrNull()

            return MusiqueArgs(
                inputPath = input,
                limit = limit,
                parallelism = parallelism,
                configPath = map["config"],
                saveDir = map["save_dir"],
                llmBaseUrl = map["llm_base_url"],
                llmName = map["llm_name"],
                embeddingName = map["embedding_name"],
                openieMode = map["openie_mode"],
                forceIndexFromScratch = map["force_index_from_scratch"],
                forceOpenieFromScratch = map["force_openie_from_scratch"],
                rerankDspyFilePath = map["rerank_dspy_file_path"],
            )
        }

        private fun usageAndExit(reason: String? = null): Nothing {
            if (reason != null) {
                System.err.println(reason)
            }
            System.err.println(
                "Usage: --input <musique.jsonl> [--limit <n>] [--parallelism <n>] [--config <config.json>] " +
                    "[--save_dir outputs] [--llm_name gpt-4o-mini] [--llm_base_url <url>] " +
                    "[--embedding_name <name>] [--openie_mode online] [--force_index_from_scratch false] " +
                    "[--force_openie_from_scratch false] [--rerank_dspy_file_path <path>]\n" +
                    "Note: file paths must resolve within the current working directory.",
            )
            exitProcess(2)
        }
    }
}
