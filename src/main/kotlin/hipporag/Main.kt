package hipporag

import hipporag.config.BaseConfig
import hipporag.utils.jsonWithDefaults
import hipporag.utils.loadConfigFromJson
import hipporag.utils.stringToBool
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import java.io.File
import kotlin.system.exitProcess

/**
 * CLI entry point for running indexing + QA over JSON input files.
 *
 * @param args CLI arguments parsed by [Args.parse].
 */
fun main(args: Array<String>) {
    val parsed = Args.parse(args)

    val json = jsonWithDefaults { ignoreUnknownKeys = true }
    val docs = readStringListJson(json, parsed.docsPath, "docs")
    val queries = readStringListJson(json, parsed.queriesPath, "queries")

    val config =
        parsed.configPath
            ?.let { loadConfigFromJson(resolveSafeFile(it, "config").path) }
            ?: BaseConfig()
    parsed.saveDir?.let { config.saveDir = it }
    parsed.llmName?.let { config.llmName = it }
    parsed.llmBaseUrl?.let { config.llmBaseUrl = it }
    parsed.embeddingName?.let { config.embeddingModelName = it }
    parsed.openieMode?.let { config.openieMode = it }
    parsed.forceIndexFromScratch?.let { config.forceIndexFromScratch = stringToBool(it) }
    parsed.forceOpenieFromScratch?.let { config.forceOpenieFromScratch = stringToBool(it) }
    parsed.rerankDspyFilePath?.let { config.rerankDspyFilePath = it }

    val hipporag = HippoRag(config = config)
    hipporag.index(docs)
    val result = hipporag.ragQa(queries = queries, goldDocs = null, goldAnswers = null)
    printAnswers(result)
}

private fun readStringListJson(
    json: kotlinx.serialization.json.Json,
    path: String,
    label: String,
): List<String> =
    json.decodeFromString(
        ListSerializer(String.serializer()),
        resolveSafeFile(path, label).readText(),
    )

private fun resolveSafeFile(
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

private fun printAnswers(result: hipporag.utils.RagQaResult) {
    println("=== Answers ===")
    result.solutions.forEachIndexed { idx, solution ->
        val answer = solution.answer ?: "(no answer)"
        println("${idx + 1}. Q: ${solution.question}")
        println("   A: $answer")
    }
}

private data class Args(
    val docsPath: String,
    val queriesPath: String,
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
        fun parse(args: Array<String>): Args {
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

            val docs = map["docs"] ?: usageAndExit("Missing required --docs.")
            val queries = map["queries"] ?: usageAndExit("Missing required --queries.")

            return Args(
                docsPath = docs,
                queriesPath = queries,
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
                "Usage: --docs <docs.json> --queries <queries.json> [--config <config.json>] [--save_dir outputs] " +
                    "[--llm_name gpt-4o-mini] [--llm_base_url <url>] [--embedding_name <name>] " +
                    "[--openie_mode online] [--force_index_from_scratch false] [--force_openie_from_scratch false] " +
                    "[--rerank_dspy_file_path <path>]\n" +
                    "Note: file paths must resolve within the current working directory.",
            )
            exitProcess(2)
        }
    }
}
