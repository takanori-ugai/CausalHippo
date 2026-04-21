package hipporag.demo

import hipporag.HippoRag
import hipporag.config.BaseConfig
import hipporag.utils.jsonWithDefaults
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import java.io.File

/**
 * Runs a demo workflow that loads docs/queries from JSON and executes HippoRAG.
 *
 * @param args CLI arguments parsed by [DemoArgs.parse].
 * @param configure callback to customize [BaseConfig] based on parsed args.
 */
fun runDemo(
    args: Array<String>,
    configure: (BaseConfig, DemoArgs) -> Unit,
) {
    val parsed = DemoArgs.parse(args)
    val json = jsonWithDefaults { ignoreUnknownKeys = true }
    val docsFile =
        File(parsed.docsPath).also {
            require(it.exists()) { "Docs file not found: ${parsed.docsPath}" }
        }
    val queriesFile =
        File(parsed.queriesPath).also {
            require(it.exists()) { "Queries file not found: ${parsed.queriesPath}" }
        }
    val docs = json.decodeFromString(ListSerializer(String.serializer()), docsFile.readText())
    val queries = json.decodeFromString(ListSerializer(String.serializer()), queriesFile.readText())

    val config =
        BaseConfig().apply {
            saveDir = parsed.saveDir
            llmName = parsed.llmName
            embeddingModelName = parsed.embeddingName
            openieMode = "online"
            configure(this, parsed)
        }

    val hipporag = HippoRag(config = config)
    hipporag.index(docs)
    hipporag.ragQa(queries = queries, goldDocs = null, goldAnswers = null)
}
