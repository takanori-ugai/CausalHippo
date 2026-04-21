package hipporag.demo

import kotlin.system.exitProcess

/**
 * Parsed CLI arguments for demo entry points.
 *
 * @property docsPath path to a JSON array of documents.
 * @property queriesPath path to a JSON array of queries.
 * @property saveDir output directory for artifacts.
 * @property llmName LLM model name.
 * @property embeddingName embedding model name.
 * @property llmBaseUrl optional base URL for LLM provider.
 * @property embeddingBaseUrl optional base URL for embedding provider.
 */
data class DemoArgs(
    val docsPath: String,
    val queriesPath: String,
    val saveDir: String,
    val llmName: String,
    val embeddingName: String,
    val llmBaseUrl: String?,
    val embeddingBaseUrl: String?,
) {
    companion object {
        /**
         * Parses CLI arguments into a [DemoArgs] instance.
         */
        fun parse(args: Array<String>): DemoArgs {
            val map = mutableMapOf<String, String>()
            var i = 0
            while (i < args.size) {
                val key = args[i]
                if (!key.startsWith("--") || i + 1 >= args.size) {
                    usageAndExit()
                }
                map[key.removePrefix("--")] = args[i + 1]
                i += 2
            }

            val docs = map["docs"] ?: usageAndExit()
            val queries = map["queries"] ?: usageAndExit()

            return DemoArgs(
                docsPath = docs,
                queriesPath = queries,
                saveDir = map["save_dir"] ?: "outputs",
                llmName = map["llm_name"] ?: "gpt-4o-mini",
                embeddingName = map["embedding_name"] ?: "text-embedding-3-large",
                llmBaseUrl = map["llm_base_url"],
                embeddingBaseUrl = map["embedding_base_url"],
            )
        }

        private fun usageAndExit(): Nothing {
            System.err.println(
                "Usage: --docs <docs.json> --queries <queries.json> [--save_dir outputs] " +
                    "[--llm_name gpt-4o-mini] [--llm_base_url <url>] [--embedding_name <name>] " +
                    "[--embedding_base_url <url>]",
            )
            exitProcess(2)
        }
    }
}
