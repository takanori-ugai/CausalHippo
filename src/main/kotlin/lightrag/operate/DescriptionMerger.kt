package lightrag.operate

import dev.langchain4j.model.chat.ChatModel
import kotlinx.serialization.json.Json
import lightrag.core.Constants
import lightrag.utils.Prompts
import lightrag.utils.computeMd5
import shared.chunking.countTokensWithJTokKit
import shared.chunking.truncateTextByTokenSize

internal class DescriptionMerger(
    config: Map<String, Any?>,
) {
    private val model = config["llm_model_func"] as? ChatModel
    private val language = config["language"] as? String ?: "English"
    private val forceAt = (config["force_llm_summary_on_merge"] as? Number)?.toInt() ?: 8
    private val maxTokens = (config["summary_max_tokens"] as? Number)?.toInt() ?: 1200
    private val contextSize = ((config["summary_context_size"] as? Number)?.toInt() ?: 12000).coerceAtLeast(1)
    private val recommendedLength = (config["summary_length_recommended"] as? Number)?.toInt() ?: 600

    fun merge(
        type: String,
        name: String,
        existing: Map<String, String>?,
        incoming: List<String>,
    ): Map<String, String> {
        val oldHashes =
            existing
                ?.get("description_hashes")
                .orEmpty()
                .split(',')
                .filter(String::isNotEmpty)
                .toSet()
        val previous = existing?.get("description").orEmpty().split(Constants.GRAPH_FIELD_SEP)
        val fresh = if (oldHashes.isEmpty()) incoming else incoming.filterNot { computeMd5(it) in oldHashes }
        val fragments = (previous + fresh).filter(String::isNotBlank).distinct()
        var parts = fragments
        var summarized = false
        while (model != null && parts.size > 1) {
            val total = parts.sumOf { countTokensWithJTokKit(it) }
            if (parts.size < forceAt && total < maxTokens && total <= contextSize) break
            summarized = true
            val sizes = parts.map { countTokensWithJTokKit(jsonRecord(it) + "\n") }
            if (sizes.sum() <= contextSize || parts.size == 2) {
                parts = listOf(summarize(model, type, name, parts))
                break
            }

            // Pair at least two inputs so each pass reduces the number of descriptions.
            val reduced = mutableListOf<String>()
            var start = 0
            while (start < parts.size) {
                var end = start
                var tokens = 0
                while (end < parts.size && (end - start < 2 || tokens + sizes[end] <= contextSize)) {
                    tokens += sizes[end]
                    end++
                }
                val group = parts.subList(start, end)
                reduced += if (group.size == 1) group.single() else summarize(model, type, name, group)
                start = end
            }
            parts = reduced
        }
        return buildMap {
            put("description", parts.joinToString(Constants.GRAPH_FIELD_SEP))
            // Keep fingerprints when compression removes the original strings used for deduplication.
            if (summarized || oldHashes.isNotEmpty()) {
                val hashes = if (summarized) oldHashes + fragments.map(::computeMd5) else oldHashes
                put("description_hashes", hashes.joinToString(","))
            }
        }
    }

    private fun summarize(
        model: ChatModel,
        type: String,
        name: String,
        parts: List<String>,
    ): String {
        fun jsonLines(limit: Int?): String =
            parts.joinToString("\n") {
                val text = if (limit == null) it else truncateTextByTokenSize(it, limit)
                jsonRecord(text)
            }
        var bounded = jsonLines(null)
        if (countTokensWithJTokKit(bounded) > contextSize) {
            // Keep valid JSON and some evidence from every input, including oversized pairs.
            var low = 1
            var high = contextSize
            bounded = jsonLines(low)
            require(countTokensWithJTokKit(bounded) <= contextSize) { "summary_context_size is too small for the description records" }
            while (low <= high) {
                val limit = low + (high - low) / 2
                val candidate = jsonLines(limit)
                if (countTokensWithJTokKit(candidate) <= contextSize) {
                    bounded = candidate
                    low = limit + 1
                } else {
                    high = limit - 1
                }
            }
        }
        val fields =
            mapOf(
                "language" to language,
                "summary_length" to recommendedLength.toString(),
                "description_name" to name,
                "description_type" to type,
                "description_list" to bounded,
            )
        val prompt = placeholder.replace(Prompts.SUMMARIZE_ENTITY_DESCRIPTIONS) { fields.getValue(it.groupValues[1]) }
        return model.chat(prompt).trim().also { check(it.isNotEmpty()) { "Empty LightRAG summary for $name" } }
    }

    private fun jsonRecord(text: String): String = Json.encodeToString(mapOf("Description" to text))

    private companion object {
        val placeholder = Regex("""\{(\w+)}""")
    }
}
