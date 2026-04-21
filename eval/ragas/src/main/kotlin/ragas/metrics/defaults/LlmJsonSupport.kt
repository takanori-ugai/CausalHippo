package ragas.metrics.defaults

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.intOrNull

internal object LlmJsonSupport {
    private val json = Json { ignoreUnknownKeys = true }

    /**
     * Extracts and parses the first valid JSON object from raw LLM output text.
     *
     * @param raw Raw model output text.
     */
    fun parseFirstJsonObject(raw: String): JsonObject? {
        var searchFrom = 0
        while (searchFrom < raw.length) {
            val start = raw.indexOf('{', searchFrom)
            if (start < 0) {
                return null
            }
            val extracted = extractJsonObjectAt(raw, start)
            if (extracted != null) {
                val parsed = runCatching { json.parseToJsonElement(extracted.first) as? JsonObject }.getOrNull()
                if (parsed != null) {
                    return parsed
                }
                searchFrom = extracted.second + 1
            } else {
                searchFrom = start + 1
            }
        }
        return null
    }

    /**
     * Extracts a string array from a JSON object field, filtering blank and null-like values.
     *
     * @param root Parsed JSON root object.
     * @param key JSON/object key.
     */
    fun readStringArray(
        root: JsonObject,
        key: String,
    ): List<String> =
        (root[key] as? JsonArray)
            .orEmpty()
            .mapNotNull { element ->
                val primitive = element as? JsonPrimitive ?: return@mapNotNull null
                if (primitive.isString) {
                    primitive.content.trim()
                } else {
                    primitive.content.trim().takeIf { value -> value.isNotBlank() && value != "null" }
                }
            }.filter { value -> value.isNotBlank() }

    /**
     * Reads an integer-like value from a JSON field.
     *
     * Supports numeric primitives directly, boolean values (`true` -> 1, `false` -> 0),
     * and numeric strings.
     *
     * @param root Parsed JSON root object.
     * @param key JSON/object key.
     */
    fun readIntLike(
        root: JsonObject,
        key: String,
    ): Int? {
        val primitive = root[key] as? JsonPrimitive ?: return null
        primitive.intOrNull?.let { return it }
        primitive.booleanOrNull?.let { return if (it) 1 else 0 }
        return primitive.content.trim().toIntOrNull()
    }

    private fun extractJsonObjectAt(
        text: String,
        start: Int,
    ): Pair<String, Int>? {
        var depth = 0
        var inString = false
        var escaped = false
        for (index in start until text.length) {
            val ch = text[index]
            if (inString) {
                if (escaped) {
                    escaped = false
                } else if (ch == '\\') {
                    escaped = true
                } else if (ch == '"') {
                    inString = false
                }
                continue
            }
            if (ch == '"') {
                inString = true
                continue
            }
            if (ch == '{') {
                depth += 1
            }
            if (ch == '}') {
                depth -= 1
                if (depth == 0) {
                    return text.substring(start, index + 1) to index
                }
            }
        }
        return null
    }
}
