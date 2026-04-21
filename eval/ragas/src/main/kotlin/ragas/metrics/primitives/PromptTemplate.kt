package ragas.metrics.primitives

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject

/**
 * Input/output pair used as a few-shot example in prompt rendering.
 *
 * @property input Example input payload.
 * @property output Expected structured output payload.
 */
data class PromptIoExample(
    val input: Map<String, Any?>,
    val output: Map<String, Any?>,
)

/**
 * Renders instruction templates with optional schema hints and few-shot examples.
 *
 * @property instructionTemplate Base template string with `{key}` placeholders.
 * @property outputSchema Optional JSON schema to guide output format.
 * @property examples Few-shot examples to include in the prompt.
 * @property includeInputOutputFrame Whether to append input/output framing instructions.
 */
class PromptTemplate(
    private val instructionTemplate: String,
    private val outputSchema: JsonElement? = null,
    private val examples: List<PromptIoExample> = emptyList(),
    private val includeInputOutputFrame: Boolean = true,
) {
    /**
     * Builds a prompt string from template [inputs].
     *
     * @param inputs Values used to replace `{key}` placeholders in the instruction template.
     * @return Rendered prompt string that may include schema guidance and examples.
     */
    fun render(inputs: Map<String, Any?>): String {
        var rendered = instructionTemplate
        inputs.forEach { (key, value) ->
            rendered = rendered.replace("{$key}", value?.toString() ?: "")
        }

        if (!includeInputOutputFrame && outputSchema == null && examples.isEmpty()) {
            return rendered
        }

        val outputSignature =
            outputSchema
                ?.let { schema ->
                    "Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:\n" +
                        compactJson.encodeToString(JsonElement.serializer(), schema) +
                        "Do not use single quotes in your response but double quotes,properly escaped with a backslash."
                }.orEmpty()

        val examplesText =
            if (examples.isNotEmpty()) {
                "\n--------EXAMPLES-----------\n" +
                    examples
                        .mapIndexed { index, example ->
                            "Example ${index + 1}\n" +
                                "Input: ${toJson(example.input, pretty = true, excludeNulls = false)}\n" +
                                "Output: ${toJson(example.output, pretty = true, excludeNulls = false)}"
                        }.joinToString("\n\n")
            } else {
                ""
            }

        return buildString {
            append(rendered)
            append("\n")
            append(outputSignature)
            append("\n")
            append(examplesText)
            if (includeInputOutputFrame) {
                append("\n-----------------------------\n")
                append("\nNow perform the same with the following input\n")
                append("input: ")
                append(toJson(inputs, pretty = true, excludeNulls = true))
                append("\n")
                append("Output: ")
            }
        }
    }

    private fun toJson(
        values: Map<String, Any?>,
        pretty: Boolean,
        excludeNulls: Boolean,
    ): String {
        val json = if (pretty) prettyJson else compactJson
        return json.encodeToString(JsonElement.serializer(), encodeMap(values, excludeNulls))
    }

    private fun encodeMap(
        values: Map<String, Any?>,
        excludeNulls: Boolean,
    ): JsonObject =
        buildJsonObject {
            values.forEach { (key, value) ->
                if (!(excludeNulls && value == null)) {
                    put(key, encodeValue(value, excludeNulls))
                }
            }
        }

    private fun encodeList(
        values: List<*>,
        excludeNulls: Boolean,
    ): JsonArray =
        buildJsonArray {
            values.forEach { value ->
                add(encodeValue(value, excludeNulls))
            }
        }

    private fun encodeValue(
        value: Any?,
        excludeNulls: Boolean,
    ): JsonElement =
        when (value) {
            null -> {
                JsonNull
            }

            is String -> {
                JsonPrimitive(value)
            }

            is Number -> {
                JsonPrimitive(value)
            }

            is Boolean -> {
                JsonPrimitive(value)
            }

            is Map<*, *> -> {
                val stringKeyMap =
                    value.entries
                        .filter { it.key is String }
                        .associate { (k, v) -> k as String to v }
                encodeMap(stringKeyMap, excludeNulls)
            }

            is List<*> -> {
                encodeList(value, excludeNulls)
            }

            else -> {
                JsonPrimitive(value.toString())
            }
        }

    companion object {
        private val compactJson = Json
        private val prettyJson = Json { prettyPrint = true }
    }
}
