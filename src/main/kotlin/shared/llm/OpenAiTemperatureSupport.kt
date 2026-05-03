package shared.llm

private val openAiModelTemperatureSupport =
    mapOf(
        "gpt-5-mini" to false,
        "gpt-4o-mini" to true,
    )

/**
 * Returns whether the OpenAI-compatible [modelName] supports configuring a temperature value.
 *
 * Unknown models default to `true` to preserve backward-compatible behavior.
 */
fun supportsTemperature(modelName: String): Boolean = openAiModelTemperatureSupport[modelName.lowercase()] ?: true
