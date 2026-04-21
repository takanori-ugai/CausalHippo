package shared.llm

private val openAiModelTemperatureSupport =
    mapOf(
        "gpt-5-mini" to false,
        "gpt-4o-mini" to true,
    )

fun supportsTemperature(modelName: String): Boolean = openAiModelTemperatureSupport[modelName.lowercase()] ?: true
