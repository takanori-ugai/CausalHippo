package ragas.prompt

import kotlinx.serialization.KSerializer
import ragas.llms.BaseRagasLlm
import ragas.llms.MultiModalRagasLlm

/**
 * Typed prompt that renders mixed text/image content and parses structured output.
 */
class ImageTextTypedPrompt<InputT, OutputT>(
    inputSerializer: KSerializer<InputT>,
    outputSerializer: KSerializer<OutputT>,
    override val model: TypedPromptModel<InputT, OutputT>,
    private val inputContentBuilder: (InputT) -> List<PromptContentPart>,
) : BasePrompt<InputT, OutputT>(
        inputSerializer = inputSerializer,
        outputSerializer = outputSerializer,
        model = model,
    ) {
    /**
     * Builds prompt content parts from typed input for multimodal generation.
     *
     * @param input Input payload.
     */
    fun toContent(input: InputT?): List<PromptContentPart> {
        val parts = mutableListOf<PromptContentPart>()
        parts += PromptContentPart.Text(model.instruction)
        parts += PromptContentPart.Text("")

        if (!model.outputJsonSchema.isNullOrBlank()) {
            parts +=
                PromptContentPart.Text(
                    "Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:",
                )
            parts += PromptContentPart.Text(model.outputJsonSchema)
            parts +=
                PromptContentPart.Text(
                    "Do not use single quotes in your response. Use double quotes, properly escaped with a backslash where necessary.",
                )
        } else {
            parts += PromptContentPart.Text("Return JSON only.")
        }

        if (model.examples.isNotEmpty()) {
            parts += PromptContentPart.Text("\n--------EXAMPLES-----------")
            model.examples.forEachIndexed { index, example ->
                parts += PromptContentPart.Text("Example ${index + 1}")
                parts += PromptContentPart.Text("Input: ${json.encodeToString(inputSerializer, example.input)}")
                parts += PromptContentPart.Text("Output: ${json.encodeToString(outputSerializer, example.output)}")
            }
        }

        if (model.includeInputOutputFrame) {
            parts += PromptContentPart.Text("\n-----------------------------")
            parts += PromptContentPart.Text("Now perform the same with the following input")
            if (input == null) {
                parts += PromptContentPart.Text("Input: (None)")
            } else {
                parts += PromptContentPart.Text("Input:")
                parts += inputContentBuilder(input)
            }
            parts += PromptContentPart.Text("Output: ")
        }

        return parts
    }

    /**
     * Renders the prompt text for the provided input.
     */
    override suspend fun format(input: InputT?): String = toContent(input).joinToString(separator = "\n") { it.toPromptText() }

    /**
     * Generates and parses structured output using configured retry behavior.
     */
    override suspend fun generate(
        llm: BaseRagasLlm,
        input: InputT?,
        config: StructuredOutputParserConfig,
    ): OutputT {
        require(config.maxParseRetries >= 0) { "maxParseRetries must be >= 0" }

        val baseContent = toContent(input)
        var currentContent = baseContent
        val failures = mutableListOf<StructuredParseAttempt>()

        repeat(config.maxParseRetries + 1) { attempt ->
            val raw =
                if (llm is MultiModalRagasLlm) {
                    llm
                        .generateContent(
                            content = currentContent,
                            temperature = config.temperature,
                            stop = config.stop,
                        ).generations
                        .firstOrNull()
                        ?.text
                        .orEmpty()
                } else {
                    llm
                        .generateText(
                            prompt = currentContent.joinToString(separator = "\n") { it.toPromptText() },
                            temperature = config.temperature,
                            stop = config.stop,
                        ).generations
                        .firstOrNull()
                        ?.text
                        .orEmpty()
                }

            val parsed =
                runCatching { parse(raw) }.getOrElse { error ->
                    failures +=
                        StructuredParseAttempt(
                            attempt = attempt + 1,
                            errorMessage = error.message ?: error::class.simpleName.orEmpty(),
                            rawOutput = raw,
                        )
                    null
                }
            if (parsed != null) {
                return parsed
            }

            if (attempt < config.maxParseRetries) {
                currentContent = buildRetryContent(baseContent, raw, failures.last().errorMessage)
            }
        }

        throw StructuredParseException(failures)
    }

    private fun buildRetryContent(
        originalContent: List<PromptContentPart>,
        previousRawOutput: String,
        parseError: String,
    ): List<PromptContentPart> =
        originalContent +
            listOf(
                PromptContentPart.Text(""),
                PromptContentPart.Text("Your previous output could not be parsed as valid JSON."),
                PromptContentPart.Text("Parse error: $parseError"),
                PromptContentPart.Text("Previous output:"),
                PromptContentPart.Text(previousRawOutput),
                PromptContentPart.Text("Return only corrected JSON."),
            )
}
