package ragas.prompt

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.PolymorphicKind
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.StructureKind
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import ragas.embeddings.BaseRagasEmbedding
import ragas.llms.BaseRagasLlm
import java.util.Collections
import java.util.IdentityHashMap

/** One few-shot input/output example used in prompt rendering. */
@Serializable
data class TypedPromptExample<InputT, OutputT>(
    val input: InputT,
    val output: OutputT,
)

/** Serializable prompt definition for structured generation. */
@Serializable
data class TypedPromptModel<InputT, OutputT>(
    val instruction: String,
    val examples: List<TypedPromptExample<InputT, OutputT>> = emptyList(),
    val outputJsonSchema: String? = null,
    val includeInputOutputFrame: Boolean = true,
    val language: String = "english",
)

/**
 * Retry and generation settings for structured parsing.
 *
 * @property maxParseRetries Maximum retry count after parse failures.
 * @property temperature Sampling temperature passed to the LLM.
 * @property stop Optional stop tokens passed to the LLM.
 */
data class StructuredOutputParserConfig(
    val maxParseRetries: Int = 3,
    val temperature: Double? = 0.01,
    val stop: List<String>? = null,
)

/**
 * One parse failure captured during retry.
 *
 * @property attempt 1-based attempt index.
 * @property errorMessage Error message from parsing failure.
 * @property rawOutput Raw model output captured for the attempt.
 */
data class StructuredParseAttempt(
    val attempt: Int,
    val errorMessage: String,
    val rawOutput: String,
)

/**
 * Thrown when structured output cannot be parsed after all retries.
 *
 * @property attempts Ordered list of parse attempts captured before failing.
 */
class StructuredParseException(
    val attempts: List<StructuredParseAttempt>,
) : IllegalStateException(
        "Failed to parse structured output after ${attempts.size} attempt(s).",
    )

/** JSON parser for typed prompt outputs backed by Kotlin serialization. */
class StructuredOutputParser<T>(
    private val serializer: KSerializer<T>,
    private val json: Json = Json { ignoreUnknownKeys = true },
) {
    private val maxDescriptorDepth = 32
    private val jsonNumberPattern = Regex("""-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?""")

    /**
     * Parses raw model text into [T], extracting the first JSON value when needed.
     *
     * @param rawText Raw model output text.
     */
    fun parse(rawText: String): T {
        val trimmed = rawText.trim()
        require(trimmed.isNotEmpty()) { "Structured output is empty." }

        return runCatching {
            json.decodeFromString(serializer, trimmed)
        }.getOrElse {
            val extracted = extractFirstJsonValue(trimmed)
            json.decodeFromString(serializer, extracted)
        }
    }

    /** Returns a human-readable JSON-shape hint derived from [serializer]. */
    @OptIn(ExperimentalSerializationApi::class)
    fun expectedShapeDescription(): String =
        describeDescriptor(
            descriptor = serializer.descriptor,
            active =
                Collections.newSetFromMap(
                    IdentityHashMap<SerialDescriptor, Boolean>(),
                ),
            depth = 0,
        )

    @OptIn(ExperimentalSerializationApi::class)
    private fun describeDescriptor(
        descriptor: SerialDescriptor,
        active: MutableSet<SerialDescriptor>,
        depth: Int,
    ): String {
        if (depth >= maxDescriptorDepth) {
            return "json"
        }
        if (!active.add(descriptor)) {
            return "recursive"
        }

        return try {
            when (descriptor.kind) {
                PrimitiveKind.BOOLEAN -> {
                    "boolean"
                }

                PrimitiveKind.BYTE,
                PrimitiveKind.SHORT,
                PrimitiveKind.INT,
                PrimitiveKind.LONG,
                PrimitiveKind.FLOAT,
                PrimitiveKind.DOUBLE,
                -> {
                    "number"
                }

                PrimitiveKind.CHAR,
                PrimitiveKind.STRING,
                -> {
                    "string"
                }

                StructureKind.LIST -> {
                    "[${describeDescriptor(descriptor.getElementDescriptor(0), active, depth + 1)}]"
                }

                StructureKind.MAP -> {
                    "{key: ${describeDescriptor(descriptor.getElementDescriptor(1), active, depth + 1)}}"
                }

                StructureKind.CLASS,
                StructureKind.OBJECT,
                -> {
                    val fields =
                        (0 until descriptor.elementsCount).joinToString(", ") { index ->
                            val name = descriptor.getElementName(index)
                            val type = describeDescriptor(descriptor.getElementDescriptor(index), active, depth + 1)
                            "\"$name\": $type"
                        }
                    "{$fields}"
                }

                is PolymorphicKind -> {
                    "polymorphic"
                }

                else -> {
                    "json"
                }
            }
        } finally {
            active.remove(descriptor)
        }
    }

    private fun extractFirstJsonValue(text: String): String {
        for (start in text.indices) {
            if (!isBoundaryBefore(text, start)) {
                continue
            }
            val candidate =
                when (text[start]) {
                    '{', '[' -> extractEnclosedValue(text, start)
                    '"' -> extractJsonString(text, start)
                    't' -> keywordCandidate(text, start, "true")
                    'f' -> keywordCandidate(text, start, "false")
                    'n' -> keywordCandidate(text, start, "null")
                    '-', in '0'..'9' -> extractJsonNumber(text, start)
                    else -> null
                } ?: continue
            if (isValidJson(candidate)) {
                return candidate
            }
        }
        error("Structured output did not contain JSON.")
    }

    private fun extractEnclosedValue(
        text: String,
        start: Int,
    ): String? {
        var inString = false
        var escaped = false
        val stack = ArrayDeque<Char>()

        for (index in start until text.length) {
            val ch = text[index]
            if (inString) {
                when {
                    escaped -> escaped = false
                    ch == '\\' -> escaped = true
                    ch == '"' -> inString = false
                }
                continue
            }

            when (ch) {
                '"' -> {
                    inString = true
                }

                '{' -> {
                    stack.addLast('}')
                }

                '[' -> {
                    stack.addLast(']')
                }

                '}', ']' -> {
                    if (stack.isEmpty() || stack.removeLast() != ch) {
                        return null
                    }
                    if (stack.isEmpty() && isBoundaryAfter(text, index + 1)) {
                        return text.substring(start, index + 1)
                    }
                }
            }
        }
        return null
    }

    private fun extractJsonString(
        text: String,
        start: Int,
    ): String? {
        var escaped = false
        for (index in start + 1 until text.length) {
            val ch = text[index]
            when {
                escaped -> {
                    escaped = false
                }

                ch == '\\' -> {
                    escaped = true
                }

                ch == '"' -> {
                    val endExclusive = index + 1
                    if (!isBoundaryAfter(text, endExclusive)) {
                        return null
                    }
                    return text.substring(start, endExclusive)
                }
            }
        }
        return null
    }

    private fun extractJsonNumber(
        text: String,
        start: Int,
    ): String? {
        val match = jsonNumberPattern.find(text, start) ?: return null
        if (match.range.first != start) {
            return null
        }
        val endExclusive = match.range.last + 1
        if (!isBoundaryAfter(text, endExclusive)) {
            return null
        }
        return match.value
    }

    private fun keywordCandidate(
        text: String,
        start: Int,
        token: String,
    ): String? {
        if (!text.startsWith(token, start)) {
            return null
        }
        val endExclusive = start + token.length
        if (!isBoundaryAfter(text, endExclusive)) {
            return null
        }
        return token
    }

    private fun isBoundaryBefore(
        text: String,
        start: Int,
    ): Boolean {
        if (start == 0) {
            return true
        }
        val previous = text[start - 1]
        return !previous.isLetterOrDigit() && previous != '_' && previous != '"'
    }

    private fun isBoundaryAfter(
        text: String,
        endExclusive: Int,
    ): Boolean {
        if (endExclusive >= text.length) {
            return true
        }
        val next = text[endExclusive]
        return !next.isLetterOrDigit() && next != '_' && next != '"' && next != '-'
    }

    private fun isValidJson(candidate: String): Boolean = runCatching { json.parseToJsonElement(candidate) }.isSuccess
}

/**
 * Base class for typed prompts that render instruction/examples and parse JSON output.
 */
abstract class BasePrompt<InputT, OutputT>(
    protected val inputSerializer: KSerializer<InputT>,
    protected val outputSerializer: KSerializer<OutputT>,
    protected open val model: TypedPromptModel<InputT, OutputT>,
    protected val json: Json =
        Json {
            prettyPrint = true
            explicitNulls = false
        },
) {
    private val parser = StructuredOutputParser(outputSerializer)

    open suspend fun format(input: InputT?): String = render(input = input, examples = model.examples)

    protected fun render(
        input: InputT?,
        examples: List<TypedPromptExample<InputT, OutputT>>,
    ): String {
        val normalizedLanguage = model.language.trim()
        val languageInstruction =
            normalizedLanguage
                .takeIf { it.isNotEmpty() && !it.equals("english", ignoreCase = true) }
                ?.let { language -> "Write the response in $language." }
        val schemaText =
            model.outputJsonSchema
                ?: "Expected JSON shape: ${parser.expectedShapeDescription()}"

        val examplesText =
            if (examples.isEmpty()) {
                ""
            } else {
                buildString {
                    appendLine("--------EXAMPLES-----------")
                    examples.forEachIndexed { index, example ->
                        appendLine("Example ${index + 1}")
                        appendLine("Input: ${json.encodeToString(inputSerializer, example.input)}")
                        appendLine("Output: ${json.encodeToString(outputSerializer, example.output)}")
                        if (index < examples.lastIndex) {
                            appendLine()
                        }
                    }
                }
            }

        return buildString {
            appendLine(model.instruction)
            appendLine()
            appendLine("Return JSON only.")
            if (languageInstruction != null) {
                appendLine(languageInstruction)
            }
            appendLine(schemaText)
            if (examplesText.isNotEmpty()) {
                appendLine()
                append(examplesText)
            }
            if (model.includeInputOutputFrame) {
                appendLine("-----------------------------")
                appendLine("Now perform the same with the following input")
                if (input == null) {
                    appendLine("Input: (None)")
                } else {
                    appendLine("Input: ${json.encodeToString(inputSerializer, input)}")
                }
                append("Output: ")
            }
        }
    }

    /**
     * Parses output text into typed output using [StructuredOutputParser].
     *
     * @param rawOutput Raw model output text.
     */
    fun parse(rawOutput: String): OutputT = parser.parse(rawOutput)

    open suspend fun generate(
        llm: BaseRagasLlm,
        input: InputT?,
        config: StructuredOutputParserConfig = StructuredOutputParserConfig(),
    ): OutputT {
        require(config.maxParseRetries >= 0) { "maxParseRetries must be >= 0" }

        val basePrompt = format(input)
        var currentPrompt = basePrompt
        val failures = mutableListOf<StructuredParseAttempt>()

        repeat(config.maxParseRetries + 1) { attempt ->
            val raw =
                llm
                    .generateText(
                        prompt = currentPrompt,
                        temperature = config.temperature,
                        stop = config.stop,
                    ).generations
                    .firstOrNull()
                    ?.text
                    .orEmpty()

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
                currentPrompt = buildRetryPrompt(basePrompt, raw, failures.last().errorMessage)
            }
        }

        throw StructuredParseException(failures)
    }

    protected open fun buildRetryPrompt(
        originalPrompt: String,
        previousRawOutput: String,
        parseError: String,
    ): String =
        buildString {
            appendLine(originalPrompt)
            appendLine()
            appendLine("Your previous output could not be parsed as valid JSON.")
            appendLine("Parse error: $parseError")
            appendLine("Previous output:")
            appendLine(previousRawOutput)
            appendLine()
            append("Return only corrected JSON.")
        }
}

/** Basic typed prompt implementation with static examples. */
class TypedPrompt<InputT, OutputT>(
    inputSerializer: KSerializer<InputT>,
    outputSerializer: KSerializer<OutputT>,
    override val model: TypedPromptModel<InputT, OutputT>,
) : BasePrompt<InputT, OutputT>(
        inputSerializer = inputSerializer,
        outputSerializer = outputSerializer,
        model = model,
    )

/**
 * Typed prompt with immutable, caller-managed few-shot examples.
 */
open class FewShotTypedPrompt<InputT, OutputT>(
    inputSerializer: KSerializer<InputT>,
    outputSerializer: KSerializer<OutputT>,
    override val model: TypedPromptModel<InputT, OutputT>,
) : BasePrompt<InputT, OutputT>(
        inputSerializer = inputSerializer,
        outputSerializer = outputSerializer,
        model = model,
    ) {
    /**
     * Returns a new prompt with an appended few-shot example.
     *
     * @param input Example input payload.
     * @param output Example output payload.
     */
    fun addExample(
        input: InputT,
        output: OutputT,
    ): FewShotTypedPrompt<InputT, OutputT> =
        FewShotTypedPrompt(
            inputSerializer = inputSerializer,
            outputSerializer = outputSerializer,
            model = model.copy(examples = model.examples + TypedPromptExample(input, output)),
        )
}

/**
 * Few-shot prompt that picks top-k similar examples via embeddings at render time.
 */
open class DynamicFewShotTypedPrompt<InputT, OutputT>(
    inputSerializer: KSerializer<InputT>,
    outputSerializer: KSerializer<OutputT>,
    override val model: TypedPromptModel<InputT, OutputT>,
    private val maxSimilarExamples: Int = 3,
    var embeddings: BaseRagasEmbedding? = null,
) : BasePrompt<InputT, OutputT>(
        inputSerializer = inputSerializer,
        outputSerializer = outputSerializer,
        model = model,
    ) {
    private val compactJson = Json { explicitNulls = false }
    private var cachedExampleVectors: List<List<Float>>? = null
    private var cachedEmbeddingsModel: BaseRagasEmbedding? = null
    private val cacheMutex = Mutex()

    init {
        require(maxSimilarExamples > 0) { "maxSimilarExamples must be greater than 0." }
    }

    /** Renders a prompt using similarity-selected examples when embeddings are available. */
    override suspend fun format(input: InputT?): String {
        val selected = selectExamples(input)
        return render(input = input, examples = selected)
    }

    private suspend fun selectExamples(input: InputT?): List<TypedPromptExample<InputT, OutputT>> {
        if (model.examples.isEmpty()) {
            return emptyList()
        }
        val embeddingModel = embeddings ?: return model.examples.take(maxSimilarExamples)
        val queryText = input?.let { compactJson.encodeToString(inputSerializer, it) } ?: "(None)"

        val queryVector = embeddingModel.embedText(queryText)
        val exampleVectors =
            cacheMutex.withLock {
                if (cachedExampleVectors != null && cachedEmbeddingsModel === embeddingModel) {
                    cachedExampleVectors!!
                } else {
                    val exampleTexts =
                        model.examples.map { example ->
                            compactJson.encodeToString(inputSerializer, example.input)
                        }
                    val vectors = embeddingModel.embedTexts(exampleTexts)
                    cachedExampleVectors = vectors
                    cachedEmbeddingsModel = embeddingModel
                    vectors
                }
            }

        return model.examples
            .zip(exampleVectors)
            .map { (example, vector) ->
                example to cosineSimilarity(queryVector, vector)
            }.sortedByDescending { (_, score) -> score }
            .take(maxSimilarExamples)
            .map { (example) -> example }
    }

    private fun cosineSimilarity(
        a: List<Float>,
        b: List<Float>,
    ): Double {
        if (a.isEmpty() || b.isEmpty() || a.size != b.size) {
            return 0.0
        }
        var dot = 0.0
        var aNorm = 0.0
        var bNorm = 0.0
        for (index in a.indices) {
            val av = a[index].toDouble()
            val bv = b[index].toDouble()
            dot += av * bv
            aNorm += av * av
            bNorm += bv * bv
        }
        if (aNorm == 0.0 || bNorm == 0.0) {
            return 0.0
        }
        return dot / (kotlin.math.sqrt(aNorm) * kotlin.math.sqrt(bNorm))
    }
}

/** Convenience wrapper that configures [FewShotTypedPrompt] with explicit JSON schema text. */
class FewShotPydanticPrompt<InputT, OutputT>(
    inputSerializer: KSerializer<InputT>,
    outputSerializer: KSerializer<OutputT>,
    instruction: String,
    outputJsonSchema: String,
    examples: List<TypedPromptExample<InputT, OutputT>> = emptyList(),
    includeInputOutputFrame: Boolean = true,
    language: String = "english",
) : FewShotTypedPrompt<InputT, OutputT>(
        inputSerializer = inputSerializer,
        outputSerializer = outputSerializer,
        model =
            TypedPromptModel(
                instruction = instruction,
                examples = examples,
                outputJsonSchema = outputJsonSchema,
                includeInputOutputFrame = includeInputOutputFrame,
                language = language,
            ),
    )

/** Dynamic few-shot variant of [FewShotPydanticPrompt]. */
class DynamicFewShotPydanticPrompt<InputT, OutputT>(
    inputSerializer: KSerializer<InputT>,
    outputSerializer: KSerializer<OutputT>,
    instruction: String,
    outputJsonSchema: String,
    examples: List<TypedPromptExample<InputT, OutputT>> = emptyList(),
    includeInputOutputFrame: Boolean = true,
    language: String = "english",
    maxSimilarExamples: Int = 3,
    embeddings: BaseRagasEmbedding? = null,
) : DynamicFewShotTypedPrompt<InputT, OutputT>(
        inputSerializer = inputSerializer,
        outputSerializer = outputSerializer,
        model =
            TypedPromptModel(
                instruction = instruction,
                examples = examples,
                outputJsonSchema = outputJsonSchema,
                includeInputOutputFrame = includeInputOutputFrame,
                language = language,
            ),
        maxSimilarExamples = maxSimilarExamples,
        embeddings = embeddings,
    )
