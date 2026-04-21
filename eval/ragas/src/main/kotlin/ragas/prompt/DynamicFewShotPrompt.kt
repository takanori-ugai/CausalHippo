package ragas.prompt

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import ragas.VERSION
import ragas.embeddings.BaseRagasEmbedding
import java.io.File

@Serializable
private data class SavedDynamicPrompt(
    @SerialName("ragas_version")
    val ragasVersion: String,
    @SerialName("original_hash")
    val originalHash: String,
    val language: String,
    val instruction: String,
    val examples: List<PromptExample>,
    val outputJsonSchema: String? = null,
    val includeInputOutputFrame: Boolean = true,
    @SerialName("max_similar_examples")
    val maxSimilarExamples: Int = 3,
)

/**
 * Prompt template that selects and injects dynamic few-shot examples per input.
 *
 * @property instruction Instruction text.
 * @property examples Few-shot examples used for similarity selection.
 * @property outputJsonSchema Optional JSON schema guidance.
 * @property includeInputOutputFrame Whether to append the input/output frame.
 * @property language Prompt language metadata.
 * @property originalHash Original prompt hash before adaptations.
 * @property maxSimilarExamples Maximum examples selected per render.
 * @property embeddings Optional embeddings model for similarity scoring.
 */
class DynamicFewShotPrompt(
    val instruction: String,
    val examples: List<PromptExample> = emptyList(),
    val outputJsonSchema: String? = null,
    val includeInputOutputFrame: Boolean = true,
    val language: String = "english",
    val originalHash: String? = null,
    val maxSimilarExamples: Int = 3,
    var embeddings: BaseRagasEmbedding? = null,
) {
    private var cachedExampleVectors: List<List<Float>>? = null
    private var cachedEmbeddingsModel: BaseRagasEmbedding? = null
    private val cacheMutex = Mutex()

    init {
        require(maxSimilarExamples > 0) { "maxSimilarExamples must be greater than 0." }
    }

    /**
     * Renders the prompt text for the provided input.
     *
     * @param values Template values map.
     */
    suspend fun format(values: Map<String, Any?>): String {
        val selected =
            if (examples.isEmpty()) {
                emptyList()
            } else {
                selectRelevantExamples(values)
            }

        return basePrompt(selected).format(values)
    }

    /**
     * Returns a copy with an additional few-shot example.
     *
     * @param input Input payload.
     * @param output Output payload.
     */
    fun addExample(
        input: Map<String, String>,
        output: Map<String, String>,
    ): DynamicFewShotPrompt = copy(examples = examples + PromptExample(input, output))

    /**
     * Saves the prompt definition to disk.
     *
     * @param path Filesystem path.
     * @param overwrite Whether an existing file at the path should be overwritten.
     */
    fun save(
        path: String,
        overwrite: Boolean = false,
    ) {
        val file = File(path)
        file.parentFile?.mkdirs()
        if (!overwrite) {
            require(!file.exists()) { "The file '$path' already exists." }
        }
        val payload =
            SavedDynamicPrompt(
                ragasVersion = VERSION,
                originalHash = stableHash(),
                language = language,
                instruction = instruction,
                examples = examples,
                outputJsonSchema = outputJsonSchema,
                includeInputOutputFrame = includeInputOutputFrame,
                maxSimilarExamples = maxSimilarExamples,
            )
        file.writeText(prettyJson.encodeToString(payload))
    }

    /**
     * Computes a deterministic hash of the prompt content.
     */
    fun stableHash(): String =
        SimplePrompt(
            instruction = instruction,
            examples = examples,
            outputJsonSchema = outputJsonSchema,
            includeInputOutputFrame = includeInputOutputFrame,
            language = language,
        ).stableHash() + "|k=$maxSimilarExamples"

    /**
     * Returns a copy with the originalHash field set to stableHash.
     */
    fun withOriginalHash(): DynamicFewShotPrompt = copy(originalHash = stableHash())

    private suspend fun selectRelevantExamples(values: Map<String, Any?>): List<PromptExample> {
        val embeddingModel = embeddings ?: return examples.take(maxSimilarExamples)
        val queryText = canonicalizeQuery(values)
        val queryVector = embeddingModel.embedText(queryText)

        val exampleVectors =
            cacheMutex.withLock {
                if (cachedExampleVectors != null && cachedEmbeddingsModel === embeddingModel) {
                    cachedExampleVectors!!
                } else {
                    val exampleTexts = examples.map { example -> canonicalizeExampleInput(example.input) }
                    val vectors = embeddingModel.embedTexts(exampleTexts)
                    cachedExampleVectors = vectors
                    cachedEmbeddingsModel = embeddingModel
                    vectors
                }
            }

        val scored =
            examples
                .zip(exampleVectors)
                .map { (example, vector) ->
                    example to cosineSimilarity(queryVector, vector)
                }.sortedByDescending { (_, score) -> score }

        return scored.take(maxSimilarExamples).map { (example) -> example }
    }

    private fun basePrompt(selectedExamples: List<PromptExample>): SimplePrompt =
        SimplePrompt(
            instruction = instruction,
            examples = selectedExamples,
            outputJsonSchema = outputJsonSchema,
            includeInputOutputFrame = includeInputOutputFrame,
            language = language,
            originalHash = originalHash,
        )

    private fun canonicalizeExampleInput(values: Map<String, String>): String =
        values.entries
            .sortedBy { (key) -> key }
            .joinToString("\n") { (key, value) -> "$key: $value" }

    private fun canonicalizeQuery(values: Map<String, Any?>): String =
        values.entries
            .sortedBy { (key) -> key }
            .joinToString("\n") { (key, value) -> "$key: ${value?.toString().orEmpty()}" }

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

    companion object {
        private val logger = KotlinLogging.logger {}
        private val prettyJson = Json { prettyPrint = true }
        private val compactJson = Json

        fun load(path: String): DynamicFewShotPrompt {
            val file = File(path)
            require(file.exists()) { "Prompt file not found: $path" }
            val saved =
                compactJson.decodeFromString(
                    SavedDynamicPrompt.serializer(),
                    file.readText(),
                )
            if (saved.ragasVersion != VERSION) {
                logger.warn {
                    "Prompt was saved with Ragas v${saved.ragasVersion}, " +
                        "but current runtime is v$VERSION. " +
                        "There might be incompatibilities."
                }
            }
            val loaded =
                DynamicFewShotPrompt(
                    instruction = saved.instruction,
                    examples = saved.examples,
                    outputJsonSchema = saved.outputJsonSchema,
                    includeInputOutputFrame = saved.includeInputOutputFrame,
                    language = saved.language,
                    originalHash = saved.originalHash,
                    maxSimilarExamples = saved.maxSimilarExamples,
                )
            if (loaded.stableHash() != saved.originalHash) {
                logger.warn { "Loaded prompt hash does not match the saved hash." }
            }
            return loaded
        }
    }

    private fun copy(
        instruction: String = this.instruction,
        examples: List<PromptExample> = this.examples,
        outputJsonSchema: String? = this.outputJsonSchema,
        includeInputOutputFrame: Boolean = this.includeInputOutputFrame,
        language: String = this.language,
        originalHash: String? = this.originalHash,
        maxSimilarExamples: Int = this.maxSimilarExamples,
        embeddings: BaseRagasEmbedding? = this.embeddings,
    ): DynamicFewShotPrompt =
        DynamicFewShotPrompt(
            instruction = instruction,
            examples = examples,
            outputJsonSchema = outputJsonSchema,
            includeInputOutputFrame = includeInputOutputFrame,
            language = language,
            originalHash = originalHash,
            maxSimilarExamples = maxSimilarExamples,
            embeddings = embeddings,
        )
}
