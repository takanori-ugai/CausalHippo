package ragas.metrics.primitives

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import ragas.llms.BaseRagasLlm
import ragas.llms.StructuredOutputRagasLlm
import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.MetricWithLlm
import ragas.metrics.SingleTurnMetric
import ragas.model.SingleTurnSample
import ragas.optimizers.OptimizerPrompt
import ragas.optimizers.asTextPrompt
import ragas.runtime.RunConfig

/**
 * Implements [DiscreteMetric].
 *
 * @property name Metric name.
 * @property llm LLM dependency.
 * @property allowedValues Allowed label set.
 * @property requiredColumns Required dataset columns.
 */
class DiscreteMetric(
    override val name: String,
    prompt: String,
    override var llm: BaseRagasLlm?,
    private val allowedValues: List<String>,
    override val requiredColumns: Map<MetricType, Set<String>> = mapOf(MetricType.SINGLE_TURN to setOf("user_input", "response")),
) : BaseMetric(name = name, requiredColumns = requiredColumns, outputType = MetricOutputType.DISCRETE),
    SingleTurnMetric,
    MetricWithLlm,
    OptimizableMetricPrompt {
    init {
        require(allowedValues.isNotEmpty()) { "allowedValues cannot be empty" }
    }

    private var promptObject: OptimizerPrompt = OptimizerPrompt.Text(prompt)

    private fun template(): PromptTemplate =
        PromptTemplate(
            instructionTemplate = promptObject.asTextPrompt(),
            outputSchema =
                buildJsonObject {
                    put("type", "object")
                    putJsonObject("properties") {
                        putJsonObject("value") {
                            put("type", "string")
                            putJsonArray("enum") {
                                allowedValues.forEach { value -> add(JsonPrimitive(value)) }
                            }
                        }
                    }
                    putJsonArray("required") {
                        add(JsonPrimitive("value"))
                    }
                },
        )

    /**
     * Executes init.
     * @param runConfig Runtime configuration for model calls and execution behavior.
     */
    override suspend fun init(runConfig: RunConfig) {
        validateRequiredColumns()
        llm?.runConfig = runConfig
    }

    /**
     * Executes singleTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any? {
        val llmInstance = checkNotNull(llm) { "Metric '$name' has no LLM configured." }
        val prompt =
            template().render(
                mapOf(
                    "user_input" to sample.userInput.orEmpty(),
                    "response" to sample.response.orEmpty(),
                    "reference" to sample.reference.orEmpty(),
                    "retrieved_contexts" to sample.retrievedContexts.orEmpty().joinToString("\n"),
                ),
            )
        val raw =
            if (llmInstance is StructuredOutputRagasLlm) {
                val structured = llmInstance.generateDiscreteValue(prompt)
                if (structured != null) {
                    structured
                } else {
                    val text = generateRawText(llmInstance, prompt)
                    parseJsonValue(text) ?: text
                }
            } else {
                val text = generateRawText(llmInstance, prompt)
                parseJsonValue(text) ?: text
            }.trim()

        val normalized = raw.lowercase()
        val selected =
            allowedValues
                .sortedByDescending { it.length }
                .firstOrNull { allowed ->
                    val candidate = allowed.lowercase()
                    normalized == candidate ||
                        Regex("(?<![\\p{L}\\p{N}_])${Regex.escape(candidate)}(?![\\p{L}\\p{N}_])")
                            .containsMatchIn(normalized)
                }
        return selected
    }

    /**
     * Executes optimizerPrompt.
     */
    override fun optimizerPrompt(): OptimizerPrompt = promptObject

    /**
     * Executes applyOptimizerPrompt.
     * @param prompt Prompt object returned by the optimizer (text or multimodal).
     */
    override fun applyOptimizerPrompt(prompt: OptimizerPrompt) {
        promptObject = prompt
    }

    private suspend fun generateRawText(
        llmInstance: BaseRagasLlm,
        prompt: String,
    ): String =
        llmInstance
            .generateText(prompt = prompt)
            .generations
            .firstOrNull()
            ?.text
            .orEmpty()

    private fun parseJsonValue(raw: String): String? =
        runCatching {
            val element = Json.parseToJsonElement(raw)
            val value = (element as? JsonObject)?.get("value") as? JsonPrimitive
            value?.content
        }.getOrNull()
}
