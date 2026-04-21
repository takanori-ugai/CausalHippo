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
 * Implements [NumericMetric].
 *
 * @property name Metric name.
 * @property llm LLM dependency.
 * @property allowedRange Allowed numeric score range.
 * @property requiredColumns Required dataset columns.
 */
class NumericMetric(
    override val name: String,
    prompt: String,
    override var llm: BaseRagasLlm?,
    private val allowedRange: ClosedFloatingPointRange<Double> = 0.0..1.0,
    override val requiredColumns: Map<MetricType, Set<String>> = mapOf(MetricType.SINGLE_TURN to setOf("user_input", "response")),
) : BaseMetric(name = name, requiredColumns = requiredColumns, outputType = MetricOutputType.CONTINUOUS),
    SingleTurnMetric,
    MetricWithLlm,
    OptimizableMetricPrompt {
    private var promptObject: OptimizerPrompt = OptimizerPrompt.Text(prompt)

    private fun template(): PromptTemplate =
        PromptTemplate(
            instructionTemplate = promptObject.asTextPrompt(),
            outputSchema =
                buildJsonObject {
                    put("type", "object")
                    putJsonObject("properties") {
                        putJsonObject("value") {
                            put("type", "number")
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
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
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
        val numeric =
            if (llmInstance is StructuredOutputRagasLlm) {
                val structured = llmInstance.generateNumericValue(prompt)
                if (structured != null) {
                    structured
                } else {
                    val raw = generateRawResponse(llmInstance, prompt)
                    parseJsonValue(raw) ?: extractFirstNumber(raw)
                }
            } else {
                val raw = generateRawResponse(llmInstance, prompt)
                parseJsonValue(raw) ?: extractFirstNumber(raw)
            }
                ?: error("Metric '$name' could not parse a numeric score from LLM response.")
        val clamped =
            when {
                numeric < allowedRange.start -> allowedRange.start
                numeric > allowedRange.endInclusive -> allowedRange.endInclusive
                else -> numeric
            }
        return clamped
    }

    /**
     * Executes optimizerPrompt.
     */
    override fun optimizerPrompt(): OptimizerPrompt = promptObject

    /**
     * Executes applyOptimizerPrompt.
     * @param prompt Prompt text returned by the optimizer.
     */
    override fun applyOptimizerPrompt(prompt: OptimizerPrompt) {
        promptObject = prompt
    }

    private fun parseJsonValue(raw: String): Double? =
        runCatching {
            val element = Json.parseToJsonElement(raw)
            val value = (element as? JsonObject)?.get("value") as? JsonPrimitive
            value?.content?.toDoubleOrNull()
        }.getOrNull()

    private fun extractFirstNumber(text: String): Double? {
        val match = numberRegex.find(text)
        return match?.value?.toDoubleOrNull()
    }

    companion object {
        private val numberRegex = Regex("[-+]?[0-9]*\\.?[0-9]+")
    }

    private suspend fun generateRawResponse(
        llmInstance: BaseRagasLlm,
        prompt: String,
    ): String =
        llmInstance
            .generateText(prompt = prompt)
            .generations
            .firstOrNull()
            ?.text
            .orEmpty()
}
