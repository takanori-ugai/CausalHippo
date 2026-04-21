package ragas.metrics.primitives

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
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
 * Implements [RankingMetric].
 *
 * @property name Metric name.
 * @property llm LLM dependency.
 * @property expectedSize Expected ranking length.
 * @property requiredColumns Required dataset columns.
 */
class RankingMetric(
    override val name: String,
    prompt: String,
    override var llm: BaseRagasLlm?,
    private val expectedSize: Int,
    override val requiredColumns: Map<MetricType, Set<String>> = mapOf(MetricType.SINGLE_TURN to setOf("user_input", "response")),
) : BaseMetric(name = name, requiredColumns = requiredColumns, outputType = MetricOutputType.RANKING),
    SingleTurnMetric,
    MetricWithLlm,
    OptimizableMetricPrompt {
    init {
        require(expectedSize > 0) { "expectedSize must be greater than 0" }
    }

    private var promptObject: OptimizerPrompt = OptimizerPrompt.Text(prompt)

    private fun template(): PromptTemplate =
        PromptTemplate(
            instructionTemplate = promptObject.asTextPrompt(),
            outputSchema =
                buildJsonObject {
                    put("type", "object")
                    putJsonObject("properties") {
                        putJsonObject("items") {
                            put("type", "array")
                            putJsonObject("items") {
                                put("type", "string")
                            }
                        }
                    }
                    putJsonArray("required") {
                        add(JsonPrimitive("items"))
                    }
                },
        )

    private val listPrefixPattern =
        Regex(
            "^\\s*(?:[-*•]+\\s*|\\d+[.):-]\\s*|[A-Za-z][.):-]\\s*|item\\s+\\d+\\s*[:.)-]\\s*)",
            RegexOption.IGNORE_CASE,
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

        val parsed =
            if (llmInstance is StructuredOutputRagasLlm) {
                llmInstance
                    .generateRankingItems(prompt)
                    ?.map { item -> item.trim() }
                    ?.filter { item -> item.isNotBlank() } ?: fallbackParse(llmInstance, prompt)
            } else {
                fallbackParse(llmInstance, prompt)
            }

        return parsed.take(expectedSize)
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

    private suspend fun fallbackParse(
        llmInstance: BaseRagasLlm,
        prompt: String,
    ): List<String> {
        val raw =
            llmInstance
                .generateText(prompt = prompt)
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val jsonItems =
            runCatching {
                val element = Json.parseToJsonElement(raw)
                val items = (element as? JsonObject)?.get("items") as? JsonArray
                items?.mapNotNull { (it as? JsonPrimitive)?.content }
            }.getOrNull()

        return (jsonItems ?: splitRankingItems(raw))
            .map { item -> item.trim() }
            .map { item -> item.replaceFirst(listPrefixPattern, "").trim() }
            .filter { item -> item.isNotBlank() }
    }

    private fun splitRankingItems(raw: String): List<String> {
        val trimmed = raw.trim()
        if (trimmed.isEmpty()) {
            return emptyList()
        }
        return if (trimmed.contains('\n')) {
            trimmed.lines()
        } else {
            trimmed.split(',')
        }
    }
}
