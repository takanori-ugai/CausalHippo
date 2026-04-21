package ragas.metrics.defaults

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import ragas.llms.BaseRagasLlm
import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.MetricWithLlm
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.metrics.tokenSet
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig

/**
 * Measures how much of the reference answer/context is covered by retrieved contexts.
 *
 * Uses an LLM attribution classifier when available, with a token-overlap fallback heuristic.
 */
class ContextRecallMetric :
    BaseMetric(
        name = "context_recall",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("user_input", "retrieved_contexts", "reference")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric,
    MetricWithLlm {
    override var llm: BaseRagasLlm? = null

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
        val llmInstance = llm
        if (llmInstance != null) {
            return llmContextRecallScore(sample, llmInstance)
        }
        return fallbackContextRecallScore(sample)
    }

    private suspend fun llmContextRecallScore(
        sample: SingleTurnSample,
        llmInstance: BaseRagasLlm,
    ): Double {
        val retrieved = sample.retrievedContexts.orEmpty()
        val question = sample.userInput.orEmpty().trim()
        val reference = sample.reference.orEmpty().trim()
        require(question.isNotBlank()) { "user_input cannot be empty" }
        require(reference.isNotBlank()) { "reference cannot be empty" }
        require(retrieved.isNotEmpty()) { "retrieved_contexts cannot be empty" }

        val prompt =
            buildString {
                appendLine("Given question, context, and answer, classify each answer sentence as attributable to context.")
                appendLine("Use attributed=1 for attributable, attributed=0 otherwise.")
                appendLine("Return JSON only with this shape:")
                appendLine("{\"classifications\":[{\"statement\":\"...\",\"reason\":\"...\",\"attributed\":0}]}")
                appendLine()
                appendLine("Input:")
                appendLine(
                    "{\"question\":${JsonPrimitive(
                        question,
                    )},\"context\":${JsonPrimitive(retrieved.joinToString("\n"))},\"answer\":${JsonPrimitive(reference)}}",
                )
                append("Output:")
            }

        val raw =
            llmInstance
                .generateText(prompt = prompt)
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return Double.NaN
        val classifications = parsed["classifications"]?.jsonArray.orEmpty()
        if (classifications.isEmpty()) {
            return Double.NaN
        }

        val normalizedAttributions =
            classifications.mapNotNull { element ->
                val obj = element as? JsonObject ?: return@mapNotNull null
                val attribution = LlmJsonSupport.readIntLike(obj, "attributed") ?: return@mapNotNull null
                if (attribution == 0 || attribution == 1) {
                    attribution
                } else {
                    null
                }
            }
        if (normalizedAttributions.size != classifications.size) {
            return Double.NaN
        }

        val score =
            normalizedAttributions.count { value -> value == 1 }.toDouble() /
                normalizedAttributions.size.toDouble()
        return clamp01(score)
    }

    private fun fallbackContextRecallScore(sample: SingleTurnSample): Double {
        val retrieved = sample.retrievedContexts.orEmpty()
        if (retrieved.isEmpty()) {
            return 0.0
        }
        val references = sample.referenceContexts.orEmpty()

        if (references.isNotEmpty()) {
            val retrievedTokens = retrieved.flatMap { context -> tokenSet(context) }.toSet()
            val referenceTokens = references.flatMap { context -> tokenSet(context) }.toSet()
            if (referenceTokens.isEmpty()) {
                return 0.0
            }
            val covered = referenceTokens.intersect(retrievedTokens).size.toDouble()
            return clamp01(covered / referenceTokens.size.toDouble())
        }

        val answer = sample.reference.orEmpty()
        val answerStatements =
            answer
                .split(Regex("[.!?;]+"))
                .map { statement -> statement.trim() }
                .filter { statement -> statement.isNotBlank() }
        if (answerStatements.isEmpty()) {
            return 0.0
        }

        val contextTokenSet = retrieved.flatMap { context -> tokenSet(context) }.toSet()
        if (contextTokenSet.isEmpty()) {
            return 0.0
        }

        val attributedCount =
            answerStatements.count { statement ->
                val statementTokens = tokenSet(statement)
                statementTokens.isNotEmpty() &&
                    statementTokens.intersect(contextTokenSet).size.toDouble() /
                    statementTokens.size.toDouble() >= MIN_ATTRIBUTION_COVERAGE
            }
        return clamp01(attributedCount.toDouble() / answerStatements.size.toDouble())
    }

    private companion object {
        const val MIN_ATTRIBUTION_COVERAGE = 0.5
    }
}
