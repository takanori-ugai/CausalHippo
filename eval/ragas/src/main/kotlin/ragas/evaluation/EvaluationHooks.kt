package ragas.evaluation

import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.llms.MultiModalRagasLlm
import ragas.llms.StructuredOutputRagasLlm
import ragas.model.EvaluationResult
import ragas.prompt.PromptContentPart
import ragas.runtime.Executor

/**
 * Best-effort prompt/completion token accounting for one or more LLM calls.
 *
 * Values are exact only when [TokenUsageParser] extracts provider-reported usage.
 * If parsing fails, tracking falls back to a whitespace-based heuristic.
 *
 * @property promptTokens Best-effort count of tokens attributed to prompts.
 * @property completionTokens Best-effort count of tokens attributed to completions.
 */
data class TokenUsage(
    val promptTokens: Int,
    val completionTokens: Int,
) {
    /** Sum of [promptTokens] and [completionTokens]. */
    val totalTokens: Int = promptTokens + completionTokens
}

/**
 * Best-effort cost estimate derived from [TokenUsage].
 *
 * @property amount Estimated monetary amount, not an exact billing value.
 * @property currency Currency code for [amount].
 */
data class CostEstimate(
    val amount: Double,
    val currency: String = "USD",
)

/** Callback sink for evaluation lifecycle events. */
fun interface EvaluationCallback {
    /**
     * Called whenever an [EvaluationEvent] is emitted.
     *
     * @param event Emitted evaluation event.
     */
    fun onEvent(event: EvaluationEvent)
}

/**
 * Structured failure payload emitted by evaluation lifecycle events.
 *
 * @property type Short error category or exception type.
 * @property message Telemetry-safe error message with sensitive details removed.
 */
data class EvaluationError(
    val type: String,
    val message: String,
)

/** Event stream emitted while evaluation runs. */
sealed interface EvaluationEvent {
    /**
     * Emitted once before any metric work starts.
     *
     * @property sampleCount Number of samples in the evaluation dataset.
     * @property metricNames Names of metrics selected for evaluation.
     */
    data class RunStarted(
        /** Number of samples in the evaluation dataset. */
        val sampleCount: Int,
        /** Names of metrics selected for evaluation. */
        val metricNames: List<String>,
    ) : EvaluationEvent

    /**
     * Emitted once an executor is created and ready to accept jobs.
     *
     * @property executor Executor handling metric tasks.
     */
    data class ExecutorReady(
        /** Executor handling metric tasks. */
        val executor: Executor,
    ) : EvaluationEvent

    /**
     * Emitted after one metric value is computed for one row.
     *
     * @property rowIndex Dataset row index.
     * @property metricName Metric name.
     * @property value Computed metric value.
     */
    data class MetricComputed(
        /** Dataset row index. */
        val rowIndex: Int,
        /** Metric name. */
        val metricName: String,
        /** Computed metric value. */
        val value: Any?,
    ) : EvaluationEvent

    /**
     * Emitted when all selected metrics are available for a row.
     *
     * @property rowIndex Dataset row index.
     * @property scores Map of metric name to computed value for the row.
     */
    data class RowCompleted(
        /** Dataset row index. */
        val rowIndex: Int,
        /** Map of metric name to computed value for the row. */
        val scores: Map<String, Any?>,
    ) : EvaluationEvent

    /**
     * Emitted after aggregated token usage is computed.
     *
     * @property usage Aggregated token usage.
     */
    data class TokenUsageComputed(
        /** Aggregated token usage. */
        val usage: TokenUsage,
    ) : EvaluationEvent

    /**
     * Emitted after optional cost estimation is computed.
     *
     * @property cost Computed cost estimate.
     */
    data class CostComputed(
        /** Computed cost estimate. */
        val cost: CostEstimate,
    ) : EvaluationEvent

    /**
     * Emitted when evaluation completes successfully.
     *
     * @property result Final evaluation result.
     */
    data class RunCompleted(
        /** Final evaluation result. */
        val result: EvaluationResult,
    ) : EvaluationEvent

    /**
     * Emitted when evaluation fails with an exception.
     *
     * @property error Terminal failure payload.
     */
    data class RunFailed(
        /** Terminal failure payload. */
        val error: EvaluationError,
    ) : EvaluationEvent {
        @Deprecated(
            message = "Passing Throwable directly is deprecated; use RunFailed.fromThrowable(error) or EvaluationError.",
            replaceWith = ReplaceWith("EvaluationEvent.RunFailed.fromThrowable(error)"),
            level = DeprecationLevel.WARNING,
        )
        constructor(error: Throwable) : this(fromThrowable(error).error)

        companion object {
            /** Builds [RunFailed] from a Throwable while exposing only a safe error payload. */
            fun fromThrowable(error: Throwable): RunFailed =
                RunFailed(
                    error =
                        EvaluationError(
                            type = error::class.simpleName ?: error::class.qualifiedName ?: "UnknownError",
                            message = "Error details omitted for safety.",
                        ),
                )
        }
    }
}

/**
 * User-supplied parser that extracts token usage from an LLM call.
 *
 * Returning `null` triggers heuristic token estimation.
 */
typealias TokenUsageParser = (prompt: String, result: LlmResult) -> TokenUsage?

/**
 * User-supplied parser that turns token usage into a cost estimate.
 *
 * The resulting value should be treated as best-effort unless it uses exact provider billing rules.
 */
typealias CostParser = (usage: TokenUsage) -> CostEstimate?

internal open class TrackingRagasLlm private constructor(
    protected val delegate: BaseRagasLlm,
    private val tokenUsageParser: TokenUsageParser,
    private val onUsage: (TokenUsage) -> Unit,
) : BaseRagasLlm,
    MultiModalRagasLlm {
    override var runConfig = delegate.runConfig
        set(value) {
            field = value
            delegate.runConfig = value
        }

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        val result = delegate.generateText(prompt, n, temperature, stop)
        onUsage(tokenUsageParser(prompt, result) ?: heuristicTokenUsage(prompt, result))
        return result
    }

    override suspend fun generateContent(
        content: List<PromptContentPart>,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        val multimodal = delegate as? MultiModalRagasLlm
        val promptText = content.joinToString("\n") { part -> part.toPromptText() }
        val result =
            if (multimodal != null) {
                multimodal.generateContent(content, n, temperature, stop)
            } else {
                delegate.generateText(promptText, n, temperature, stop)
            }
        onUsage(tokenUsageParser(promptText, result) ?: heuristicTokenUsage(promptText, result))
        return result
    }

    protected fun onStructuredUsage(
        prompt: String,
        completion: String,
    ) {
        val result = LlmResult(generations = listOf(LlmGeneration(completion)))
        onUsage(tokenUsageParser(prompt, result) ?: heuristicTokenUsage(prompt, result))
    }

    private fun heuristicTokenUsage(
        prompt: String,
        result: LlmResult,
    ): TokenUsage {
        fun countTokens(text: String): Int =
            text
                .trim()
                .split(Regex("\\s+"))
                .count { token -> token.isNotBlank() }

        val promptTokens = countTokens(prompt)
        val completionTokens = result.generations.sumOf { generation -> countTokens(generation.text) }
        return TokenUsage(promptTokens = promptTokens, completionTokens = completionTokens)
    }

    companion object {
        operator fun invoke(
            delegate: BaseRagasLlm,
            tokenUsageParser: TokenUsageParser,
            onUsage: (TokenUsage) -> Unit,
        ): TrackingRagasLlm =
            if (delegate is StructuredOutputRagasLlm) {
                TrackingStructuredRagasLlm(delegate, tokenUsageParser, onUsage)
            } else {
                TrackingRagasLlm(delegate, tokenUsageParser, onUsage)
            }
    }

    private class TrackingStructuredRagasLlm(
        delegate: BaseRagasLlm,
        tokenUsageParser: TokenUsageParser,
        onUsage: (TokenUsage) -> Unit,
    ) : TrackingRagasLlm(delegate, tokenUsageParser, onUsage),
        StructuredOutputRagasLlm {
        private val structuredDelegate =
            delegate as? StructuredOutputRagasLlm
                ?: error("Delegate LLM does not support structured output.")

        override suspend fun generateNumericValue(prompt: String): Double? {
            val result = structuredDelegate.generateNumericValue(prompt)
            if (result != null) {
                onStructuredUsage(prompt, result.toString())
            }
            return result
        }

        override suspend fun generateDiscreteValue(prompt: String): String? {
            val result = structuredDelegate.generateDiscreteValue(prompt)
            if (result != null) {
                onStructuredUsage(prompt, result)
            }
            return result
        }

        override suspend fun generateRankingItems(prompt: String): List<String>? {
            val result = structuredDelegate.generateRankingItems(prompt)
            if (result != null) {
                onStructuredUsage(prompt, result.joinToString("\n"))
            }
            return result
        }
    }
}
