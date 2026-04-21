package ragas.llms

import ragas.runtime.RunConfig

/**
 * One generated candidate returned by an LLM call.
 *
 * @property text Text content.
 * @property finishReason Reason the generation stopped (e.g., "STOP", "LENGTH"), or null if unspecified.
 */
data class LlmGeneration(
    val text: String,
    val finishReason: String? = null,
)

/**
 * Container for one LLM response containing one or more generations.
 *
 * @property generations List of generated candidates from the LLM call.
 */
data class LlmResult(
    val generations: List<LlmGeneration>,
)

/**
 * Core LLM contract used by prompts and metrics.
 */
interface BaseRagasLlm {
    var runConfig: RunConfig

    /**
     * Generates one or more text completions for the supplied prompt.
     *
     * @param prompt Prompt text input.
     * @param n Number of generations requested.
     * @param temperature Sampling temperature.
     * @param stop Stop token list.
     */
    suspend fun generateText(
        prompt: String,
        n: Int = 1,
        temperature: Double? = 0.01,
        stop: List<String>? = null,
    ): LlmResult

    /**
     * Returns true when all generations completed normally.
     *
     * A generation is considered complete if its finish reason is null, "STOP", or "LENGTH".
     *
     * @param result LLM result containing generations to check.
     */
    fun isFinished(result: LlmResult): Boolean =
        result.generations.all { generation ->
            generation.finishReason == null ||
                generation.finishReason.equals("STOP", ignoreCase = true) ||
                generation.finishReason.equals("LENGTH", ignoreCase = true)
        }
}
