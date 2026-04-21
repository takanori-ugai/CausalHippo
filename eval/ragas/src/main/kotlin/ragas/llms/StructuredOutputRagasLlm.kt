package ragas.llms

/**
 * Optional extension for LLMs that can generate structured scalar/list outputs directly.
 */
interface StructuredOutputRagasLlm {
    /**
     * Generates and parses a numeric value from model output.
     *
     * @param prompt Prompt text input.
     */
    suspend fun generateNumericValue(prompt: String): Double?

    /**
     * Generates and parses one label from a fixed set of choices.
     *
     * @param prompt Prompt text input.
     */
    suspend fun generateDiscreteValue(prompt: String): String?

    /**
     * Generates and parses ranked items from model output.
     *
     * @param prompt Prompt text input.
     */
    suspend fun generateRankingItems(prompt: String): List<String>?
}
