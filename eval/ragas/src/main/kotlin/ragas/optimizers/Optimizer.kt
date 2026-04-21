package ragas.optimizers

import ragas.prompt.PromptContentPart

/**
 * Single optimization training example.
 *
 * @property promptInput Prompt input fields.
 * @property expectedOutput Expected output text.
 */
data class OptimizationExample(
    val promptInput: Map<String, String>,
    val expectedOutput: String,
)

/**
 * Dataset used to score candidate prompts for one metric.
 *
 * @property metricName Metric being optimized.
 * @property examples Training examples used by evaluator scoring.
 */
data class OptimizationDataset(
    val metricName: String,
    val examples: List<OptimizationExample>,
)

/**
 * Generic optimizer hyperparameters.
 *
 * @property iterations Number of optimizer iterations.
 * @property populationSize Candidate count per generation.
 * @property mutationProbability Probability of mutating a candidate.
 * @property seed Random seed for deterministic behavior.
 */
data class OptimizerConfig(
    val iterations: Int = 3,
    val populationSize: Int = 6,
    val mutationProbability: Double = 0.2,
    val seed: Int = 42,
) {
    init {
        require(iterations > 0) { "iterations must be > 0" }
        require(populationSize > 0) { "populationSize must be > 0" }
        require(mutationProbability in 0.0..1.0) { "mutationProbability must be between 0 and 1" }
    }
}

/** Legacy evaluator operating on plain text prompts. */
fun interface PromptEvaluator {
    /**
     * Returns higher-is-better score for a prompt on [dataset].
     *
     * @param prompt Prompt text to evaluate.
     * @param dataset Dataset to score against.
     */
    fun score(
        prompt: String,
        dataset: OptimizationDataset,
    ): Double
}

/** Prompt candidate type, supporting text-only and multimodal forms. */
sealed interface OptimizerPrompt {
    /**
     * Text-only prompt payload.
     *
     * @property value Plain text prompt.
     */
    data class Text(
        val value: String,
    ) : OptimizerPrompt

    /**
     * Multimodal prompt payload.
     *
     * @property content Ordered multimodal prompt parts.
     */
    data class MultiModal(
        val content: List<PromptContentPart>,
    ) : OptimizerPrompt
}

/**
 * Flattens an [OptimizerPrompt] into a single text prompt representation.
 */
fun OptimizerPrompt.asTextPrompt(): String =
    when (this) {
        is OptimizerPrompt.Text -> value
        is OptimizerPrompt.MultiModal -> content.joinToString(separator = "\n") { it.toPromptText() }
    }

/** Evaluator operating on structured [OptimizerPrompt] values. */
fun interface PromptObjectEvaluator {
    /**
     * Returns higher-is-better score for a prompt on [dataset].
     *
     * @param prompt Prompt object to evaluate.
     * @param dataset Dataset to score against.
     */
    fun score(
        prompt: OptimizerPrompt,
        dataset: OptimizationDataset,
    ): Double
}

/**
 * Final optimization output and additional metadata emitted by the optimizer.
 *
 * @property optimizedPrompt Best prompt candidate found.
 * @property metadata Additional optimizer metadata.
 */
data class OptimizerOutcome(
    val optimizedPrompt: OptimizerPrompt,
    val metadata: Map<String, String> = emptyMap(),
) {
    /** Converts the outcome to the legacy map-based API shape. */
    fun toLegacyMap(): Map<String, String> {
        val base =
            when (optimizedPrompt) {
                is OptimizerPrompt.Text -> {
                    mapOf("optimized_prompt" to optimizedPrompt.value)
                }

                is OptimizerPrompt.MultiModal -> {
                    mapOf(
                        "optimized_prompt" to optimizedPrompt.asTextPrompt(),
                        "optimized_prompt_type" to "multimodal",
                    )
                }
            }
        // Keep legacy core keys authoritative even when metadata contains the same names.
        return metadata + base
    }
}

/** Contract implemented by prompt-optimization algorithms. */
interface Optimizer {
    /**
     * Optimizes structured prompt candidates and returns the best outcome.
     *
     * @param dataset Dataset used for scoring.
     * @param initialPrompts Initial prompt candidates.
     * @param evaluator Evaluator used to score each candidate.
     * @param config Optimizer hyperparameters.
     */
    fun optimizePrompts(
        dataset: OptimizationDataset,
        initialPrompts: List<OptimizerPrompt>,
        evaluator: PromptObjectEvaluator,
        config: OptimizerConfig = OptimizerConfig(),
    ): OptimizerOutcome

    /**
     * Legacy text-prompt optimization API.
     *
     * This adapts string prompts/evaluator into [optimizePrompts].
     *
     * @param dataset Dataset to evaluate or optimize.
     * @param initialPrompts Initial prompt candidates.
     * @param evaluator Evaluator used for scoring prompt candidates.
     * @param config Optimizer configuration.
     */
    fun optimize(
        dataset: OptimizationDataset,
        initialPrompts: List<String>,
        evaluator: PromptEvaluator,
        config: OptimizerConfig = OptimizerConfig(),
    ): Map<String, String> =
        optimizePrompts(
            dataset = dataset,
            initialPrompts = initialPrompts.map { OptimizerPrompt.Text(it) },
            evaluator =
                PromptObjectEvaluator { prompt, ds ->
                    when (prompt) {
                        is OptimizerPrompt.Text -> evaluator.score(prompt.value, ds)
                        is OptimizerPrompt.MultiModal -> evaluator.score(prompt.asTextPrompt(), ds)
                    }
                },
            config = config,
        ).toLegacyMap()
}
