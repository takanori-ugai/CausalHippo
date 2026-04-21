package ragas.metrics.primitives

import ragas.optimizers.OptimizationDataset
import ragas.optimizers.Optimizer
import ragas.optimizers.OptimizerConfig
import ragas.optimizers.OptimizerOutcome
import ragas.optimizers.OptimizerPrompt
import ragas.optimizers.PromptObjectEvaluator

/**
 * Contract for metrics whose prompt can be read, optimized, and replaced.
 */
interface OptimizableMetricPrompt {
    /**
     * Returns the current prompt payload used by this metric.
     */
    fun optimizerPrompt(): OptimizerPrompt

    /**
     * Replaces the metric's current prompt payload.
     *
     * @param prompt Optimizer prompt payload.
     */
    fun applyOptimizerPrompt(prompt: OptimizerPrompt)
}

/**
 * Applies [outcome.optimizedPrompt] to this prompt holder.
 *
 * Side effect: mutates the prompt state of the receiver.
 *
 * @param outcome Optimizer outcome payload.
 */
fun OptimizableMetricPrompt.applyOptimizerOutcome(outcome: OptimizerOutcome) {
    applyOptimizerPrompt(outcome.optimizedPrompt)
}

/**
 * Runs optimization using the current prompt as the initial candidate, then applies the best prompt.
 *
 * Side effect: mutates the receiver by calling [applyOptimizerOutcome] with the selected result.
 *
 * @param optimizer Optimizer implementation.
 * @param dataset Optimization dataset.
 * @param evaluator Prompt evaluator.
 * @param config Optimizer configuration.
 * @return Full optimization outcome, including the selected prompt and metadata.
 */
fun OptimizableMetricPrompt.optimizePrompt(
    optimizer: Optimizer,
    dataset: OptimizationDataset,
    evaluator: PromptObjectEvaluator,
    config: OptimizerConfig = OptimizerConfig(),
): OptimizerOutcome {
    val outcome =
        optimizer.optimizePrompts(
            dataset = dataset,
            initialPrompts = listOf(optimizerPrompt()),
            evaluator = evaluator,
            config = config,
        )
    applyOptimizerOutcome(outcome)
    return outcome
}
