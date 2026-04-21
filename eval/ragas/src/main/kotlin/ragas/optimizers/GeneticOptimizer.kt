package ragas.optimizers

import ragas.prompt.PromptContentPart
import kotlin.math.max
import kotlin.random.Random

/**
 * Evolutionary prompt optimizer based on mutation and fitness scoring.
 */
class GeneticOptimizer : Optimizer {
    /**
     * Optimizes a prompt on a dataset and returns the best discovered outcome.
     */
    override fun optimizePrompts(
        dataset: OptimizationDataset,
        initialPrompts: List<OptimizerPrompt>,
        evaluator: PromptObjectEvaluator,
        config: OptimizerConfig,
    ): OptimizerOutcome {
        require(initialPrompts.isNotEmpty()) { "initialPrompts cannot be empty" }
        require(config.populationSize > 0) { "populationSize must be > 0" }
        require(config.iterations > 0) { "iterations must be > 0" }

        val random = Random(config.seed)

        var population =
            seedPopulation(
                initialPrompts = initialPrompts,
                populationSize = config.populationSize,
                random = random,
            )

        repeat(config.iterations) {
            val scored = population.map { prompt -> prompt to evaluator.score(prompt, dataset) }
            val sorted = scored.sortedByDescending { (_, score) -> score }
            val eliteCount = max(1, config.populationSize / 2)
            val elite = sorted.take(eliteCount).map { (prompt) -> prompt }

            val next = elite.toMutableList()
            while (next.size < config.populationSize) {
                val p1 = elite.random(random)
                val p2 = elite.random(random)
                val crossed = crossoverPrompt(p1, p2, random)
                val mutated = maybeMutatePrompt(crossed, random, config.mutationProbability)
                next += mutated
            }
            population = next
        }

        val best = population.maxBy { prompt -> evaluator.score(prompt, dataset) }
        return OptimizerOutcome(
            optimizedPrompt = best,
            metadata = mapOf("optimizer" to "genetic"),
        )
    }

    private fun seedPopulation(
        initialPrompts: List<OptimizerPrompt>,
        populationSize: Int,
        random: Random,
    ): List<OptimizerPrompt> {
        val population = initialPrompts.toMutableList()
        while (population.size < populationSize) {
            val source = initialPrompts.random(random)
            population += makeVariant(source, population.size + 1)
        }
        return population.take(populationSize)
    }

    private fun makeVariant(
        source: OptimizerPrompt,
        variantIndex: Int,
    ): OptimizerPrompt =
        when (source) {
            is OptimizerPrompt.Text -> {
                OptimizerPrompt.Text("${source.value}\n(variant $variantIndex)")
            }

            is OptimizerPrompt.MultiModal -> {
                OptimizerPrompt.MultiModal(
                    source.content + PromptContentPart.Text("(variant $variantIndex)"),
                )
            }
        }

    private fun crossoverPrompt(
        parent1: OptimizerPrompt,
        parent2: OptimizerPrompt,
        random: Random,
    ): OptimizerPrompt =
        when {
            parent1 is OptimizerPrompt.Text && parent2 is OptimizerPrompt.Text -> {
                OptimizerPrompt.Text(crossoverText(parent1.value, parent2.value, random))
            }

            parent1 is OptimizerPrompt.MultiModal && parent2 is OptimizerPrompt.MultiModal -> {
                crossoverMultiModal(parent1, parent2, random)
            }

            random.nextBoolean() -> {
                parent1
            }

            else -> {
                parent2
            }
        }

    private fun crossoverText(
        parent1: String,
        parent2: String,
        random: Random,
    ): String {
        if (parent1.isBlank()) return parent2
        if (parent2.isBlank()) return parent1
        val split1 = parent1.trim().split("\\s+".toRegex())
        val split2 = parent2.trim().split("\\s+".toRegex())

        val cut1 = random.nextInt(split1.size)
        val cut2 = random.nextInt(split2.size)
        val child = split1.take(cut1) + split2.drop(cut2)
        return child.joinToString(" ").ifBlank { parent1 }
    }

    private fun crossoverMultiModal(
        parent1: OptimizerPrompt.MultiModal,
        parent2: OptimizerPrompt.MultiModal,
        random: Random,
    ): OptimizerPrompt.MultiModal {
        val content1 = parent1.content
        val content2 = parent2.content
        if (content1.isEmpty()) return parent2
        if (content2.isEmpty()) return parent1

        val cut1 = random.nextInt(content1.size)
        val cut2 = random.nextInt(content2.size)
        val child = content1.take(cut1) + content2.drop(cut2)
        return OptimizerPrompt.MultiModal(child.ifEmpty { content1 })
    }

    private fun maybeMutatePrompt(
        prompt: OptimizerPrompt,
        random: Random,
        mutationProbability: Double,
    ): OptimizerPrompt {
        if (random.nextDouble() >= mutationProbability) {
            return prompt
        }

        return when (prompt) {
            is OptimizerPrompt.Text -> mutateTextPrompt(prompt, random)
            is OptimizerPrompt.MultiModal -> mutateMultiModalPrompt(prompt, random)
        }
    }

    private fun mutateTextPrompt(
        prompt: OptimizerPrompt.Text,
        random: Random,
    ): OptimizerPrompt.Text {
        val mutations =
            listOf(
                "${prompt.value}\nBe concise.",
                "${prompt.value}\nExplain your reasoning briefly.",
                "${prompt.value}\nReturn only the final value.",
            )
        return OptimizerPrompt.Text(mutations.random(random))
    }

    private fun mutateMultiModalPrompt(
        prompt: OptimizerPrompt.MultiModal,
        random: Random,
    ): OptimizerPrompt.MultiModal {
        val guidance =
            listOf(
                "Be concise.",
                "Explain your reasoning briefly.",
                "Return only the final value.",
            ).random(random)
        return OptimizerPrompt.MultiModal(
            prompt.content + PromptContentPart.Text(guidance),
        )
    }
}
