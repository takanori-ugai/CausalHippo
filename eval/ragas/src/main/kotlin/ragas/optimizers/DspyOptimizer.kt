package ragas.optimizers

import ragas.cache.CacheBackend
import ragas.cache.stableCacheKey

/**
 * Optimizer that uses an optional DSPy adapter to propose prompt candidates.
 */
class DspyOptimizer : Optimizer {
    constructor() : this(
        adapter = DspyAdapterLoader.loadFirstOrNull() ?: HeuristicDspyAdapter(),
        cache = null,
    )

    constructor(cache: CacheBackend) : this(
        adapter = DspyAdapterLoader.loadFirstOrNull() ?: HeuristicDspyAdapter(),
        cache = cache,
    )

    constructor(
        adapter: DspyAdapter,
        cache: CacheBackend? = null,
    ) {
        this.adapter = adapter
        this.cache = cache
    }

    private val adapter: DspyAdapter
    private val cache: CacheBackend?

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
        require(config.iterations > 0) { "iterations must be > 0" }

        val datasetSignature = cache?.let { signatureForDataset(dataset) }
        val promptSignatures = mutableMapOf<OptimizerPrompt, String>()

        fun signature(prompt: OptimizerPrompt): String = promptSignatures.getOrPut(prompt) { signatureForPrompt(prompt) }

        fun bestWithScore(prompts: List<OptimizerPrompt>): Pair<OptimizerPrompt, Double> {
            var currentBest = prompts.first()
            var currentBestScore = scoreWithCache(currentBest, signature(currentBest), datasetSignature, dataset, evaluator)
            prompts.drop(1).forEach { prompt ->
                val score = scoreWithCache(prompt, signature(prompt), datasetSignature, dataset, evaluator)
                if (score > currentBestScore) {
                    currentBest = prompt
                    currentBestScore = score
                }
            }
            return currentBest to currentBestScore
        }

        var (best, bestScore) = bestWithScore(initialPrompts)

        repeat(config.iterations) { iteration ->
            val context =
                DspyCompileContext(
                    iteration = iteration + 1,
                    dataset = dataset,
                    currentBestPrompt = best,
                )
            val candidates = (listOf(best) + adapter.proposeCandidates(context)).distinctBy(::signature)
            val (candidateBest, candidateScore) = bestWithScore(candidates)
            if (candidateScore >= bestScore) {
                best = candidateBest
                bestScore = candidateScore
            }
        }

        return OptimizerOutcome(
            optimizedPrompt = best,
            metadata =
                mapOf(
                    "optimizer" to "dspy",
                    "adapter" to adapter.javaClass.simpleName,
                ),
        )
    }

    private fun scoreWithCache(
        prompt: OptimizerPrompt,
        promptSignature: String,
        datasetSignature: String?,
        dataset: OptimizationDataset,
        evaluator: PromptObjectEvaluator,
    ): Double {
        val cacheBackend = cache ?: return evaluator.score(prompt, dataset)
        val stableDatasetSignature = datasetSignature ?: signatureForDataset(dataset)
        val key =
            stableCacheKey(
                buildString {
                    append("optimizer|dspy|score|")
                    append(stableDatasetSignature)
                    append("|")
                    append(promptSignature)
                },
            )
        val cached = cacheBackend.get(key)
        if (cached is Number) {
            return cached.toDouble()
        }

        val computed = evaluator.score(prompt, dataset)
        cacheBackend.put(key, computed)
        return computed
    }

    private fun signatureForDataset(dataset: OptimizationDataset): String =
        buildString {
            append("metric:")
            append(dataset.metricName.length)
            append(":")
            append(dataset.metricName)
            append("|examples:")
            append(dataset.examples.size)
            dataset.examples.forEach { ex ->
                append("|in:")
                append(ex.promptInput.size)
                ex.promptInput
                    .toSortedMap()
                    .forEach { (k, v) ->
                        append("|k:")
                        append(k.length)
                        append(":")
                        append(k)
                        append("|v:")
                        append(v.length)
                        append(":")
                        append(v)
                    }
                append("|out:")
                append(ex.expectedOutput.length)
                append(":")
                append(ex.expectedOutput)
            }
        }

    private fun signatureForPrompt(prompt: OptimizerPrompt): String =
        when (prompt) {
            is OptimizerPrompt.Text -> {
                "text:${prompt.value}"
            }

            is OptimizerPrompt.MultiModal -> {
                "multimodal:${prompt.asTextPrompt()}"
            }
        }
}
