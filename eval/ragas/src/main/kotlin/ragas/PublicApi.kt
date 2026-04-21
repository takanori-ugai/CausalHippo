@file:JvmName("Ragas")

package ragas

import ragas.backends.BACKEND_REGISTRY
import ragas.backends.BackendRegistry
import ragas.cache.CacheBackend
import ragas.embeddings.BaseRagasEmbedding
import ragas.embeddings.CachedRagasEmbedding
import ragas.evaluation.CostParser
import ragas.evaluation.EvaluationCallback
import ragas.evaluation.TokenUsageParser
import ragas.llms.BaseRagasLlm
import ragas.llms.CachedRagasLlm
import ragas.metrics.Metric
import ragas.metrics.collections.agentToolCallTier2Metrics
import ragas.metrics.collections.answerQualityTier3Metrics
import ragas.metrics.collections.retrievalGroundednessTier1Metrics
import ragas.metrics.collections.tier4CollectionMetrics
import ragas.metrics.defaults.defaultSingleTurnMetrics
import ragas.model.EvaluationDataset
import ragas.model.EvaluationResult
import ragas.model.Sample
import ragas.optimizers.DspyOptimizer
import ragas.optimizers.GeneticOptimizer
import ragas.optimizers.Optimizer
import ragas.runtime.Executor
import ragas.runtime.RunConfig
import ragas.evaluation.aevaluate as aevaluateInternal
import ragas.evaluation.evaluate as evaluateInternal

/**
 * Library version for the `ragas-kotlin` artifact.
 */
const val VERSION: String = "0.0.1"

/**
 * Synchronously evaluates a dataset with one or more metrics.
 *
 * This is the blocking wrapper around [aevaluate]. If [metrics] is null, default
 * single-turn metrics are selected by the evaluation engine.
 */
fun evaluate(
    dataset: EvaluationDataset<out Sample>,
    metrics: List<Metric>? = null,
    llm: BaseRagasLlm? = null,
    embeddings: BaseRagasEmbedding? = null,
    runConfig: RunConfig = RunConfig(),
    raiseExceptions: Boolean = false,
    batchSize: Int? = null,
    callbacks: List<EvaluationCallback> = emptyList(),
    columnMap: Map<String, String> = emptyMap(),
    tokenUsageParser: TokenUsageParser? = null,
    costParser: CostParser? = null,
    executorObserver: ((Executor) -> Unit)? = null,
): EvaluationResult =
    evaluateInternal(
        dataset = dataset,
        metrics = metrics,
        llm = llm,
        embeddings = embeddings,
        runConfig = runConfig,
        raiseExceptions = raiseExceptions,
        batchSize = batchSize,
        callbacks = callbacks,
        columnMap = columnMap,
        tokenUsageParser = tokenUsageParser,
        costParser = costParser,
        executorObserver = executorObserver,
    )

/**
 * Suspends while evaluating a dataset with one or more metrics.
 */
suspend fun aevaluate(
    dataset: EvaluationDataset<out Sample>,
    metrics: List<Metric>? = null,
    llm: BaseRagasLlm? = null,
    embeddings: BaseRagasEmbedding? = null,
    runConfig: RunConfig = RunConfig(),
    raiseExceptions: Boolean = false,
    batchSize: Int? = null,
    callbacks: List<EvaluationCallback> = emptyList(),
    columnMap: Map<String, String> = emptyMap(),
    tokenUsageParser: TokenUsageParser? = null,
    costParser: CostParser? = null,
    executorObserver: ((Executor) -> Unit)? = null,
): EvaluationResult =
    aevaluateInternal(
        dataset = dataset,
        metrics = metrics,
        llm = llm,
        embeddings = embeddings,
        runConfig = runConfig,
        raiseExceptions = raiseExceptions,
        batchSize = batchSize,
        callbacks = callbacks,
        columnMap = columnMap,
        tokenUsageParser = tokenUsageParser,
        costParser = costParser,
        executorObserver = executorObserver,
    )

/**
 * Returns the default single-turn metric set.
 */
fun defaultMetrics(): List<Metric> = defaultSingleTurnMetrics()

/** Returns the Tier-1 retrieval groundedness metric set. */
fun tier1Metrics(): List<Metric> = retrievalGroundednessTier1Metrics()

/** Returns the Tier-2 agent/tool-call metric set. */
fun tier2Metrics(): List<Metric> = agentToolCallTier2Metrics()

/** Returns the Tier-3 answer-quality metric set. */
fun tier3Metrics(): List<Metric> = answerQualityTier3Metrics()

/** Returns the Tier-4 advanced metric set. */
fun tier4Metrics(): List<Metric> = tier4CollectionMetrics()

/**
 * Wraps an LLM with a cache layer.
 */
fun withCache(
    llm: BaseRagasLlm,
    cache: CacheBackend,
): BaseRagasLlm = CachedRagasLlm(llm, cache)

/**
 * Wraps an embeddings provider with a cache layer.
 */
fun withCache(
    embedding: BaseRagasEmbedding,
    cache: CacheBackend,
): BaseRagasEmbedding = CachedRagasEmbedding(embedding, cache)

/** Returns the process-wide backend registry instance. */
fun backendRegistry(): BackendRegistry = BACKEND_REGISTRY

/** Creates a genetic optimizer implementation. */
fun geneticOptimizer(): Optimizer = GeneticOptimizer()

/**
 * Creates a DSPy-style optimizer.
 *
 * When [cache] is provided, prompt-scoring results are memoized.
 */
fun dspyOptimizer(cache: CacheBackend? = null): Optimizer =
    if (cache == null) {
        DspyOptimizer()
    } else {
        DspyOptimizer(cache = cache)
    }
