package ragas.evaluation

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import ragas.embeddings.BaseRagasEmbedding
import ragas.llms.BaseRagasLlm
import ragas.metrics.Metric
import ragas.metrics.MetricWithEmbeddings
import ragas.metrics.MetricWithLlm
import ragas.metrics.MultiTurnMetric
import ragas.metrics.SingleTurnMetric
import ragas.metrics.defaults.defaultSingleTurnMetrics
import ragas.model.EvaluationDataset
import ragas.model.EvaluationResult
import ragas.model.MultiTurnSample
import ragas.model.Sample
import ragas.model.SingleTurnSample
import ragas.runtime.Executor
import ragas.runtime.RunConfig
import java.util.Collections
import java.util.IdentityHashMap
import java.util.WeakHashMap
import kotlin.reflect.KClass

private const val EVALUATION_DESC = "Evaluating"
private val metricLocks: MutableMap<Metric, Mutex> = Collections.synchronizedMap(WeakHashMap())

/**
 * Asynchronously evaluates a dataset with the selected metrics.
 *
 * @param dataset Dataset to evaluate or optimize.
 * @param metrics Metrics to run.
 * @param llm LLM dependency used during generation/evaluation.
 * @param embeddings Embedding dependency used during evaluation.
 * @param runConfig Runtime retry/concurrency configuration.
 * @param raiseExceptions Whether metric failures should be thrown.
 * @param batchSize Optional batch size for executor submissions.
 * @param callbacks Evaluation lifecycle callbacks.
 * @param columnMap Column remapping from user keys to canonical metric keys.
 * @param tokenUsageParser Optional parser for model token usage.
 * @param costParser Optional parser that converts token usage to cost.
 * @param executorObserver Optional observer invoked when executor is created.
 */
@Suppress("LongMethod", "TooGenericExceptionCaught")
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
): EvaluationResult {
    val remappedDataset = remapDataset(dataset, columnMap)
    val selectedMetrics = metrics ?: defaultMetricsForDataset(remappedDataset)
    require(selectedMetrics.isNotEmpty()) { "Provide at least one metric for evaluation." }

    validateRequiredColumns(remappedDataset, selectedMetrics)
    validateSupportedMetrics(remappedDataset, selectedMetrics)

    val llmChanged = mutableListOf<MetricWithLlm>()
    val embeddingsChanged = mutableListOf<MetricWithEmbeddings>()
    val acquiredLocks = mutableListOf<Mutex>()
    val usageLock = Any()
    val callbackDispatchLock = Any()
    var usagePromptTokens = 0
    var usageCompletionTokens = 0

    /**
     * Notifies registered callbacks about one evaluation event.
     *
     * @param event Evaluation or tracing event payload.
     */
    fun dispatchEvent(event: EvaluationEvent) {
        synchronized(callbackDispatchLock) {
            callbacks.forEach { callback -> callback.onEvent(event) }
        }
    }

    try {
        dispatchEvent(
            EvaluationEvent.RunStarted(
                sampleCount = remappedDataset.samples.size,
                metricNames = selectedMetrics.map { metric -> metric.name },
            ),
        )

        lockMetrics(selectedMetrics, acquiredLocks)

        val effectiveLlm =
            if (llm != null && (tokenUsageParser != null || costParser != null)) {
                TrackingRagasLlm(
                    delegate = llm,
                    tokenUsageParser = tokenUsageParser ?: { _, _ -> null },
                ) { usage ->
                    synchronized(usageLock) {
                        usagePromptTokens += usage.promptTokens
                        usageCompletionTokens += usage.completionTokens
                    }
                }
            } else {
                llm
            }

        selectedMetrics.forEach { metric ->
            if (metric is MetricWithLlm && metric.llm == null && effectiveLlm != null) {
                metric.llm = effectiveLlm
                llmChanged += metric
            }
            if (metric is MetricWithEmbeddings && metric.embeddings == null && embeddings != null) {
                metric.embeddings = embeddings
                embeddingsChanged += metric
            }
            metric.init(runConfig)
        }

        val executor =
            Executor(
                description = EVALUATION_DESC,
                raiseExceptions = raiseExceptions,
                runConfig = runConfig,
                batchSize = batchSize,
            )
        executorObserver?.invoke(executor)
        dispatchEvent(EvaluationEvent.ExecutorReady(executor))

        remappedDataset.samples.forEachIndexed { rowIndex, sample ->
            selectedMetrics.forEach { metric ->
                val metricName = metric.name
                when {
                    sample is SingleTurnSample && metric is SingleTurnMetric -> {
                        executor.submit(name = "$metricName-$rowIndex") {
                            val value = metric.singleTurnAscore(sample)
                            dispatchEvent(
                                EvaluationEvent.MetricComputed(
                                    rowIndex = rowIndex,
                                    metricName = metricName,
                                    value = value,
                                ),
                            )
                            value
                        }
                    }

                    sample is MultiTurnSample && metric is MultiTurnMetric -> {
                        executor.submit(name = "$metricName-$rowIndex") {
                            val value = metric.multiTurnAscore(sample)
                            dispatchEvent(
                                EvaluationEvent.MetricComputed(
                                    rowIndex = rowIndex,
                                    metricName = metricName,
                                    value = value,
                                ),
                            )
                            value
                        }
                    }
                }
            }
        }

        val flatResults = executor.aresults()
        val rowScores = mutableListOf<Map<String, Any?>>()
        var cursor = 0
        repeat(remappedDataset.samples.size) { rowIndex ->
            val row = linkedMapOf<String, Any?>()
            selectedMetrics.forEach { metric ->
                row[metric.name] = flatResults[cursor++]
            }
            rowScores += row
            dispatchEvent(
                EvaluationEvent.RowCompleted(
                    rowIndex = rowIndex,
                    scores = row,
                ),
            )
        }

        val result =
            EvaluationResult(
                scores = rowScores,
                dataset = remappedDataset,
            )

        if (tokenUsageParser != null || costParser != null) {
            val usage =
                TokenUsage(
                    promptTokens = usagePromptTokens,
                    completionTokens = usageCompletionTokens,
                )
            dispatchEvent(EvaluationEvent.TokenUsageComputed(usage))
            val cost = costParser?.invoke(usage)
            if (cost != null) {
                dispatchEvent(EvaluationEvent.CostComputed(cost))
            }
        }
        dispatchEvent(EvaluationEvent.RunCompleted(result))
        return result
    } catch (error: Throwable) {
        synchronized(callbackDispatchLock) {
            callbacks.forEach { callback ->
                runCatching { callback.onEvent(EvaluationEvent.RunFailed.fromThrowable(error)) }
            }
        }
        throw error
    } finally {
        llmChanged.forEach { metric -> metric.llm = null }
        embeddingsChanged.forEach { metric -> metric.embeddings = null }
        acquiredLocks.asReversed().forEach { lock -> lock.unlock() }
    }
}

/**
 * Synchronous wrapper around asynchronous evaluation.
 *
 * @param dataset Dataset to evaluate or optimize.
 * @param metrics Metrics to run.
 * @param llm LLM dependency used during generation/evaluation.
 * @param embeddings Embedding dependency used during evaluation.
 * @param runConfig Runtime retry/concurrency configuration.
 * @param raiseExceptions Whether metric failures should be thrown.
 * @param batchSize Optional batch size for executor submissions.
 * @param callbacks Evaluation lifecycle callbacks.
 * @param columnMap Column remapping from user keys to canonical metric keys.
 * @param tokenUsageParser Optional parser for model token usage.
 * @param costParser Optional parser that converts token usage to cost.
 * @param executorObserver Optional observer invoked when executor is created.
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
    runBlocking {
        aevaluate(
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
    }

private fun remapDataset(
    dataset: EvaluationDataset<out Sample>,
    columnMap: Map<String, String>,
): EvaluationDataset<out Sample> {
    if (columnMap.isEmpty()) {
        return dataset
    }
    val normalizedEntries =
        columnMap.entries.map { (target, source) ->
            normalizeColumnName(target) to normalizeColumnName(source)
        }
    val duplicateTargets =
        normalizedEntries
            .groupingBy { (target, _) -> target }
            .eachCount()
            .filterValues { count -> count > 1 }
            .keys
    require(duplicateTargets.isEmpty()) {
        "Duplicate columnMap targets after normalization: ${duplicateTargets.sorted().joinToString(", ")}"
    }
    val normalizedMap = normalizedEntries.toMap()
    validateColumnMapKeys(dataset.getSampleType(), normalizedMap)
    val remappedSamples =
        dataset.samples.map { sample ->
            when (sample) {
                is SingleTurnSample -> remapSingleTurnSample(sample, normalizedMap)
                is MultiTurnSample -> remapMultiTurnSample(sample, normalizedMap)
            }
        }
    return EvaluationDataset(remappedSamples)
}

private fun remapSingleTurnSample(
    sample: SingleTurnSample,
    columnMap: Map<String, String>,
): SingleTurnSample {
    val values = mutableMapOf<String, Any?>()
    SINGLE_TURN_COLUMNS.forEach { column -> values[column] = readSingleTurnField(sample, column) }
    columnMap.forEach { (target, source) ->
        val targetValue = values[target]
        val sourceValue = values[source]
        if (
            isNullOrEmptyValue(targetValue) &&
            !isNullOrEmptyValue(sourceValue) &&
            isCompatibleSingleTurnValue(target, sourceValue)
        ) {
            values[target] = sourceValue
        }
    }
    return SingleTurnSample(
        userInput = asString(values["user_input"]),
        retrievedContexts = asStringList(values["retrieved_contexts"]),
        referenceContexts = asStringList(values["reference_contexts"]),
        retrievedContextIds = asStringList(values["retrieved_context_ids"]),
        referenceContextIds = asStringList(values["reference_context_ids"]),
        response = asString(values["response"]),
        multiResponses = asStringList(values["multi_responses"]),
        reference = asString(values["reference"]),
        rubrics = asStringMap(values["rubrics"]),
        personaName = asString(values["persona_name"]),
        queryStyle = asString(values["query_style"]),
        queryLength = asString(values["query_length"]),
    )
}

private fun remapMultiTurnSample(
    sample: MultiTurnSample,
    columnMap: Map<String, String>,
): MultiTurnSample {
    val values = mutableMapOf<String, Any?>()
    MULTI_TURN_COLUMNS.forEach { column -> values[column] = readMultiTurnField(sample, column) }
    columnMap.forEach { (target, source) ->
        val targetValue = values[target]
        val sourceValue = values[source]
        if (
            isNullOrEmptyValue(targetValue) &&
            !isNullOrEmptyValue(sourceValue) &&
            isCompatibleMultiTurnValue(target, sourceValue)
        ) {
            values[target] = sourceValue
        }
    }
    return MultiTurnSample(
        userInput = asConversationMessageList(values["user_input"]) ?: sample.userInput,
        reference = asString(values["reference"]),
        referenceToolCalls = asToolCallList(values["reference_tool_calls"]),
        rubrics = asStringMap(values["rubrics"]),
        referenceTopics = asStringList(values["reference_topics"]),
    )
}

private fun readSingleTurnField(
    sample: SingleTurnSample,
    column: String,
): Any? =
    when (column) {
        "user_input" -> sample.userInput
        "retrieved_contexts" -> sample.retrievedContexts
        "reference_contexts" -> sample.referenceContexts
        "retrieved_context_ids" -> sample.retrievedContextIds
        "reference_context_ids" -> sample.referenceContextIds
        "response" -> sample.response
        "multi_responses" -> sample.multiResponses
        "reference" -> sample.reference
        "rubrics" -> sample.rubrics
        "persona_name" -> sample.personaName
        "query_style" -> sample.queryStyle
        "query_length" -> sample.queryLength
        else -> null
    }

private fun readMultiTurnField(
    sample: MultiTurnSample,
    column: String,
): Any? =
    when (column) {
        "user_input" -> sample.userInput
        "reference" -> sample.reference
        "reference_tool_calls" -> sample.referenceToolCalls
        "rubrics" -> sample.rubrics
        "reference_topics" -> sample.referenceTopics
        else -> null
    }

private fun normalizeColumnName(name: String): String {
    val normalized = name.trim().lowercase()
    return COLUMN_ALIASES[normalized] ?: normalized
}

private fun validateColumnMapKeys(
    sampleType: KClass<out Sample>?,
    columnMap: Map<String, String>,
) {
    val supportedColumns =
        when (sampleType) {
            SingleTurnSample::class -> SINGLE_TURN_COLUMNS
            MultiTurnSample::class -> MULTI_TURN_COLUMNS
            else -> return
        }
    val sampleTypeName = sampleType.simpleName ?: "Sample"
    val supportedList = supportedColumns.sorted().joinToString(", ")
    columnMap.forEach { (target, source) ->
        require(target in supportedColumns) {
            "Unsupported columnMap target '$target' for $sampleTypeName. Supported columns: $supportedList"
        }
        require(source in supportedColumns) {
            "Unsupported columnMap source '$source' for $sampleTypeName. Supported columns: $supportedList"
        }
    }
}

private fun isNullOrEmptyValue(value: Any?): Boolean =
    when (value) {
        null -> true
        is String -> value.isBlank()
        is Collection<*> -> value.isEmpty()
        is Map<*, *> -> value.isEmpty()
        else -> false
    }

private fun isCompatibleSingleTurnValue(
    column: String,
    value: Any?,
): Boolean =
    when (column) {
        "user_input", "response", "reference", "persona_name", "query_style", "query_length" -> {
            value is String
        }

        "retrieved_contexts", "reference_contexts", "retrieved_context_ids", "reference_context_ids", "multi_responses" -> {
            isStringList(
                value,
            )
        }

        "rubrics" -> {
            isStringMap(value)
        }

        else -> {
            false
        }
    }

private fun isCompatibleMultiTurnValue(
    column: String,
    value: Any?,
): Boolean =
    when (column) {
        "user_input" -> isConversationMessageList(value)
        "reference" -> value is String
        "reference_tool_calls" -> isToolCallList(value)
        "rubrics" -> isStringMap(value)
        "reference_topics" -> isStringList(value)
        else -> false
    }

private fun asString(value: Any?): String? = value as? String

private fun asStringList(value: Any?): List<String>? {
    val list = value as? List<*> ?: return null
    if (!list.all { it is String }) {
        return null
    }
    return list.map { item -> item as String }
}

private fun asConversationMessageList(value: Any?): List<ragas.model.ConversationMessage>? {
    val list = value as? List<*> ?: return null
    if (!list.all { it is ragas.model.ConversationMessage }) {
        return null
    }
    return list.map { item -> item as ragas.model.ConversationMessage }
}

private fun asToolCallList(value: Any?): List<ragas.model.ToolCall>? {
    val list = value as? List<*> ?: return null
    if (!list.all { it is ragas.model.ToolCall }) {
        return null
    }
    return list.map { item -> item as ragas.model.ToolCall }
}

private fun asStringMap(value: Any?): Map<String, String>? {
    val map = value as? Map<*, *> ?: return null
    if (!map.entries.all { entry -> entry.key is String && entry.value is String }) {
        return null
    }
    return map.entries.associate { entry -> entry.key as String to entry.value as String }
}

private fun isStringList(value: Any?): Boolean = asStringList(value) != null

private fun isConversationMessageList(value: Any?): Boolean = asConversationMessageList(value) != null

private fun isToolCallList(value: Any?): Boolean = asToolCallList(value) != null

private fun isStringMap(value: Any?): Boolean = asStringMap(value) != null

private val SINGLE_TURN_COLUMNS =
    setOf(
        "user_input",
        "retrieved_contexts",
        "reference_contexts",
        "retrieved_context_ids",
        "reference_context_ids",
        "response",
        "multi_responses",
        "reference",
        "rubrics",
        "persona_name",
        "query_style",
        "query_length",
    )

private val MULTI_TURN_COLUMNS =
    setOf(
        "user_input",
        "reference",
        "reference_tool_calls",
        "rubrics",
        "reference_topics",
    )

private val COLUMN_ALIASES =
    mapOf(
        "question" to "user_input",
        "query" to "user_input",
        "answer" to "response",
        "ground_truth" to "reference",
        "groundtruth" to "reference",
        "contexts" to "retrieved_contexts",
    )

private fun validateRequiredColumns(
    dataset: EvaluationDataset<out Sample>,
    metrics: List<Metric>,
) {
    val datasetFeatures = dataset.features()
    metrics.forEach { metric ->
        metric.requiredColumns.values.flatten().forEach { column ->
            require(column in datasetFeatures) {
                "Metric '${metric.name}' requires column '$column', but dataset has $datasetFeatures"
            }
        }
    }
}

private suspend fun lockMetrics(
    metrics: List<Metric>,
    acquiredLocks: MutableList<Mutex>,
) {
    distinctByIdentity(metrics).sortedBy { metric -> System.identityHashCode(metric) }.forEach { metric ->
        val lock =
            synchronized(metricLocks) {
                metricLocks.getOrPut(metric) { Mutex() }
            }
        lock.lock()
        acquiredLocks += lock
    }
}

private fun distinctByIdentity(metrics: List<Metric>): List<Metric> {
    val seen = IdentityHashMap<Metric, Boolean>(metrics.size)
    return metrics.filter { metric -> seen.put(metric, true) == null }
}

private fun defaultMetricsForDataset(dataset: EvaluationDataset<out Sample>): List<Metric> =
    when (dataset.getSampleType()) {
        SingleTurnSample::class -> {
            defaultSingleTurnMetrics()
        }

        MultiTurnSample::class -> {
            error("Default metrics are currently implemented only for single-turn datasets.")
        }

        else -> {
            error("Unsupported sample type: ${dataset.getSampleType()}")
        }
    }

private fun validateSupportedMetrics(
    dataset: EvaluationDataset<out Sample>,
    metrics: List<Metric>,
) {
    val sampleType = dataset.getSampleType() ?: return
    when (sampleType) {
        SingleTurnSample::class -> {
            require(metrics.all { metric -> metric is SingleTurnMetric }) {
                "Single-turn datasets support only SingleTurnMetric instances."
            }
        }

        MultiTurnSample::class -> {
            require(metrics.all { metric -> metric is MultiTurnMetric }) {
                "Multi-turn datasets support only MultiTurnMetric instances."
            }
        }

        else -> {
            error("Unsupported sample type: $sampleType")
        }
    }
}
