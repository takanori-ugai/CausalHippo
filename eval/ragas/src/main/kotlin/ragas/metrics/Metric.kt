package ragas.metrics

import ragas.embeddings.BaseRagasEmbedding
import ragas.llms.BaseRagasLlm
import ragas.model.MultiTurnSample
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig

/** Supported dataset granularity for metrics. */
enum class MetricType {
    SINGLE_TURN,
    MULTI_TURN,
}

/** Canonical metric result shape. */
enum class MetricOutputType {
    BINARY,
    DISCRETE,
    CONTINUOUS,
    RANKING,
}

/** Allowed sample column names that metrics may declare in [Metric.requiredColumns]. */
val VALID_COLUMNS: Set<String> =
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
        "reference_tool_calls",
        "reference_topics",
    )

/** Base contract implemented by all metrics. */
interface Metric {
    val name: String
    val requiredColumns: Map<MetricType, Set<String>>
    val outputType: MetricOutputType?

    /**
     * Initializes the metric before scoring begins.
     *
     * @param runConfig Runtime configuration for retries, timeouts, and concurrency.
     */
    suspend fun init(runConfig: RunConfig) {
        validateRequiredColumns()
    }

    /** Validates that declared required columns exist in [VALID_COLUMNS]. */
    fun validateRequiredColumns() {
        requiredColumns.forEach { (_, columns) ->
            columns.forEach { column ->
                require(column in VALID_COLUMNS) {
                    "Invalid column '$column' in metric '$name'. Allowed columns: $VALID_COLUMNS"
                }
            }
        }
    }
}

/** Marker for metrics that consume an LLM. */
interface MetricWithLlm {
    var llm: BaseRagasLlm?
}

/** Marker for metrics that consume an embeddings model. */
interface MetricWithEmbeddings {
    var embeddings: BaseRagasEmbedding?
}

/** Metric executable on [SingleTurnSample]. */
interface SingleTurnMetric : Metric {
    /**
     * Computes a score for one single-turn sample.
     *
     * @param sample Single-turn sample to score.
     */
    suspend fun singleTurnAscore(sample: SingleTurnSample): Any?
}

/** Metric executable on [MultiTurnSample]. */
interface MultiTurnMetric : Metric {
    /**
     * Computes a score for one multi-turn sample.
     *
     * @param sample Multi-turn sample to score.
     */
    suspend fun multiTurnAscore(sample: MultiTurnSample): Any?
}

abstract class BaseMetric(
    override val name: String,
    override val requiredColumns: Map<MetricType, Set<String>> = emptyMap(),
    override val outputType: MetricOutputType? = null,
) : Metric
