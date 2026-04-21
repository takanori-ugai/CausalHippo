package ragas.metrics

/**
 * Named metric score with optional reason metadata.
 *
 * @property value Metric score/result value.
 * @property reason Optional human-readable explanation of the result.
 * @property traces Optional structured trace/debug payload for the metric computation.
 */
data class MetricResult<T>(
    val value: T,
    val reason: String? = null,
    val traces: Map<String, Any?>? = null,
)
