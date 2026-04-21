package ragas.integrations.tracing

private const val DEFAULT_TRACE_FRAMEWORK = "ragas-kotlin"
private const val DEFAULT_TRACE_RUN_NAME = "run"

/** Base event emitted by evaluation/tracing integrations. */
sealed interface TraceEvent {
    /** Unique identifier of the run this event belongs to. */
    val runId: String

    /** Source framework that emitted the event (for example `ragas-kotlin`). */
    val framework: String

    /** Human-readable run name used for display and reporting. */
    val runName: String

    /** Event timestamp in epoch milliseconds. */
    val timestampMs: Long
}

/**
 * Event emitted when a run starts.
 *
 * @property runId Unique run identifier.
 * @property framework Source framework that emitted the event.
 * @property runName Human-readable run name.
 * @property timestampMs Start timestamp in epoch milliseconds.
 * @property tags User-defined tags attached to the run.
 * @property metadata Additional run metadata.
 */
data class RunStarted(
    override val runId: String,
    override val framework: String = DEFAULT_TRACE_FRAMEWORK,
    override val runName: String = DEFAULT_TRACE_RUN_NAME,
    override val timestampMs: Long = System.currentTimeMillis(),
    val tags: Map<String, String> = emptyMap(),
    val metadata: Map<String, String> = emptyMap(),
) : TraceEvent

/**
 * Event emitted when metric scores are produced for a dataset row.
 *
 * @property runId Unique run identifier.
 * @property framework Source framework that emitted the event.
 * @property runName Human-readable run name.
 * @property timestampMs Event timestamp in epoch milliseconds.
 * @property rowIndex Zero-based index of the evaluated row.
 * @property scores Metric values recorded for the row.
 */
data class MetricRowLogged(
    override val runId: String,
    override val framework: String = DEFAULT_TRACE_FRAMEWORK,
    override val runName: String = DEFAULT_TRACE_RUN_NAME,
    override val timestampMs: Long = System.currentTimeMillis(),
    val rowIndex: Int,
    val scores: Map<String, Any?>,
) : TraceEvent

/**
 * Event emitted when a run finishes successfully.
 *
 * @property runId Unique run identifier.
 * @property framework Source framework that emitted the event.
 * @property runName Human-readable run name.
 * @property timestampMs Completion timestamp in epoch milliseconds.
 * @property durationMs Run duration in milliseconds.
 * @property aggregateMetrics Final aggregate metric values for the run.
 */
data class RunCompleted(
    override val runId: String,
    override val framework: String = DEFAULT_TRACE_FRAMEWORK,
    override val runName: String = DEFAULT_TRACE_RUN_NAME,
    override val timestampMs: Long = System.currentTimeMillis(),
    val durationMs: Long,
    val aggregateMetrics: Map<String, Double>,
) : TraceEvent

/**
 * Event emitted when a run terminates with an error.
 *
 * @property runId Unique run identifier.
 * @property framework Source framework that emitted the event.
 * @property runName Human-readable run name.
 * @property timestampMs Failure timestamp in epoch milliseconds.
 * @property durationMs Elapsed run duration before failure, in milliseconds.
 * @property errorType Short error category or exception type.
 * @property errorMessage Human-readable error message.
 */
data class RunFailed(
    override val runId: String,
    override val framework: String = DEFAULT_TRACE_FRAMEWORK,
    override val runName: String = DEFAULT_TRACE_RUN_NAME,
    override val timestampMs: Long = System.currentTimeMillis(),
    val durationMs: Long,
    val errorType: String,
    val errorMessage: String,
) : TraceEvent
