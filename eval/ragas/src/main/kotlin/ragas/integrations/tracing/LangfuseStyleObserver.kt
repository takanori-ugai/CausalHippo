package ragas.integrations.tracing

/**
 * Simplified Langfuse-style trace span derived from run-level events.
 *
 * @property traceId Unique trace identifier (mapped from run ID).
 * @property name Display name for the trace.
 * @property startTimeMs Start timestamp in epoch milliseconds.
 * @property endTimeMs End timestamp in epoch milliseconds, if completed or failed.
 * @property tags User-defined trace tags.
 * @property metadata Additional trace metadata.
 */
data class LangfuseSpan(
    val traceId: String,
    val name: String,
    val startTimeMs: Long,
    var endTimeMs: Long? = null,
    val tags: Map<String, String> = emptyMap(),
    val metadata: Map<String, String> = emptyMap(),
)

/** Converts [TraceEvent] streams into Langfuse-style run spans. */
class LangfuseStyleObserver : TraceObserver {
    private val traces = mutableMapOf<String, LangfuseSpan>()

    /**
     * Updates the span store using an incoming trace [event].
     *
     * @param event Event that mutates span state.
     */
    override fun onEvent(event: TraceEvent) {
        when (event) {
            is RunStarted -> {
                traces[event.runId] =
                    LangfuseSpan(
                        traceId = event.runId,
                        name = event.runName,
                        startTimeMs = event.timestampMs,
                        tags = event.tags,
                        metadata = event.metadata,
                    )
            }

            is RunCompleted -> {
                traces[event.runId]?.endTimeMs = event.timestampMs
            }

            is RunFailed -> {
                traces[event.runId]?.endTimeMs = event.timestampMs
            }

            is MetricRowLogged -> {
                // row-level spans can be derived externally; we keep trace-level state only
            }
        }
    }

    /**
     * Returns the span tracked for [runId], if present.
     *
     * @param runId Run identifier used as the trace key.
     * @return Matching span, or null when no trace exists.
     */
    fun getTrace(runId: String): LangfuseSpan? = traces[runId]

    /**
     * Returns all tracked spans.
     *
     * @return Current list of spans.
     */
    fun allTraces(): List<LangfuseSpan> = traces.values.toList()
}
