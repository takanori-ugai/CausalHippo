package ragas.integrations.tracing

/** Observer that receives tracing events produced during a run. */
fun interface TraceObserver {
    /**
     * Handles a single trace event.
     *
     * @param event Event to consume.
     */
    fun onEvent(event: TraceEvent)
}

/** In-memory [TraceObserver] implementation that stores all received events. */
class InMemoryTraceObserver : TraceObserver {
    private val _events = mutableListOf<TraceEvent>()

    /** Snapshot of currently captured events in insertion order. */
    val events: List<TraceEvent>
        get() = _events.toList()

    /**
     * Adds [event] to the in-memory event buffer.
     *
     * @param event Event to append.
     */
    override fun onEvent(event: TraceEvent) {
        _events += event
    }
}
