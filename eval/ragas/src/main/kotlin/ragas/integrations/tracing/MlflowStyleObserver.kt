package ragas.integrations.tracing

/**
 * Simplified MLflow-style run record derived from trace events.
 *
 * @property runId Unique run identifier.
 * @property runName Human-readable run name.
 * @property params Run parameters and tags captured at start.
 * @property metrics Final metric values associated with the run.
 * @property status Run lifecycle status (for example `RUNNING`, `FINISHED`, `FAILED`).
 */
data class MlflowRun(
    val runId: String,
    val runName: String,
    val params: Map<String, String>,
    val metrics: Map<String, Double> = emptyMap(),
    val status: String = "RUNNING",
)

/** Converts [TraceEvent] streams into MLflow-style run snapshots. */
class MlflowStyleObserver : TraceObserver {
    private val runs = mutableMapOf<String, MlflowRun>()

    /**
     * Updates the run store using an incoming trace [event].
     *
     * @param event Event that mutates run state.
     */
    override fun onEvent(event: TraceEvent) {
        when (event) {
            is RunStarted -> {
                runs[event.runId] =
                    MlflowRun(
                        runId = event.runId,
                        runName = event.runName,
                        params = (event.tags + event.metadata).toMap(),
                    )
            }

            is MetricRowLogged -> {
                // MLflow-style row logs are usually reduced into final metrics
            }

            is RunCompleted -> {
                val run = runs[event.runId] ?: return
                runs[event.runId] =
                    run.copy(
                        metrics = run.metrics + event.aggregateMetrics,
                        status = "FINISHED",
                    )
            }

            is RunFailed -> {
                val run = runs[event.runId] ?: return
                runs[event.runId] = run.copy(status = "FAILED")
            }
        }
    }

    /**
     * Returns a defensive copy of the run identified by [runId].
     *
     * @param runId Run identifier.
     * @return Run snapshot, or null when no run exists for the identifier.
     */
    fun getRun(runId: String): MlflowRun? =
        runs[runId]?.let { run ->
            run.copy(params = run.params.toMap(), metrics = run.metrics.toMap())
        }

    /**
     * Returns defensive copies of all tracked runs.
     *
     * @return List of run snapshots in insertion order.
     */
    fun allRuns(): List<MlflowRun> =
        runs.values.map { run ->
            run.copy(params = run.params.toMap(), metrics = run.metrics.toMap())
        }
}
