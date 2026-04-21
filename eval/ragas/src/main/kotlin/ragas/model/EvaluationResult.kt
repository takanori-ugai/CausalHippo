package ragas.model

/**
 * Output of an evaluation run.
 *
 * [scores] holds per-row metric values keyed by metric name.
 *
 * @property scores Per-row metric scores.
 * @property dataset Dataset used for evaluation.
 * @property binaryColumns Columns treated as binary.
 * @property traces Optional trace payloads exported by integrations.
 */
data class EvaluationResult(
    val scores: List<Map<String, Any?>>,
    val dataset: EvaluationDataset<out Sample>,
    val binaryColumns: Set<String> = emptySet(),
    val traces: Map<String, Any?> = emptyMap(),
) {
    /**
     * Returns all row values for a metric.
     *
     * @param metricName Metric name key.
     */
    fun metricValues(metricName: String): List<Any?> = scores.map { row -> row[metricName] }

    /**
     * Returns the arithmetic mean of numeric values for a metric, or `null` if absent.
     *
     * @param metricName Metric name key.
     */
    fun metricMean(metricName: String): Double? {
        val numericValues =
            metricValues(metricName).filterIsInstance<Number>().map { it.toDouble() }
        return if (numericValues.isEmpty()) {
            null
        } else {
            numericValues.average()
        }
    }
}
