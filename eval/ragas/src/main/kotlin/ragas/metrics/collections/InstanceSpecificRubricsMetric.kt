package ragas.metrics.collections

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.model.SingleTurnSample

/**
 * Scores a response against rubrics provided per sample instance.
 */
class InstanceSpecificRubricsMetric(
    name: String = "instance_specific_rubrics",
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("user_input", "response", "rubrics")),
        outputType = MetricOutputType.DISCRETE,
    ),
    SingleTurnMetric {
    /**
     * Executes singleTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val sampleRubrics = sample.rubrics
        require(!sampleRubrics.isNullOrEmpty()) {
            "rubrics must be provided for instance-specific evaluation"
        }
        val userInput =
            requireNotNull(sample.userInput?.takeIf { it.isNotBlank() }) {
                "user_input must be provided for instance-specific evaluation"
            }
        val response =
            requireNotNull(sample.response?.takeIf { it.isNotBlank() }) {
                "response must be provided for instance-specific evaluation"
            }
        val normalizedRubrics = normalizeRubrics(sampleRubrics)

        return computeRubricScore(
            rubrics = normalizedRubrics,
            userInput = userInput,
            response = response,
            reference = sample.reference,
            retrievedContexts = sample.retrievedContexts,
            referenceContexts = sample.referenceContexts,
            withReference = !sample.reference.isNullOrBlank(),
        )
    }
}
