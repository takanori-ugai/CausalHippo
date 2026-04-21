package ragas.metrics.collections

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.model.SingleTurnSample

open class DomainSpecificRubricsMetric(
    name: String = "domain_specific_rubrics",
    rubrics: Map<String, String>? = null,
    private val withReference: Boolean = false,
) : BaseMetric(
        name = name,
        requiredColumns =
            mapOf(
                MetricType.SINGLE_TURN to
                    if (withReference) {
                        setOf("user_input", "response", "reference")
                    } else {
                        setOf("user_input", "response")
                    },
            ),
        outputType = MetricOutputType.DISCRETE,
    ),
    SingleTurnMetric {
    private val normalizedRubrics: Map<Int, String>

    init {
        val selectedRubrics =
            rubrics
                ?: if (withReference) {
                    DEFAULT_WITH_REFERENCE_RUBRICS
                } else {
                    DEFAULT_REFERENCE_FREE_RUBRICS
                }
        normalizedRubrics = normalizeRubrics(selectedRubrics)
    }

    /**
     * Executes singleTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val userInput = sample.userInput.orEmpty().trim()
        val response = sample.response.orEmpty().trim()
        require(userInput.isNotBlank()) { "user_input is missing. Please provide a question to evaluate against." }
        require(response.isNotBlank()) { "response is missing. Please provide a response to evaluate." }

        return computeRubricScore(
            rubrics = normalizedRubrics,
            userInput = userInput,
            response = response,
            reference = sample.reference,
            retrievedContexts = sample.retrievedContexts,
            referenceContexts = sample.referenceContexts,
            withReference = withReference,
        )
    }
}

/**
 * Implements [RubricsScoreWithoutReferenceMetric].
 */
class RubricsScoreWithoutReferenceMetric(
    rubrics: Map<String, String>? = null,
    name: String = "rubrics_score_without_reference",
) : DomainSpecificRubricsMetric(
        name = name,
        rubrics = rubrics,
        withReference = false,
    )

/**
 * Implements [RubricsScoreWithReferenceMetric].
 */
class RubricsScoreWithReferenceMetric(
    rubrics: Map<String, String>? = null,
    name: String = "rubrics_score_with_reference",
) : DomainSpecificRubricsMetric(
        name = name,
        rubrics = rubrics,
        withReference = true,
    )
