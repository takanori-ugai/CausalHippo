package ragas.metrics.collections

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.model.SingleTurnSample
import kotlin.test.Test
import kotlin.test.assertTrue

class NoiseAndSummaryFixtureTest {
    @Test
    fun noiseSensitivityMatchesFixtureScoreBands() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val relevant = NoiseSensitivityMetric(mode = NoiseSensitivityMetric.Mode.RELEVANT, name = "noise_sensitivity_relevant")
            val irrelevant = NoiseSensitivityMetric(mode = NoiseSensitivityMetric.Mode.IRRELEVANT, name = "noise_sensitivity_irrelevant")

            fixture.getValue("noise_sensitivity_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        userInput = obj.getValue("user_input").jsonPrimitive.content,
                        response = obj.getValue("response").jsonPrimitive.content,
                        reference = obj.getValue("reference").jsonPrimitive.content,
                        retrievedContexts = obj.getValue("retrieved_contexts").jsonArray.map { it.jsonPrimitive.content },
                    )
                val expectedScores = obj.getValue("expected_scores").jsonObject
                val expectedBands = obj.getValue("expected_bands").jsonObject

                val relScore = (relevant.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    relScore,
                    expectedScores.getValue(relevant.name).jsonPrimitive.double,
                    relevant.name,
                )
                AgentFixtureTestSupport.assertScoreBand(
                    relScore,
                    expectedBands.getValue(relevant.name).jsonPrimitive.content,
                    relevant.name,
                )

                val irrScore = (irrelevant.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    irrScore,
                    expectedScores.getValue(irrelevant.name).jsonPrimitive.double,
                    irrelevant.name,
                )
                AgentFixtureTestSupport.assertScoreBand(
                    irrScore,
                    expectedBands.getValue(irrelevant.name).jsonPrimitive.content,
                    irrelevant.name,
                )
            }
        }

    @Test
    fun summaryScoreMatchesFixtureScoreBands() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val defaultMetric = SummaryScoreMetric(name = "summary_score_default")
            val noPenaltyMetric = SummaryScoreMetric(name = "summary_score_no_penalty", lengthPenalty = false)

            fixture.getValue("summary_score_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        response = obj.getValue("response").jsonPrimitive.content,
                        referenceContexts = obj.getValue("reference_contexts").jsonArray.map { it.jsonPrimitive.content },
                    )
                val expectedScores = obj.getValue("expected_scores").jsonObject
                val expectedBands = obj.getValue("expected_bands").jsonObject

                val defaultScore = (defaultMetric.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    defaultScore,
                    expectedScores.getValue(defaultMetric.name).jsonPrimitive.double,
                    defaultMetric.name,
                )
                AgentFixtureTestSupport.assertScoreBand(
                    defaultScore,
                    expectedBands.getValue(defaultMetric.name).jsonPrimitive.content,
                    defaultMetric.name,
                )

                val noPenaltyScore = (noPenaltyMetric.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    noPenaltyScore,
                    expectedScores.getValue(noPenaltyMetric.name).jsonPrimitive.double,
                    noPenaltyMetric.name,
                )
                AgentFixtureTestSupport.assertScoreBand(
                    noPenaltyScore,
                    expectedBands.getValue(noPenaltyMetric.name).jsonPrimitive.content,
                    noPenaltyMetric.name,
                )
            }
        }

    @Test
    fun summaryScoreValidatesCoefficientBounds() {
        val tooLow =
            runCatching {
                SummaryScoreMetric(coeff = -0.1)
            }.exceptionOrNull()
        assertTrue(tooLow is IllegalArgumentException)

        val tooHigh =
            runCatching {
                SummaryScoreMetric(coeff = 1.1)
            }.exceptionOrNull()
        assertTrue(tooHigh is IllegalArgumentException)
    }

    @Test
    fun tier3MetricListIncludesNoiseAndSummaryPorts() {
        AgentFixtureTestSupport.assertTier3MetricRegistryIncludes(
            "noise_sensitivity",
            "summary_score",
            "quoted_spans_alignment",
            "chrf_score",
            "bleu_score",
            "rouge_score",
            "semantic_similarity",
        )
    }

    private companion object {
        private const val FIXTURE_PATH = "fixtures/metrics/ws3_tier3_noise_summary_fixture.json"
    }
}
