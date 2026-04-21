package ragas.metrics.collections

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.model.SingleTurnSample
import kotlin.test.Test
import kotlin.test.assertTrue

class SemanticSimilarityFixtureTest {
    @Test
    fun semanticSimilarityMatchesFixtureScoreBands() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val metric = SemanticSimilarityMetric()

            fixture.getValue("cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        reference = obj.getValue("reference").jsonPrimitive.content,
                        response = obj.getValue("response").jsonPrimitive.content,
                    )
                val score = (metric.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    score,
                    obj.getValue("expected_score").jsonPrimitive.double,
                    metric.name,
                )
                AgentFixtureTestSupport.assertScoreBand(
                    score,
                    obj.getValue("expected_band").jsonPrimitive.content,
                    metric.name,
                )
            }
        }

    @Test
    fun semanticSimilarityThresholdMatchesFixture() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject

            fixture.getValue("threshold_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val metric = SemanticSimilarityMetric(threshold = obj.getValue("threshold").jsonPrimitive.double)
                val sample =
                    SingleTurnSample(
                        reference = obj.getValue("reference").jsonPrimitive.content,
                        response = obj.getValue("response").jsonPrimitive.content,
                    )
                val score = (metric.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    score,
                    obj.getValue("expected_binary").jsonPrimitive.double,
                    metric.name,
                )
            }
        }

    @Test
    fun semanticSimilarityValidatesThresholdBoundsAndFiniteness() {
        assertTrue(runCatching { SemanticSimilarityMetric(threshold = -0.1) }.exceptionOrNull() is IllegalArgumentException)
        assertTrue(runCatching { SemanticSimilarityMetric(threshold = 1.1) }.exceptionOrNull() is IllegalArgumentException)
        assertTrue(runCatching { SemanticSimilarityMetric(threshold = Double.NaN) }.exceptionOrNull() is IllegalArgumentException)
        assertTrue(
            runCatching { SemanticSimilarityMetric(threshold = Double.POSITIVE_INFINITY) }.exceptionOrNull() is IllegalArgumentException,
        )
    }

    @Test
    fun tier3MetricListIncludesSemanticSimilarityPort() {
        AgentFixtureTestSupport.assertTier3MetricRegistryIncludes("semantic_similarity")
    }

    private companion object {
        private const val FIXTURE_PATH = "fixtures/metrics/ws3_tier3_semantic_similarity_fixture.json"
    }
}
