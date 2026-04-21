package ragas.metrics.collections

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.model.SingleTurnSample
import kotlin.test.Test

class BleuAndRougeFixtureTest {
    @Test
    fun bleuScoreMatchesFixtureScoreBands() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val metric = BleuScoreMetric()

            fixture.getValue("bleu_score_cases").jsonArray.forEach { row ->
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
    fun rougeScoreModesMatchFixtureScoreBands() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val fmeasure = RougeScoreMetric(name = "rouge_fmeasure", mode = RougeScoreMetric.Mode.FMEASURE)
            val precision = RougeScoreMetric(name = "rouge_precision", mode = RougeScoreMetric.Mode.PRECISION)
            val recall = RougeScoreMetric(name = "rouge_recall", mode = RougeScoreMetric.Mode.RECALL)

            fixture.getValue("rouge_score_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        reference = obj.getValue("reference").jsonPrimitive.content,
                        response = obj.getValue("response").jsonPrimitive.content,
                    )
                val expectedScores = obj.getValue("expected_scores").jsonObject
                val expectedBands = obj.getValue("expected_bands").jsonObject

                val f = (fmeasure.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(f, expectedScores.getValue(fmeasure.name).jsonPrimitive.double, fmeasure.name)
                AgentFixtureTestSupport.assertScoreBand(f, expectedBands.getValue(fmeasure.name).jsonPrimitive.content, fmeasure.name)

                val p = (precision.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(p, expectedScores.getValue(precision.name).jsonPrimitive.double, precision.name)
                AgentFixtureTestSupport.assertScoreBand(p, expectedBands.getValue(precision.name).jsonPrimitive.content, precision.name)

                val r = (recall.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(r, expectedScores.getValue(recall.name).jsonPrimitive.double, recall.name)
                AgentFixtureTestSupport.assertScoreBand(r, expectedBands.getValue(recall.name).jsonPrimitive.content, recall.name)
            }
        }

    @Test
    fun tier3MetricListIncludesBleuAndRougePorts() {
        AgentFixtureTestSupport.assertTier3MetricRegistryIncludes(
            "bleu_score",
            "rouge_score",
            "semantic_similarity",
        )
    }

    private companion object {
        private const val FIXTURE_PATH = "fixtures/metrics/ws3_tier3_bleu_rouge_fixture.json"
    }
}
