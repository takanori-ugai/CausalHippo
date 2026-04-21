package ragas

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.evaluation.evaluate
import ragas.model.EvaluationDataset
import ragas.model.EvaluationResult
import ragas.model.SingleTurnSample
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GoldenFixturesTest {
    @Test
    fun defaultMetricsFixtureMatchesExpectedScores() {
        val fixture = readFixture("default_metrics_golden.json")
        val datasetRows = fixture.jsonObject.getValue("dataset").jsonArray

        val dataset =
            EvaluationDataset(
                datasetRows.map { row ->
                    val obj = row.jsonObject
                    SingleTurnSample(
                        userInput = obj.getValue("user_input").jsonPrimitive.content,
                        response = obj.getValue("response").jsonPrimitive.content,
                        retrievedContexts = obj.getValue("retrieved_contexts").jsonArray.map { it.jsonPrimitive.content },
                        referenceContexts = obj.getValue("reference_contexts").jsonArray.map { it.jsonPrimitive.content },
                        reference = obj.getValue("reference").jsonPrimitive.content,
                    )
                },
            )

        val result = evaluate(dataset = dataset, metrics = null)
        val expected = fixture.jsonObject.getValue("expected_scores").jsonObject
        val row = result.scores.first()

        expected.forEach { (metric, value) ->
            val actual = (row[metric] as Number).toDouble()
            val target = value.jsonPrimitive.double
            assertTrue(abs(actual - target) < 1e-9, "metric=$metric expected=$target actual=$actual")
        }
    }

    @Test
    fun aggregationFixtureMatchesEvaluationResultMean() {
        val fixture = readFixture("aggregation_fixture.json")
        val scores =
            fixture.jsonObject.getValue("scores").jsonArray.map { row ->
                val obj = row.jsonObject
                mapOf(
                    "metric_a" to obj.getValue("metric_a").jsonPrimitive.double,
                    "metric_b" to obj.getValue("metric_b").jsonPrimitive.double,
                )
            }

        val eval =
            EvaluationResult(
                scores = scores,
                dataset = EvaluationDataset(emptyList<SingleTurnSample>()),
            )

        val expected = fixture.jsonObject.getValue("expected").jsonObject
        assertTrue(abs(eval.metricMean("metric_a")!! - expected.getValue("metric_a_mean").jsonPrimitive.double) < 1e-9)
        assertTrue(abs(eval.metricMean("metric_b")!! - expected.getValue("metric_b_mean").jsonPrimitive.double) < 1e-9)
        assertEquals(null, eval.metricMean("missing_metric"))
    }

    private fun readFixture(name: String) =
        Json.parseToJsonElement(
            requireNotNull(javaClass.classLoader.getResourceAsStream("fixtures/evaluation/$name")) {
                "Fixture not found on classpath: fixtures/evaluation/$name"
            }.bufferedReader().use { it.readText() },
        )
}
