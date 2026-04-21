package ragas.metrics.collections

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.model.SingleTurnSample
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

class Tier1RetrievalGroundednessFixtureTest {
    @Test
    fun contextRelevanceMatchesPythonFixtureBands() =
        runBlocking {
            val fixture = readFixture()
            val metric = ContextRelevanceMetric()

            fixture.jsonObject.getValue("context_relevance_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        userInput = obj.getValue("user_input").jsonPrimitive.content,
                        retrievedContexts = obj.getValue("retrieved_contexts").jsonArray.map { it.jsonPrimitive.content },
                    )

                val score = (metric.singleTurnAscore(sample) as Number).toDouble()
                assertFixtureScore(score, obj.getValue("expected_score").jsonPrimitive.double, metric.name)
            }
        }

    @Test
    fun responseGroundednessMatchesPythonFixtureBands() =
        runBlocking {
            val fixture = readFixture()
            val metric = ResponseGroundednessMetric()

            fixture.jsonObject.getValue("response_groundedness_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        response = obj.getValue("response").jsonPrimitive.content,
                        retrievedContexts = obj.getValue("retrieved_contexts").jsonArray.map { it.jsonPrimitive.content },
                    )

                val score = (metric.singleTurnAscore(sample) as Number).toDouble()
                assertFixtureScore(score, obj.getValue("expected_score").jsonPrimitive.double, metric.name)
            }
        }

    @Test
    fun tier1MetricListIncludesPorts() {
        val names = retrievalGroundednessTier1Metrics().map { metric -> metric.name }
        assertTrue("context_relevance" in names)
        assertTrue("response_groundedness" in names)
        assertTrue("context_precision_with_reference" in names)
        assertTrue("context_precision_without_reference" in names)
        assertTrue("id_based_context_precision" in names)
        assertTrue("context_entity_recall" in names)
    }

    private fun assertFixtureScore(
        score: Double,
        expected: Double,
        metricName: String,
    ) {
        assertTrue(abs(score - expected) < 1e-9, "metric=$metricName expected=$expected actual=$score")
    }

    private fun readFixture() =
        Json.parseToJsonElement(
            requireNotNull(javaClass.classLoader.getResourceAsStream("fixtures/metrics/ws3_tier1_retrieval_groundedness_fixture.json")) {
                "Fixture not found on classpath: fixtures/metrics/ws3_tier1_retrieval_groundedness_fixture.json"
            }.bufferedReader().use { it.readText() },
        )
}
