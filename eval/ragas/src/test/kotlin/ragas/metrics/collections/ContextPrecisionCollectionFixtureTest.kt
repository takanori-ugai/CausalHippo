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

class ContextPrecisionCollectionFixtureTest {
    @Test
    fun withReferenceMatchesFixtureBands() =
        runBlocking {
            val fixture = readFixture()
            val metric = ContextPrecisionWithReferenceMetric()

            fixture.jsonObject.getValue("cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        userInput = obj.getValue("user_input").jsonPrimitive.content,
                        reference = obj.getValue("reference").jsonPrimitive.content,
                        retrievedContexts = obj.getValue("retrieved_contexts").jsonArray.map { it.jsonPrimitive.content },
                    )

                val score = (metric.singleTurnAscore(sample) as Number).toDouble()
                val expected =
                    obj
                        .getValue("expected_scores")
                        .jsonObject
                        .getValue(metric.name)
                        .jsonPrimitive.double
                assertFixtureScore(score, expected, metric.name)
            }
        }

    @Test
    fun withoutReferenceMatchesFixtureBands() =
        runBlocking {
            val fixture = readFixture()
            val metric = ContextPrecisionWithoutReferenceMetric()

            fixture.jsonObject.getValue("cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        userInput = obj.getValue("user_input").jsonPrimitive.content,
                        response = obj.getValue("response").jsonPrimitive.content,
                        retrievedContexts = obj.getValue("retrieved_contexts").jsonArray.map { it.jsonPrimitive.content },
                    )

                val score = (metric.singleTurnAscore(sample) as Number).toDouble()
                val expected =
                    obj
                        .getValue("expected_scores")
                        .jsonObject
                        .getValue(metric.name)
                        .jsonPrimitive.double
                assertFixtureScore(score, expected, metric.name)
            }
        }

    @Test
    fun wrappersMatchBaseImplementations() =
        runBlocking {
            val sample =
                SingleTurnSample(
                    userInput = "What is the capital of France?",
                    reference = "Paris is the capital of France.",
                    response = "Paris is the capital of France.",
                    retrievedContexts =
                        listOf(
                            "Paris is the capital and largest city of France.",
                            "Berlin is the capital of Germany.",
                        ),
                )

            val wrappedWithReference = ContextPrecisionCollectionMetric().singleTurnAscore(sample) as Number
            val baseWithReference = ContextPrecisionWithReferenceMetric().singleTurnAscore(sample) as Number
            assertTrue(abs(wrappedWithReference.toDouble() - baseWithReference.toDouble()) < 1e-12)

            val wrappedWithoutReference = ContextUtilizationMetric().singleTurnAscore(sample) as Number
            val baseWithoutReference = ContextPrecisionWithoutReferenceMetric().singleTurnAscore(sample) as Number
            assertTrue(abs(wrappedWithoutReference.toDouble() - baseWithoutReference.toDouble()) < 1e-12)
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
            requireNotNull(javaClass.classLoader.getResourceAsStream("fixtures/metrics/ws3_tier1_context_precision_fixture.json")) {
                "Fixture not found on classpath: fixtures/metrics/ws3_tier1_context_precision_fixture.json"
            }.bufferedReader().use { it.readText() },
        )
}
