package ragas.metrics.collections

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.model.SingleTurnSample
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Tier4RubricsFixtureTest {
    @Test
    fun domainSpecificRubricsWithoutReferenceMatchesFixture() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val metric = DomainSpecificRubricsMetric(withReference = false)

            fixture.getValue("domain_without_reference_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        userInput = obj.getValue("user_input").jsonPrimitive.content,
                        response = obj.getValue("response").jsonPrimitive.content,
                        retrievedContexts = obj["retrieved_contexts"]?.jsonArray?.map { it.jsonPrimitive.content },
                        referenceContexts = obj["reference_contexts"]?.jsonArray?.map { it.jsonPrimitive.content },
                    )
                val score = (metric.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    score,
                    obj.getValue("expected_score").jsonPrimitive.double,
                    metric.name,
                )
            }
        }

    @Test
    fun domainSpecificRubricsWithReferenceMatchesFixture() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val metric = DomainSpecificRubricsMetric(withReference = true)

            fixture.getValue("domain_with_reference_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        userInput = obj.getValue("user_input").jsonPrimitive.content,
                        response = obj.getValue("response").jsonPrimitive.content,
                        reference = obj.getValue("reference").jsonPrimitive.content,
                        retrievedContexts = obj["retrieved_contexts"]?.jsonArray?.map { it.jsonPrimitive.content },
                        referenceContexts = obj["reference_contexts"]?.jsonArray?.map { it.jsonPrimitive.content },
                    )
                val score = (metric.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    score,
                    obj.getValue("expected_score").jsonPrimitive.double,
                    metric.name,
                )
            }
        }

    @Test
    fun convenienceRubricsMetricsMatchFixture() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val withoutReferenceMetric = RubricsScoreWithoutReferenceMetric()
            val withReferenceMetric = RubricsScoreWithReferenceMetric()

            fixture.getValue("convenience_metric_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val metricName = obj.getValue("metric").jsonPrimitive.content
                val metric =
                    when (metricName) {
                        withoutReferenceMetric.name -> withoutReferenceMetric
                        withReferenceMetric.name -> withReferenceMetric
                        else -> error("Unknown convenience metric in fixture: $metricName")
                    }
                val sample =
                    SingleTurnSample(
                        userInput = obj.getValue("user_input").jsonPrimitive.content,
                        response = obj.getValue("response").jsonPrimitive.content,
                        reference = obj["reference"]?.jsonPrimitive?.content,
                        retrievedContexts = obj["retrieved_contexts"]?.jsonArray?.map { it.jsonPrimitive.content },
                        referenceContexts = obj["reference_contexts"]?.jsonArray?.map { it.jsonPrimitive.content },
                    )
                val score = (metric.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    score,
                    obj.getValue("expected_score").jsonPrimitive.double,
                    metric.name,
                )
            }
        }

    @Test
    fun domainSpecificRubricsRequiresUserInput() =
        runBlocking {
            val err =
                runCatching {
                    DomainSpecificRubricsMetric(withReference = false).singleTurnAscore(
                        SingleTurnSample(
                            response = "Photosynthesis lets plants make food from sunlight.",
                        ),
                    )
                }.exceptionOrNull()
            assertTrue(err is IllegalArgumentException)
        }

    @Test
    fun domainSpecificRubricsRequiresResponse() =
        runBlocking {
            val err =
                runCatching {
                    DomainSpecificRubricsMetric(withReference = false).singleTurnAscore(
                        SingleTurnSample(
                            userInput = "Explain photosynthesis",
                        ),
                    )
                }.exceptionOrNull()
            assertTrue(err is IllegalArgumentException)
        }

    @Test
    fun instanceSpecificRubricsMatchesFixture() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val metric = InstanceSpecificRubricsMetric()

            fixture.getValue("instance_specific_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        userInput = obj.getValue("user_input").jsonPrimitive.content,
                        response = obj.getValue("response").jsonPrimitive.content,
                        reference = obj["reference"]?.jsonPrimitive?.content,
                        retrievedContexts = obj["retrieved_contexts"]?.jsonArray?.map { it.jsonPrimitive.content },
                        referenceContexts = obj["reference_contexts"]?.jsonArray?.map { it.jsonPrimitive.content },
                        rubrics = obj.getValue("rubrics").jsonObject.mapValues { (_, value) -> value.jsonPrimitive.content },
                    )
                val score = (metric.singleTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    score,
                    obj.getValue("expected_score").jsonPrimitive.double,
                    metric.name,
                )
            }
        }

    @Test
    fun instanceSpecificRubricsRequiresRubrics() =
        runBlocking {
            val err =
                runCatching {
                    InstanceSpecificRubricsMetric().singleTurnAscore(
                        SingleTurnSample(
                            userInput = "Explain photosynthesis",
                            response = "Photosynthesis lets plants make food from sunlight.",
                        ),
                    )
                }.exceptionOrNull()
            assertTrue(err is IllegalArgumentException)
        }

    @Test
    fun instanceSpecificRubricsRequiresUserInput() =
        runBlocking {
            val err =
                runCatching {
                    InstanceSpecificRubricsMetric().singleTurnAscore(
                        SingleTurnSample(
                            response = "Photosynthesis lets plants make food from sunlight.",
                            rubrics = mapOf("0" to "wrong", "5" to "correct"),
                        ),
                    )
                }.exceptionOrNull()
            assertTrue(err is IllegalArgumentException)
        }

    @Test
    fun instanceSpecificRubricsRequiresResponse() =
        runBlocking {
            val err =
                runCatching {
                    InstanceSpecificRubricsMetric().singleTurnAscore(
                        SingleTurnSample(
                            userInput = "Explain photosynthesis",
                            rubrics = mapOf("0" to "wrong", "5" to "correct"),
                        ),
                    )
                }.exceptionOrNull()
            assertTrue(err is IllegalArgumentException)
        }

    @Test
    fun tier4MetricListIncludesRubricsPorts() {
        AgentFixtureTestSupport.assertTier4MetricRegistryMatchesExpected()
    }

    @Test
    fun advancedRubricsTier4MetricsReturnsOnlyRubricsMetrics() {
        val names = advancedRubricsTier4Metrics().map { metric -> metric.name }.toSet()
        assertEquals(4, names.size)
        assertTrue("domain_specific_rubrics" in names)
        assertTrue("rubrics_score_without_reference" in names)
        assertTrue("rubrics_score_with_reference" in names)
        assertTrue("instance_specific_rubrics" in names)
    }

    private companion object {
        private const val FIXTURE_PATH = "fixtures/metrics/ws3_tier4_rubrics_fixture.json"
    }
}
