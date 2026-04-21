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

class Tier4AdvancedFixtureTest {
    @Test
    fun sqlSemanticEquivalenceMatchesFixture() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val metric = SqlSemanticEquivalenceMetric()

            fixture.getValue("sql_semantic_equivalence_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        response = obj.getValue("response").jsonPrimitive.content,
                        reference = obj.getValue("reference").jsonPrimitive.content,
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
    fun dataCompyScoreMatchesFixture() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject

            fixture.getValue("datacompy_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val mode =
                    when (obj.getValue("mode").jsonPrimitive.content) {
                        "rows" -> DataCompyScoreMetric.Mode.ROWS
                        "columns" -> DataCompyScoreMetric.Mode.COLUMNS
                        else -> error("Unsupported mode in fixture")
                    }
                val metricType =
                    when (obj.getValue("metric").jsonPrimitive.content) {
                        "precision" -> DataCompyScoreMetric.Metric.PRECISION
                        "recall" -> DataCompyScoreMetric.Metric.RECALL
                        "f1" -> DataCompyScoreMetric.Metric.F1
                        else -> error("Unsupported metric in fixture")
                    }
                val metric = DataCompyScoreMetric(mode = mode, metric = metricType)
                val sample =
                    SingleTurnSample(
                        reference = obj.getValue("reference").jsonPrimitive.content,
                        response = obj.getValue("response").jsonPrimitive.content,
                    )
                val score = (metric.singleTurnAscore(sample) as Number).toDouble()
                val expectedRaw = obj.getValue("expected_score").jsonPrimitive.content
                if (expectedRaw == "NaN") {
                    assertTrue(score.isNaN(), "metric=${metric.name} expected=NaN actual=$score")
                } else {
                    AgentFixtureTestSupport.assertFixtureScore(
                        score,
                        expectedRaw.toDouble(),
                        metric.name,
                    )
                }
            }
        }

    @Test
    fun dataCompyScoreParsesQuotedMultilineCells() =
        runBlocking {
            val csv =
                """
                id,notes
                1,"first line
                second line"
                """.trimIndent()
            val metric = DataCompyScoreMetric(mode = DataCompyScoreMetric.Mode.ROWS, metric = DataCompyScoreMetric.Metric.F1)
            val sample =
                SingleTurnSample(
                    reference = csv,
                    response = csv,
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(1.0, score)
        }

    @Test
    fun dataCompyRowModeMatchesUsingOnlySharedColumns() =
        runBlocking {
            val reference =
                """
                id,name
                1,Alice
                2,Bob
                """.trimIndent()
            val response =
                """
                id,name,age
                1,Alice,30
                2,Bob,40
                """.trimIndent()
            val metric = DataCompyScoreMetric(mode = DataCompyScoreMetric.Mode.ROWS, metric = DataCompyScoreMetric.Metric.F1)
            val sample =
                SingleTurnSample(
                    reference = reference,
                    response = response,
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(1.0, score)
        }

    @Test
    fun dataCompyRowModeReturnsZeroWhenNoSharedColumns() =
        runBlocking {
            val reference =
                """
                id,name
                1,Alice
                """.trimIndent()
            val response =
                """
                age,city
                30,Tokyo
                """.trimIndent()
            val metric = DataCompyScoreMetric(mode = DataCompyScoreMetric.Mode.ROWS, metric = DataCompyScoreMetric.Metric.F1)
            val sample =
                SingleTurnSample(
                    reference = reference,
                    response = response,
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(0.0, score)
        }

    @Test
    fun looksLikeImageContextClassifiesHttpUrlsByExtensionOnly() {
        assertTrue("https://example.com/cat.jpg".looksLikeImageContext())
        assertTrue("https://example.com/cat.png?size=large".looksLikeImageContext())
        assertTrue(!"https://example.com/docs/index.html".looksLikeImageContext())
    }

    @Test
    fun multiModalRelevanceTreatsNonImageHttpUrlAsTextContext() =
        runBlocking {
            val metric = MultiModalRelevanceMetric()
            val sample =
                SingleTurnSample(
                    userInput = "Explain how nuclear fusion powers the sun",
                    response = "Nuclear fusion powers the sun by combining hydrogen into helium.",
                    retrievedContexts = listOf("https://example.com/docs/solar-fusion.html"),
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(1.0, score)
        }

    @Test
    fun multiModalRelevanceMatchesFixture() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val metric = MultiModalRelevanceMetric()

            fixture.getValue("multi_modal_relevance_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        userInput = obj.getValue("user_input").jsonPrimitive.content,
                        response = obj.getValue("response").jsonPrimitive.content,
                        retrievedContexts = obj.getValue("retrieved_contexts").jsonArray.map { it.jsonPrimitive.content },
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
    fun multiModalFaithfulnessMatchesFixture() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val metric = MultiModalFaithfulnessMetric()

            fixture.getValue("multi_modal_faithfulness_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample =
                    SingleTurnSample(
                        response = obj.getValue("response").jsonPrimitive.content,
                        retrievedContexts = obj.getValue("retrieved_contexts").jsonArray.map { it.jsonPrimitive.content },
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
    fun sqlSemanticEquivalenceValidatesRequiredInputs() =
        runBlocking {
            val err =
                runCatching {
                    SqlSemanticEquivalenceMetric().singleTurnAscore(
                        SingleTurnSample(
                            response = "",
                            reference = "SELECT 1",
                        ),
                    )
                }.exceptionOrNull()
            assertTrue(err is IllegalArgumentException)
        }

    @Test
    fun sqlSemanticEquivalenceValidatesRequiredReference() =
        runBlocking {
            val err =
                runCatching {
                    SqlSemanticEquivalenceMetric().singleTurnAscore(
                        SingleTurnSample(
                            response = "SELECT 1",
                            reference = "",
                        ),
                    )
                }.exceptionOrNull()
            assertTrue(err is IllegalArgumentException)
        }

    @Test
    fun sqlSemanticEquivalenceNormalizesBareBooleanWhereLiteral() =
        runBlocking {
            val metric = SqlSemanticEquivalenceMetric()
            val sample =
                SingleTurnSample(
                    response = "SELECT id FROM users WHERE true;",
                    reference = "SELECT id FROM users WHERE 1;",
                )

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(1.0, score)
        }

    @Test
    fun tier4MetricListIncludesAllPorts() {
        AgentFixtureTestSupport.assertTier4MetricRegistryMatchesExpected()
    }

    private companion object {
        private const val FIXTURE_PATH = "fixtures/metrics/ws3_tier4_advanced_fixture.json"
    }
}
