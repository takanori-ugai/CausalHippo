package ragas.metrics.collections

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.model.SingleTurnSample
import kotlin.test.Test
import kotlin.test.assertEquals

class QuotedAndChrfFixtureTest {
    @Test
    fun quotedSpansAlignmentMatchesFixtureScoreBands() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val metric = QuotedSpansAlignmentMetric()

            fixture.getValue("quoted_spans_cases").jsonArray.forEach { row ->
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
                AgentFixtureTestSupport.assertScoreBand(
                    score,
                    obj.getValue("expected_band").jsonPrimitive.content,
                    metric.name,
                )
            }
        }

    @Test
    fun chrfScoreMatchesFixtureScoreBands() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject
            val metric = ChrfScoreMetric()

            fixture.getValue("chrf_score_cases").jsonArray.forEach { row ->
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
    fun quotedSpanVariantsAreConfigurable() =
        runBlocking {
            val sample =
                SingleTurnSample(
                    response = "He said \"Alpha Beta\" in notes.",
                    retrievedContexts = listOf("alpha beta was recorded"),
                )

            val strict = QuotedSpansAlignmentMetric(casefold = false, minSpanWords = 2)
            val relaxed = QuotedSpansAlignmentMetric(casefold = true, minSpanWords = 2)

            val strictScore = (strict.singleTurnAscore(sample) as Number).toDouble()
            val relaxedScore = (relaxed.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(0.0, strictScore)
            assertEquals(1.0, relaxedScore)
        }

    @Test
    fun apostrophesInsideWordsFollowPythonQuotedSpanBehavior() =
        runBlocking {
            val sample =
                SingleTurnSample(
                    response = "It's called 'Alpha Beta' in the report.",
                    retrievedContexts = listOf("The report mentions alpha beta as the codename."),
                )
            val metric = QuotedSpansAlignmentMetric(casefold = true, minSpanWords = 2)

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(1.0, score)
        }

    @Test
    fun chrfShortExactMatchIsPerfect() =
        runBlocking {
            val sample =
                SingleTurnSample(
                    reference = "hi",
                    response = "hi",
                )
            val metric = ChrfScoreMetric()

            val score = (metric.singleTurnAscore(sample) as Number).toDouble()
            assertEquals(1.0, score)
        }

    @Test
    fun tier3MetricListIncludesQuotedAndChrfPorts() {
        AgentFixtureTestSupport.assertTier3MetricRegistryIncludes(
            "quoted_spans_alignment",
            "chrf_score",
            "bleu_score",
            "rouge_score",
            "semantic_similarity",
        )
    }

    private companion object {
        private const val FIXTURE_PATH = "fixtures/metrics/ws3_tier3_quoted_chrf_fixture.json"
    }
}
