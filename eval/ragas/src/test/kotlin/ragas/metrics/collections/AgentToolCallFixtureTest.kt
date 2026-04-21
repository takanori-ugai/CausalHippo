package ragas.metrics.collections

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.model.AiMessage
import ragas.model.HumanMessage
import ragas.model.MultiTurnSample
import ragas.model.ToolCall
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AgentToolCallFixtureTest {
    @Test
    fun toolCallAccuracyMatchesFixtureScoreBands() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH)
            val strictMetric = ToolCallAccuracyMetric(strictOrder = true, name = "tool_call_accuracy_strict")
            val relaxedMetric = ToolCallAccuracyMetric(strictOrder = false, name = "tool_call_accuracy_relaxed")

            fixture.jsonObject.getValue("cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample = parseMultiTurnSample(obj)
                val expectedScores = obj.getValue("expected_scores").jsonObject
                val expectedBands = obj.getValue("expected_bands").jsonObject

                val strictScore = (strictMetric.multiTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    strictScore,
                    expectedScores.getValue(strictMetric.name).jsonPrimitive.double,
                    strictMetric.name,
                )
                AgentFixtureTestSupport.assertScoreBand(
                    strictScore,
                    expectedBands.getValue(strictMetric.name).jsonPrimitive.content,
                    strictMetric.name,
                )

                val relaxedScore = (relaxedMetric.multiTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    relaxedScore,
                    expectedScores.getValue(relaxedMetric.name).jsonPrimitive.double,
                    relaxedMetric.name,
                )
                AgentFixtureTestSupport.assertScoreBand(
                    relaxedScore,
                    expectedBands.getValue(relaxedMetric.name).jsonPrimitive.content,
                    relaxedMetric.name,
                )
            }
        }

    @Test
    fun toolCallF1MatchesFixtureScoreBands() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH)
            val metric = ToolCallF1Metric()

            fixture.jsonObject.getValue("cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample = parseMultiTurnSample(obj)
                val expectedScores = obj.getValue("expected_scores").jsonObject
                val expectedBands = obj.getValue("expected_bands").jsonObject

                val score = (metric.multiTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    score,
                    expectedScores.getValue(metric.name).jsonPrimitive.double,
                    metric.name,
                )
                AgentFixtureTestSupport.assertScoreBand(
                    score,
                    expectedBands.getValue(metric.name).jsonPrimitive.content,
                    metric.name,
                )
            }
        }

    @Test
    fun tier2MetricListIncludesExpectedMetrics() {
        val names = agentToolCallTier2Metrics().map { metric -> metric.name }
        assertTrue("tool_call_accuracy" in names)
        assertTrue("tool_call_f1" in names)
        assertTrue("agent_goal_accuracy_with_reference" in names)
        assertTrue("agent_goal_accuracy_without_reference" in names)
        assertTrue("agent_workflow_completion" in names)
    }

    @Test
    fun toolCallAccuracyRelaxedIgnoresNestedObjectKeyOrderInSortKey() =
        runBlocking {
            val metric = ToolCallAccuracyMetric(strictOrder = false)
            val sample =
                MultiTurnSample(
                    userInput =
                        listOf(
                            AiMessage(
                                content = "calls",
                                toolCalls =
                                    listOf(
                                        ToolCall(
                                            name = "search",
                                            args = mapOf("payload" to Json.parseToJsonElement("""{"m":0}""").jsonObject),
                                        ),
                                        ToolCall(
                                            name = "search",
                                            args = mapOf("payload" to Json.parseToJsonElement("""{"z":1,"a":0}""").jsonObject),
                                        ),
                                    ),
                            ),
                        ),
                    referenceToolCalls =
                        listOf(
                            ToolCall(
                                name = "search",
                                args = mapOf("payload" to Json.parseToJsonElement("""{"a":0,"z":1}""").jsonObject),
                            ),
                            ToolCall(
                                name = "search",
                                args = mapOf("payload" to Json.parseToJsonElement("""{"m":0}""").jsonObject),
                            ),
                        ),
                )

            val score = (metric.multiTurnAscore(sample) as Number).toDouble()
            assertTrue(abs(score - 1.0) < 1e-9, "expected relaxed score=1.0 actual=$score")
        }

    @Test
    fun toolCallF1IsZeroWhenBothExpectedAndActualAreEmpty() =
        runBlocking {
            val metric = ToolCallF1Metric()
            val sample =
                MultiTurnSample(
                    userInput =
                        listOf(
                            HumanMessage(content = "Say hello."),
                            AiMessage(content = "Hello!"),
                        ),
                    referenceToolCalls = emptyList(),
                )

            val score = (metric.multiTurnAscore(sample) as Number).toDouble()
            assertEquals(0.0, score)
        }

    private fun parseMultiTurnSample(obj: JsonObject): MultiTurnSample {
        val messages =
            obj
                .getValue("user_input")
                .jsonArray
                .map { AgentFixtureTestSupport.parseMessage(it.jsonObject) }

        val referenceToolCalls =
            obj
                .getValue("reference_tool_calls")
                .jsonArray
                .map { AgentFixtureTestSupport.parseToolCall(it.jsonObject) }

        return MultiTurnSample(userInput = messages, referenceToolCalls = referenceToolCalls)
    }

    private companion object {
        private const val FIXTURE_PATH = "fixtures/metrics/ws3_tier2_agent_tool_call_fixture.json"
    }
}
