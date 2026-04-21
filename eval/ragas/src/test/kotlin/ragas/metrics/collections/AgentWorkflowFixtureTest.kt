package ragas.metrics.collections

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.model.MultiTurnSample
import kotlin.test.Test

class AgentWorkflowFixtureTest {
    @Test
    fun agentGoalVariantsMatchFixtureScoreBands() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH)
            val withReference = AgentGoalAccuracyWithReferenceMetric()
            val withoutReference = AgentGoalAccuracyWithoutReferenceMetric()

            fixture.jsonObject.getValue("cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val sample = parseMultiTurnSample(obj)
                val expectedScores = obj.getValue("expected_scores").jsonObject
                val expectedBands = obj.getValue("expected_bands").jsonObject

                val withReferenceScore = (withReference.multiTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    withReferenceScore,
                    expectedScores.getValue(withReference.name).jsonPrimitive.double,
                    withReference.name,
                )
                AgentFixtureTestSupport.assertScoreBand(
                    withReferenceScore,
                    expectedBands.getValue(withReference.name).jsonPrimitive.content,
                    withReference.name,
                )

                val withoutReferenceScore = (withoutReference.multiTurnAscore(sample) as Number).toDouble()
                AgentFixtureTestSupport.assertFixtureScore(
                    withoutReferenceScore,
                    expectedScores.getValue(withoutReference.name).jsonPrimitive.double,
                    withoutReference.name,
                )
                AgentFixtureTestSupport.assertScoreBand(
                    withoutReferenceScore,
                    expectedBands.getValue(withoutReference.name).jsonPrimitive.content,
                    withoutReference.name,
                )
            }
        }

    @Test
    fun workflowCompletionMatchesFixtureScoreBands() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH)
            val metric = AgentWorkflowCompletionMetric()

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

    private fun parseMultiTurnSample(obj: JsonObject): MultiTurnSample {
        val messages =
            obj
                .getValue("user_input")
                .jsonArray
                .map { AgentFixtureTestSupport.parseMessage(it.jsonObject) }

        val reference = obj["reference"]?.jsonPrimitive?.content
        return MultiTurnSample(userInput = messages, reference = reference)
    }

    private companion object {
        private const val FIXTURE_PATH = "fixtures/metrics/ws3_tier2_agent_workflow_fixture.json"
    }
}
