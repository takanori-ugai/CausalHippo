package ragas.metrics.collections

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.metrics.MultiTurnMetric
import ragas.metrics.SingleTurnMetric
import ragas.metrics.defaults.ContextRecallMetric
import ragas.metrics.defaults.FaithfulnessMetric
import ragas.model.MultiTurnSample
import ragas.model.SingleTurnSample
import kotlin.test.Test

class WS9CrossLanguagePartialGoldenTest {
    @Test
    fun partialMetricsCrossLanguageGoldenBandsHold() =
        runBlocking {
            val fixture = AgentFixtureTestSupport.readFixture(FIXTURE_PATH).jsonObject

            fixture.getValue("single_turn_metric_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val metricName = obj.getValue("metric").jsonPrimitive.content
                val caseId = obj.getValue("case_id").jsonPrimitive.content
                val expectedBand = obj.getValue("expected_band").jsonPrimitive.content
                val sample = parseSingleTurnSample(obj.getValue("sample").jsonObject)
                val metric = singleTurnMetricByName(metricName)
                val score = (metric.singleTurnAscore(sample) as Number).toDouble()

                AgentFixtureTestSupport.assertScoreBand(score, expectedBand, "$metricName/$caseId")
            }

            fixture.getValue("multi_turn_metric_cases").jsonArray.forEach { row ->
                val obj = row.jsonObject
                val metricName = obj.getValue("metric").jsonPrimitive.content
                val caseId = obj.getValue("case_id").jsonPrimitive.content
                val expectedBand = obj.getValue("expected_band").jsonPrimitive.content
                val sample = parseMultiTurnSample(obj.getValue("sample").jsonObject)
                val metric = multiTurnMetricByName(metricName)
                val score = (metric.multiTurnAscore(sample) as Number).toDouble()

                AgentFixtureTestSupport.assertScoreBand(score, expectedBand, "$metricName/$caseId")
            }
        }

    private fun singleTurnMetricByName(name: String): SingleTurnMetric =
        when (name) {
            "faithfulness" -> FaithfulnessMetric()
            "context_recall" -> ContextRecallMetric()
            "answer_accuracy" -> AnswerAccuracyMetric()
            "answer_correctness" -> AnswerCorrectnessMetric()
            "factual_correctness" -> FactualCorrectnessMetric()
            "noise_sensitivity" -> NoiseSensitivityMetric()
            "summary_score" -> SummaryScoreMetric()
            "quoted_spans_alignment" -> QuotedSpansAlignmentMetric()
            "chrf_score" -> ChrfScoreMetric()
            "bleu_score" -> BleuScoreMetric()
            "rouge_score" -> RougeScoreMetric()
            "semantic_similarity" -> SemanticSimilarityMetric()
            else -> error("Unsupported single-turn metric in WS9 cross-language fixture: $name")
        }

    private fun multiTurnMetricByName(name: String): MultiTurnMetric =
        when (name) {
            "agent_goal_accuracy_with_reference" -> AgentGoalAccuracyWithReferenceMetric()
            "agent_goal_accuracy_without_reference" -> AgentGoalAccuracyWithoutReferenceMetric()
            "agent_workflow_completion" -> AgentWorkflowCompletionMetric()
            "topic_adherence" -> TopicAdherenceMetric()
            else -> error("Unsupported multi-turn metric in WS9 cross-language fixture: $name")
        }

    private fun parseSingleTurnSample(obj: JsonObject): SingleTurnSample =
        SingleTurnSample(
            userInput = obj["user_input"]?.jsonPrimitive?.content,
            retrievedContexts = obj["retrieved_contexts"]?.jsonArray?.map { it.jsonPrimitive.content },
            referenceContexts = obj["reference_contexts"]?.jsonArray?.map { it.jsonPrimitive.content },
            response = obj["response"]?.jsonPrimitive?.content,
            reference = obj["reference"]?.jsonPrimitive?.content,
        )

    private fun parseMultiTurnSample(obj: JsonObject): MultiTurnSample =
        MultiTurnSample(
            userInput = obj.getValue("messages").jsonArray.map { message -> AgentFixtureTestSupport.parseMessage(message.jsonObject) },
            reference = obj["reference"]?.jsonPrimitive?.content,
            referenceTopics = obj["reference_topics"]?.jsonArray?.map { it.jsonPrimitive.content },
        )

    private companion object {
        const val FIXTURE_PATH = "fixtures/metrics/ws9_cross_language_partial_metrics_fixture.json"
    }
}
