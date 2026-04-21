package ragas

import kotlinx.coroutines.runBlocking
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.metrics.collections.TopicAdherenceMetric
import ragas.model.AiMessage
import ragas.model.HumanMessage
import ragas.model.MultiTurnSample
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TopicAdherenceMetricParityTest {
    @Test
    fun llmPathUsesExtractRefusalAndClassificationPipeline() =
        runBlocking {
            val metric =
                TopicAdherenceMetric(mode = TopicAdherenceMetric.Mode.F1).also { adherence ->
                    adherence.llm =
                        ScriptedTopicAdherenceLlm(
                            outputs =
                                listOf(
                                    """{"topics":["Physics","Cooking"]}""",
                                    """{"refused_to_answer":false}""",
                                    """{"refused_to_answer":true}""",
                                    """{"classifications":[true,false]}""",
                                ),
                        )
                }
            val sample =
                MultiTurnSample(
                    userInput =
                        listOf(
                            HumanMessage("Tell me about physics and pasta."),
                            AiMessage("I can explain physics but not cooking advice."),
                        ),
                    referenceTopics = listOf("Physics", "Science"),
                )

            val score = (metric.multiTurnAscore(sample) as Number).toDouble()
            assertEquals(1.0, score, 1e-9)
        }

    @Test
    fun llmPathPadsClassificationVectorToMatchExtractedTopics() =
        runBlocking {
            val metric =
                TopicAdherenceMetric(mode = TopicAdherenceMetric.Mode.PRECISION).also { adherence ->
                    adherence.llm =
                        ScriptedTopicAdherenceLlm(
                            outputs =
                                listOf(
                                    """{"topics":["Physics","Math","Cooking"]}""",
                                    """{"refused_to_answer":false}""",
                                    """{"refused_to_answer":false}""",
                                    """{"refused_to_answer":false}""",
                                    """{"classifications":[true]}""",
                                ),
                        )
                }
            val sample =
                MultiTurnSample(
                    userInput =
                        listOf(
                            HumanMessage("Talk about physics, math and cooking."),
                            AiMessage("Here are notes on all three."),
                        ),
                    referenceTopics = listOf("Physics"),
                )

            val score = (metric.multiTurnAscore(sample) as Number).toDouble()
            assertEquals(1.0 / 3.0, score, 1e-9)
        }

    @Test
    fun llmPathReturnsNanWhenNoTopicsAreExtracted() =
        runBlocking {
            val metric =
                TopicAdherenceMetric().also { adherence ->
                    adherence.llm = ScriptedTopicAdherenceLlm(outputs = listOf("""{"topics":[]}"""))
                }
            val sample =
                MultiTurnSample(
                    userInput = listOf(HumanMessage("hello"), AiMessage("hi")),
                    referenceTopics = listOf("General"),
                )

            val score = (metric.multiTurnAscore(sample) as Number).toDouble()
            assertTrue(score.isNaN())
        }

    @Test
    fun llmPathValidatesReferenceTopicsPresence() =
        runBlocking {
            val metric = TopicAdherenceMetric().also { it.llm = ScriptedTopicAdherenceLlm(outputs = emptyList()) }
            val sample =
                MultiTurnSample(
                    userInput = listOf(HumanMessage("hello"), AiMessage("hi")),
                    referenceTopics = emptyList(),
                )

            val error = assertFailsWith<IllegalArgumentException> { metric.multiTurnAscore(sample) }
            assertEquals("reference_topics must be a non-empty list of topics", error.message)
        }
}

private class ScriptedTopicAdherenceLlm(
    private val outputs: List<String>,
) : BaseRagasLlm {
    private var cursor = 0
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        val value = outputs.getOrElse(cursor) { outputs.lastOrNull().orEmpty() }
        cursor += 1
        return LlmResult(generations = listOf(LlmGeneration(value)))
    }
}
