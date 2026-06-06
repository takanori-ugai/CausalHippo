package com.youtu.graphrag.server.api

import com.youtu.graphrag.server.api.contracts.GraphConstructionRequest
import com.youtu.graphrag.server.api.contracts.GraphConstructionResponse
import com.youtu.graphrag.server.api.contracts.ProgressEvent
import com.youtu.graphrag.server.api.contracts.QaCompleteEvent
import com.youtu.graphrag.server.api.contracts.QaUpdateEvent
import com.youtu.graphrag.server.api.contracts.QuestionRequest
import com.youtu.graphrag.server.api.contracts.QuestionResponse
import com.youtu.graphrag.server.api.contracts.ReasoningStep
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ApiContractsTest {
    private val json = Json { encodeDefaults = true }

    @Test
    fun `graph construction request decodes python-style snake_case field`() {
        val decoded = json.decodeFromString<GraphConstructionRequest>("""{"dataset_name":"demo"}""")
        assertEquals("demo", decoded.datasetName)
    }

    @Test
    fun `question response serializes snake_case compatibility fields`() {
        val encoded =
            json.encodeToString(
                QuestionResponse(
                    answer = "Tokyo",
                    subQuestions = listOf(mapOf("sub-question" to "Where is Project Alpha based?")),
                    retrievedTriples = listOf("""["Project Alpha","based_in","Tokyo"]"""),
                    retrievedChunks = listOf("Project Alpha is based in Tokyo."),
                    reasoningSteps =
                        listOf(
                            ReasoningStep(
                                type = "ircot_step",
                                question = "Where is Project Alpha based?",
                                triples = listOf("""["Project Alpha","based_in","Tokyo"]"""),
                                triplesCount = 1,
                                chunkContents = listOf("Project Alpha is based in Tokyo."),
                                chunksCount = 1,
                                processingTime = 0.25,
                            ),
                        ),
                    visualizationData =
                        buildJsonObject {
                            put("dummy", "ok")
                        },
                ),
            )

        assertTrue("\"sub_questions\"" in encoded)
        assertTrue("\"retrieved_triples\"" in encoded)
        assertTrue("\"retrieved_chunks\"" in encoded)
        assertTrue("\"reasoning_steps\"" in encoded)
        assertTrue("\"visualization_data\"" in encoded)
    }

    @Test
    fun `progress and qa complete events keep stable websocket envelope fields`() {
        val progressEncoded =
            json.encodeToString(
                ProgressEvent(
                    stage = "construction",
                    progress = 45,
                    message = "Building graph",
                    timestamp = "2026-05-13T00:00:00Z",
                ),
            )
        val completeEncoded =
            json.encodeToString(
                QaCompleteEvent(
                    answerPreview = "Tokyo",
                    timestamp = "2026-05-13T00:00:00Z",
                ),
            )

        assertTrue("\"type\":\"progress\"" in progressEncoded)
        assertTrue("\"stage\":\"construction\"" in progressEncoded)
        assertTrue("\"answer_preview\":\"Tokyo\"" in completeEncoded)
        assertTrue("\"type\":\"qa_complete\"" in completeEncoded)
    }

    @Test
    fun `request and response contracts keep snake_case api compatibility`() {
        val requestEncoded =
            json.encodeToString(
                QuestionRequest(
                    question = "Who leads Project Alpha?",
                    datasetName = "demo",
                ),
            )
        val graphResponseEncoded =
            json.encodeToString(
                GraphConstructionResponse(
                    success = true,
                    message = "ok",
                    graphData =
                        JsonObject(
                            mapOf(
                                "nodes" to JsonPrimitive("[]"),
                            ),
                        ),
                ),
            )

        assertTrue("\"dataset_name\":\"demo\"" in requestEncoded)
        assertTrue("\"graph_data\"" in graphResponseEncoded)
    }

    @Test
    fun `qa update event preserves websocket envelope contract`() {
        val encoded =
            json.encodeToString(
                QaUpdateEvent(
                    stage = "ircot",
                    message = "Iterative retrieval step 1...",
                    timestamp = "2026-05-13T00:00:00Z",
                ),
            )

        assertTrue("\"type\":\"qa_update\"" in encoded)
        assertTrue("\"stage\":\"ircot\"" in encoded)
        assertTrue("\"message\":\"Iterative retrieval step 1...\"" in encoded)
    }
}
