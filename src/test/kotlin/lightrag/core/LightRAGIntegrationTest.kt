package lightrag.core

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.request.ChatRequest
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.output.Response
import io.mockk.CapturingSlot
import io.mockk.every
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.runBlocking
import lightrag.buildTestLightRag
import org.junit.Test
import java.io.File
import kotlin.test.assertTrue

/** Integration tests verifying ingest + query flows across naive and local modes. */
class LightRAGIntegrationTest {
    /** Confirms naive mode surfaces inserted text in the returned context/system prompt. */
    @Test
    fun testInsertAndQueryNaiveMode() {
        runBlocking {
            runIntegrationQuery("naive") { systemText ->
                assertTrue(
                    systemText.contains("The capital of France is Paris") || systemText.contains("Paris"),
                    "Context should contain inserted text. Actual system prompt: $systemText",
                )
            }
        }
    }

    /** Confirms local mode surfaces extracted entities in the returned context/system prompt. */
    @Test
    fun testInsertAndQueryLocalMode() {
        runBlocking {
            runIntegrationQuery("local") { systemText ->
                assertTrue(
                    systemText.contains("France") || systemText.contains("Paris"),
                    "Context should contain extracted entities. Actual system prompt: $systemText",
                )
            }
        }
    }

    private suspend fun runIntegrationQuery(
        mode: String,
        verifier: (String) -> Unit,
    ) {
        val tempDir = File("build/tmp/test_rag_it_${mode}_${System.currentTimeMillis()}")
        tempDir.mkdirs()

        try {
            val (rag, messagesSlot) = buildMockedRag(tempDir)
            rag.insert("The capital of France is Paris.")
            rag.query("What is the capital of France?", QueryParam(mode = mode))

            val systemMessage = messagesSlot.captured.find { it is SystemMessage } as? SystemMessage
            val systemText = systemMessage?.text() ?: ""
            verifier(systemText)
        } finally {
            tempDir.deleteRecursively()
        }
    }

    private fun buildMockedRag(tempDir: File): Pair<LightRAG, CapturingSlot<List<ChatMessage>>> {
        val mockChatModel = mockk<ChatModel>()
        val mockEmbeddingModel = mockk<EmbeddingModel>()
        mockEmbeddings(mockEmbeddingModel)

        val messagesSlot = slot<List<ChatMessage>>()
        val responseProvider = buildResponseProvider()

        every { mockChatModel.chat(capture(messagesSlot)) } answers { responseProvider(messagesSlot.captured) }
        every { mockChatModel.chat(any<ChatRequest>()) } answers {
            val request = invocation.args[0] as ChatRequest
            responseProvider(request.messages())
        }
        every { mockChatModel.supportedCapabilities() } returns emptySet()

        return buildTestLightRag(mockChatModel, mockEmbeddingModel, tempDir) to messagesSlot
    }

    private fun mockEmbeddings(mockEmbeddingModel: EmbeddingModel) {
        every { mockEmbeddingModel.embed(any<String>()) } returns Response.from(Embedding(FloatArray(384) { 0.1f }))
        every { mockEmbeddingModel.embed(any<TextSegment>()) } returns Response.from(Embedding(FloatArray(384) { 0.1f }))
        every { mockEmbeddingModel.embedAll(any<List<TextSegment>>()) } answers {
            val input = firstArg<List<TextSegment>>()
            Response.from(input.map { Embedding(FloatArray(384) { 0.1f }) })
        }
    }

    private fun buildResponseProvider(): (List<ChatMessage>) -> ChatResponse {
        val extractionJson =
            """
            {
              "entities": [
                {"name": "France", "type": "Location", "description": "A country in Europe"},
                {"name": "Paris", "type": "Location", "description": "The capital of France"}
              ],
              "relations": [
                {"source": "France", "target": "Paris", "keywords": "capital", "description": "Paris is the capital of France"}
              ]
            }
            """.trimIndent()

        return { messages ->
            val lastMessage = messages.last()
            val lastText = if (lastMessage is UserMessage) lastMessage.singleText() ?: "" else ""
            val response = AiMessage.from(extractionJson)

            if (lastText.contains("Entity_types", ignoreCase = true)) {
                ChatResponse.builder().aiMessage(response).build()
            } else {
                ChatResponse.builder().aiMessage(response).build()
            }
        }
    }
}
