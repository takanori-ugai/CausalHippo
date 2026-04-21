package lightrag.core

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.ChatMessage
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
import lightrag.core.types.DocProcessingStatus
import lightrag.core.types.DocStatus
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/** Integration-style tests covering document ingestion and deduplication. */
class LightRAGInsertTest {
    /** Ensures a single document insert processes, populates graph nodes/edges, and marks status processed. */
    @Test
    fun testInsertSingleDocument() {
        runBlocking {
            withMockedRag { rag, tempDir ->
                val trackId = rag.insert("Apple released the new iPhone yesterday.")
                assertNotNull(trackId)

                val processedDocs: Map<String, DocProcessingStatus> =
                    rag.storageManager.docStatusStorage.getDocsByStatus(DocStatus.PROCESSED)
                assertEquals(1, processedDocs.size)

                val docId = processedDocs.keys.first()
                val docData: DocProcessingStatus? = processedDocs[docId]
                assertEquals(DocStatus.PROCESSED, docData?.status)

                val graph = rag.storageManager.chunkEntityRelationGraph
                val node = graph.getNode("Apple")
                assertNotNull(node, "Node 'Apple' should exist in graph")
                assertEquals("Organization", node["entity_type"])

                val edge = graph.getEdge("Apple", "iPhone")
                assertNotNull(edge, "Edge between 'Apple' and 'iPhone' should exist")

                tempDir.deleteRecursively()
            }
        }
    }

    /** Verifies inserting duplicate content does not create additional processed records or pending docs. */
    @Test
    fun testDuplicateDocumentInsertion() {
        runBlocking {
            try {
                // Setup
                val mockChatModel = mockk<ChatModel>(relaxed = true)
                val mockEmbeddingModel = mockk<EmbeddingModel>(relaxed = true)

                every { mockEmbeddingModel.embed(any<String>()) } returns
                    Response.from(Embedding(FloatArray(384) { 0.1f }))
                every { mockEmbeddingModel.embedAll(any<List<TextSegment>>()) } answers {
                    val input = firstArg<List<TextSegment>>()
                    Response.from(input.map { Embedding(FloatArray(384) { 0.1f }) })
                }
                val extractionJson =
                    """
                    {
                      "entities": [
                        {"name": "Doc", "type": "Concept", "description": "Test entity"}
                      ],
                      "relations": []
                    }
                    """.trimIndent()

                every { mockChatModel.chat(any<List<ChatMessage>>()) } returns
                    ChatResponse.builder().aiMessage(AiMessage.from(extractionJson)).build()
                every { mockChatModel.chat(any<ChatRequest>()) } returns
                    ChatResponse.builder().aiMessage(AiMessage.from(extractionJson)).build()
                every { mockChatModel.supportedCapabilities() } returns emptySet()

                val tempDir = File("build/tmp/test_rag_storage_dup_mockk_${System.currentTimeMillis()}")
                tempDir.mkdirs()

                val rag = buildTestLightRag(mockChatModel, mockEmbeddingModel, tempDir)

                val content = "This is a test document."

                // First Insert
                rag.insert(content)

                var processedDocs: Map<String, DocProcessingStatus> =
                    rag.storageManager.docStatusStorage.getDocsByStatus(DocStatus.PROCESSED)
                assertEquals(1, processedDocs.size)

                // Second Insert (Same content)
                rag.insert(content)

                processedDocs = rag.storageManager.docStatusStorage.getDocsByStatus(DocStatus.PROCESSED)
                assertEquals(1, processedDocs.size, "Should still be 1 document after duplicate insertion")

                val pendingDocs = rag.storageManager.docStatusStorage.getDocsByStatus(DocStatus.PENDING)
                assertEquals(0, pendingDocs.size)

                // Cleanup
                tempDir.deleteRecursively()
            } catch (e: Throwable) {
                e.printStackTrace()
                throw e
            }
        }
    }

    private suspend fun withMockedRag(block: suspend (LightRAG, File) -> Unit) {
        val tempDir = File("build/tmp/test_rag_storage_mockk_${System.currentTimeMillis()}").apply { mkdirs() }
        val (rag) = buildMockedRag(tempDir)
        try {
            block(rag, tempDir)
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

        val rag = buildTestLightRag(mockChatModel, mockEmbeddingModel, tempDir)
        return rag to messagesSlot
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
                {"name": "Apple", "type": "Organization", "description": "A tech company"},
                {"name": "iPhone", "type": "Product", "description": "A mobile phone"}
              ],
              "relations": [
                {"source": "Apple", "target": "iPhone", "keywords": "manufactures", "description": "Apple manufactures iPhone"}
              ]
            }
            """.trimIndent()

        return {
            ChatResponse.builder().aiMessage(AiMessage.from(extractionJson)).build()
        }
    }
}
