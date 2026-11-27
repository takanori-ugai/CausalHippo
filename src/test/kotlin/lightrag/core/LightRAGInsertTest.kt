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
import lightrag.utils.computeMd5
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

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

    @Test
    fun `reinsert updates file path without losing document content`() {
        runBlocking {
            withMockedRag { rag, _ ->
                val document = "Apple released the new iPhone yesterday."
                val id = computeMd5(document)
                rag.insert(document, "old.txt")
                rag.insert(document, "new.txt")

                assertEquals(
                    document,
                    rag.storageManager.fullDocs
                        .getById(id)
                        ?.get("content"),
                )
                assertEquals(
                    "new.txt",
                    rag.storageManager.textChunks
                        .getById(id)
                        ?.get("file_path"),
                )
                assertEquals(
                    setOf(id),
                    rag.storageManager.docStatusStorage
                        .getDocsByStatus(DocStatus.PROCESSED)
                        .keys,
                )
                assertEquals(
                    "1.0",
                    rag.storageManager.chunkEntityRelationGraph
                        .getEdge("Apple", "iPhone")
                        ?.get("weight"),
                )
            }
        }
    }

    @Test
    fun `delete removes document evidence and preserves shared nodes and edges`() {
        runBlocking {
            withMockedRag { rag, _ ->
                val first = "Apple released an iPhone yesterday."
                val second = "Apple builds the iPhone."
                val firstId = computeMd5(first)
                val secondId = computeMd5(second)
                rag.insert(first)
                rag.insert(second)
                val stores = rag.storageManager
                val graph = stores.chunkEntityRelationGraph

                assertEquals(
                    setOf(firstId, secondId),
                    graph
                        .getNode("Apple")
                        ?.get("source_id")
                        ?.split(Constants.GRAPH_FIELD_SEP)
                        ?.toSet(),
                )
                assertEquals("2.0", graph.getEdge("Apple", "iPhone")?.get("weight"))

                assertEquals("success", rag.deleteByDocId(firstId)["status"])
                assertEquals("not_found", rag.deleteByDocId(firstId)["status"])
                assertNull(stores.fullDocs.getById(firstId))
                assertNull(stores.textChunks.getById(firstId))
                assertNull(stores.chunksVdb.getById(firstId))
                assertEquals(secondId, graph.getNode("Apple")?.get("source_id"))
                assertEquals(secondId, graph.getEdge("Apple", "iPhone")?.get("source_id"))
                assertEquals("1.0", graph.getEdge("Apple", "iPhone")?.get("weight"))
                assertEquals(second, stores.textChunks.getById(secondId)?.get("content"))
            }
        }
    }

    @Test
    fun `failed extraction marks the document failed and does not publish chunks`() {
        runBlocking {
            val tempDir =
                kotlin.io.path
                    .createTempDirectory("lightrag-failed-extraction")
                    .toFile()
            val model = mockk<ChatModel>(relaxed = true)
            every { model.chat(any<ChatRequest>()) } throws IllegalStateException("provider unavailable")
            every { model.chat(any<List<ChatMessage>>()) } throws IllegalStateException("provider unavailable")
            every { model.supportedCapabilities() } returns emptySet()
            val embeddings = mockk<EmbeddingModel>().also(::mockEmbeddings)
            try {
                val rag = buildTestLightRag(model, embeddings, tempDir)
                val document = "This extraction should fail."
                val docId = computeMd5(document)
                rag.insert(document)
                val stores = rag.storageManager

                assertEquals(setOf(docId), stores.docStatusStorage.getDocsByStatus(DocStatus.FAILED).keys)
                assertEquals(emptySet(), stores.docStatusStorage.getDocsByStatus(DocStatus.PROCESSED).keys)
                assertNull(stores.textChunks.getById(docId))
                assertNull(stores.chunksVdb.getById(docId))
            } finally {
                tempDir.deleteRecursively()
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
        val mockChatModel = mockk<ChatModel>(relaxed = true)
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
