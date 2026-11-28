package lightrag.operate

import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.request.ChatRequest
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.embedding.EmbeddingModel
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.runBlocking
import lightrag.core.Constants
import lightrag.core.types.BaseVectorStorage
import lightrag.kg.memory.InMemoryGraphStorage
import lightrag.utils.computeMd5
import org.junit.Test
import shared.chunking.countTokensWithJTokKit
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class GraphMergeTest {
    @Test
    fun `oversized descriptions are reduced in bounded groups and retries reuse the merged evidence`() {
        val prompts = mutableListOf<String>()
        val model = mockk<ChatModel>()
        every { model.chat(capture(prompts)) } answers { "summary " + prompts.size }
        val merger = DescriptionMerger(mapOf("llm_model_func" to model, "force_llm_summary_on_merge" to 2, "summary_context_size" to 40))
        val fragments = (1..5).map { id -> "source$id " + "long evidence ".repeat(80) }
        val merged = merger.merge("Entity", "A", null, fragments)
        assertEquals(4, prompts.size)
        val records = Regex("""\{"Description":"[^"]*"}""")
        prompts.forEach { prompt ->
            val inputs = records.findAll(prompt).map { it.value }.toList()
            assertEquals(2, inputs.size)
            assertTrue(countTokensWithJTokKit(inputs.joinToString("\n")) <= 40)
        }
        assertEquals("summary 4", merged["description"])
        assertEquals(merged, merger.merge("Entity", "A", merged, fragments))
        assertEquals(4, prompts.size)
    }

    @Test
    fun `summaries index combined evidence once and preserve chunk provenance`() =
        runBlocking {
            val graph = InMemoryGraphStorage("graph", "test", embeddingFunc = mockk<EmbeddingModel>())
            val vectors = mockk<BaseVectorStorage>(relaxed = true)
            val model = mockk<ChatModel>()
            every { model.chat(match<String> { "first description" in it && "second description" in it }) } returns "combined evidence"
            val config = mapOf("llm_model_func" to model, "force_llm_summary_on_merge" to 2)
            for (id in listOf("first", "second", "second")) {
                val extraction =
                    ExtractionResult(
                        entities = listOf(ExtractedEntity("A", "Person", "$id description")),
                        relations = listOf(ExtractedRelation("A", "B", "knows", "$id description")),
                    )
                val (nodes, edges) = processExtractionResult(extraction, id)
                mergeNodesAndEdges(nodes, edges, graph, vectors, vectors, globalConfig = config)
            }
            assertEquals("combined evidence", graph.getNode("A")?.get("description"))
            val edge = assertNotNull(graph.getEdge("A", "B"))
            assertEquals("combined evidence", edge["description"])
            assertEquals("2.0", edge["weight"])
            assertEquals(listOf("first", "second"), edge.getValue("source_id").split(Constants.GRAPH_FIELD_SEP))
            verify(exactly = 2) { model.chat(any<String>()) }
        }

    @Test
    fun `correction pass keeps longer descriptions and newly discovered relations within their source chunk`() =
        runBlocking {
            val requests = mutableListOf<String>()
            val model =
                object : ChatModel {
                    override fun doChat(request: ChatRequest): ChatResponse {
                        val user = UserMessage.findLast(request.messages()).get().singleText()
                        requests.add(user)
                        val response =
                            if ("Previous extraction:" in user) {
                                """{"entities":[{"name":"A","type":"Place","description":"corrected description"}],
                                    "relations":[{"source":" B ","target":"A","description":"new link","keywords":"link"}]}"""
                            } else {
                                """{"entities":[{"name":" A ","type":"Place","description":"short"}],"relations":[]}"""
                            }
                        return ChatResponse.builder().aiMessage(AiMessage.from(response)).build()
                    }
                }
            val config = mapOf("llm_model_func" to model, "entity_types" to listOf("Place"))
            val chunks = mapOf("chunk" to mapOf("content" to "A is linked to B."))
            val (nodes, edges) = extractEntities(chunks, config)
            assertEquals(2, requests.size)
            assertTrue("A is linked to B." in requests.last() && "short" in requests.last())
            assertEquals("corrected description", nodes.getValue("A").single().description)
            assertEquals("chunk", edges.getValue("A#B").single().sourceId)
            for (setting in listOf("entity_extract_max_gleaning" to 0, "max_extract_input_tokens" to 1)) {
                requests.clear()
                assertTrue(extractEntities(chunks, config + setting).second.isEmpty())
                assertEquals(1, requests.size)
            }
        }

    @Test
    fun `invalid rows do not create empty nodes or self loops`() {
        val (nodes, edges) =
            processExtractionResult(
                ExtractionResult(
                    listOf(ExtractedEntity(" ", "Place", "invalid"), ExtractedEntity("A", "Place", " ")),
                    listOf(ExtractedRelation(" A ", "A", "self", "invalid"), ExtractedRelation("", "B", "link", "invalid")),
                ),
                "chunk",
            )
        assertTrue(nodes.isEmpty() && edges.isEmpty())
    }

    @Test
    fun `reversed relations share one vector ID and missing endpoints are created`() {
        runBlocking {
            val graph = InMemoryGraphStorage("graph", "test", embeddingFunc = mockk<EmbeddingModel>())
            val entities = mockk<BaseVectorStorage>(relaxed = true)
            val relations = mockk<BaseVectorStorage>(relaxed = true)
            for ((source, target, chunk) in listOf(Triple("A", "B", "first"), Triple("B", "A", "second"))) {
                val extraction =
                    ExtractionResult(relations = listOf(ExtractedRelation(source, target, "related, $chunk", "Evidence $chunk")))
                val (nodes, edges) = processExtractionResult(extraction, chunk)
                mergeNodesAndEdges(nodes, edges, graph, entities, relations, chunkFilePaths = mapOf(chunk to "$chunk.txt"))
            }

            assertEquals("UNKNOWN", graph.getNode("A")?.get("entity_type"))
            assertEquals("UNKNOWN", graph.getNode("B")?.get("entity_type"))
            val edge = assertNotNull(graph.getEdge("A", "B"))
            assertEquals(setOf("first", "second"), edge.getValue("source_id").split(Constants.GRAPH_FIELD_SEP).toSet())
            assertEquals("2.0", edge["weight"])
            assertEquals("first,related,second", edge["keywords"])
            assertEquals(listOf("first.txt", "second.txt"), edge.getValue("file_path").split(Constants.GRAPH_FIELD_SEP))
            assertEquals(edge["file_path"], graph.getNode("A")?.get("file_path"))
            assertEquals(edge["source_id"], graph.getNode("A")?.get("source_id"))
            assertEquals(edge["source_id"], graph.getNode("B")?.get("source_id"))
            assertEquals(1, graph.getAllEdges().size)
            coVerify(exactly = 2) { relations.upsert(match { it.keys == setOf(computeMd5("A#B")) }) }
            coVerify(exactly = 0) { relations.upsert(match { computeMd5("B#A") in it }) }
        }
    }
}
