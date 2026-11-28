package lightrag.operate

import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.request.ChatRequest
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import lightrag.TestEmbeddings
import lightrag.core.Constants
import lightrag.core.QueryParam
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class QueryProcessorTest {
    private val graph = mockk<BaseGraphStorage>(relaxed = true)
    private val entities = mockk<BaseVectorStorage>(relaxed = true)
    private val relations = mockk<BaseVectorStorage>(relaxed = true)
    private val textChunks = mockk<BaseKVStorage>(relaxed = true)
    private val chat = mockk<ChatModel>(relaxed = true)
    private val processor = QueryProcessor(graph, entities, relations, textChunks, chat, null, emptyMap())

    @Test
    fun `related chunks use cosine order and graph support when vectors are incomplete`() =
        runBlocking {
            val rows =
                listOf(listOf("first", "shared", "best"), listOf("shared", "other")).map {
                    mapOf("source_id" to it.joinToString(Constants.GRAPH_FIELD_SEP))
                }
            val vectors = mockk<BaseVectorStorage>()
            every { vectors.embeddingFunc } returns TestEmbeddings.mockEmbeddingModel(default = listOf(1f, 0f))
            val stored = mapOf("first" to listOf(0f, 1f), "shared" to listOf(0f, 1f), "best" to listOf(1f, 0f), "other" to listOf(1f, 0f))
            coEvery { vectors.getVectorsByIds(any()) } returns stored
            val selector = RelatedChunkSelector("question", vectors, mapOf("related_chunk_number" to 2))
            assertEquals(listOf("best", "other"), selector.select(rows))
            coEvery { vectors.getVectorsByIds(any()) } returns stored - "best"
            assertEquals(listOf("shared", "first", "other"), selector.select(rows))
            assertEquals(listOf("first", "best", "other"), selector.select(rows, setOf("shared")))
            assertTrue(RelatedChunkSelector("question", vectors, mapOf("related_chunk_number" to 0)).select(rows).isEmpty())
        }

    @Test
    fun `mix includes vector only and graph only chunks without duplicates`() {
        runBlocking {
            val vectors = mockk<BaseVectorStorage>()
            coEvery { vectors.getVectorsByIds(any()) } returns emptyMap()
            val topK = 20
            coEvery { vectors.query("question", topK) } returns
                listOf(mapOf("id" to "vector", "content" to "vector only"), mapOf("id" to "graph", "content" to "graph only"))
            coEvery { entities.query("term", any()) } returns listOf(mapOf("entity_name" to "Entity"))
            coEvery { graph.getNodesBatch(listOf("Entity")) } returns mapOf("Entity" to mapOf("source_id" to "graph"))
            coEvery { textChunks.getById("graph") } returns mapOf("content" to "graph only")
            val params = QueryParam(mode = "mix", onlyNeedContext = true, llKeywords = listOf("term"), chunkTopK = topK)

            val result = assertNotNull(processor.kgQuery("question", params, chunksVdb = vectors))
            val chunks = result.rawData?.get("chunks") as List<*>
            assertEquals(listOf("vector", "graph"), chunks.map { (it as Map<*, *>)["id"] })
        }
    }

    @Test
    fun `hybrid retains ranked local and global relationships and their evidence`() {
        runBlocking {
            coEvery { entities.query("term", any()) } returns listOf(mapOf("entity_name" to "A"))
            coEvery { graph.getNodesBatch(listOf("A")) } returns mapOf("A" to emptyMap())
            val local = listOf("A" to "B", "A" to "C")
            coEvery { graph.getNodesEdgesBatch(listOf("A")) } returns mapOf("A" to local)
            coEvery { relations.query("theme", any()) } returns
                listOf(mapOf("src_id" to "Y", "tgt_id" to "X"), mapOf("src_id" to "B", "tgt_id" to "A"))
            coEvery { graph.getEdgesBatch(any()) } returns
                (local + listOf("Y" to "X", "B" to "A")).associateWith { (_, target) -> mapOf("source_id" to target) }
            coEvery { textChunks.getById(any()) } answers { mapOf("content" to firstArg<String>()) }
            val params = QueryParam(mode = "hybrid", onlyNeedContext = true, llKeywords = listOf("term"), hlKeywords = listOf("theme"))
            val capped = assertNotNull(processor.kgQuery("question", params.copy(topK = 1, chunkTopK = 1)))
            assertEquals(3, (capped.rawData?.get("relations") as List<*>).size)
            assertEquals(1, (capped.rawData?.get("chunks") as List<*>).size)
            assertEquals("1", capped.referenceList.single()["reference_id"])
            val data = assertNotNull(processor.kgQuery("question", params)?.rawData)
            val edges = (data["relations"] as List<*>).map { it as Map<*, *> }
            assertEquals(listOf("A" to "B", "Y" to "X", "A" to "C"), edges.map { it["src_id"] to it["tgt_id"] })
            assertEquals(listOf("B", "X", "C"), (data["chunks"] as List<*>).map { (it as Map<*, *>)["id"] })

            val limited = assertNotNull(processor.kgQuery("question", params.copy(maxRelationTokens = 1))?.rawData)
            assertEquals(emptyList<Any>(), limited["relations"])
            assertEquals(emptyList<Any>(), limited["chunks"])
            val tiny = assertNotNull(processor.kgQuery("question", params.copy(maxTotalTokens = 1)))
            assertEquals("", tiny.content)
        }
    }

    @Test
    fun `single-level search uses available keywords when its preferred level is missing`() =
        runBlocking {
            processor.kgQuery("question", QueryParam(mode = "global", onlyNeedContext = true, llKeywords = listOf("name")))
            coVerify(exactly = 1) { entities.query("name", any()) }
            processor.kgQuery("question", QueryParam(mode = "local", onlyNeedContext = true, hlKeywords = listOf("topic")))
            coVerify(exactly = 1) { relations.query("topic", any()) }
            val longQuery = "q".repeat(50)
            val result = processor.kgQuery(longQuery, QueryParam(mode = "global", onlyNeedContext = true, hlKeywords = listOf(" ")))
            assertEquals("", result?.content)
            coVerify(exactly = 0) { entities.query(longQuery, any()) }
        }

    @Test
    fun `reusing query params does not retain generated fallback keywords`() {
        runBlocking {
            every { chat.supportedCapabilities() } returns emptySet()
            every { chat.chat(any<ChatRequest>()) } throws IllegalStateException("keyword service unavailable")
            every { chat.chat(any<List<ChatMessage>>()) } throws IllegalStateException("keyword service unavailable")
            val params = QueryParam(mode = "local", onlyNeedContext = true)

            processor.kgQuery("first", params)
            processor.kgQuery("second", params)

            coVerify(exactly = 1) { entities.query("first", any()) }
            coVerify(exactly = 1) { entities.query("second", any()) }
            assertTrue(params.llKeywords.isEmpty())
        }
    }
}
