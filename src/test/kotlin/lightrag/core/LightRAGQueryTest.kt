package lightrag.core

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import lightrag.TestEmbeddings
import lightrag.core.types.BaseGraphStorage
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager
import org.junit.Test
import shared.rag.spi.persistence.InMemoryPersistenceBackend
import shared.rag.spi.persistence.VectorRecord
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

class LightRAGQueryTest {
    private val ingestionService = mockk<IngestionService>(relaxed = true)
    private val queryService = mockk<QueryService>(relaxed = true)
    private val storageManager = mockk<StorageManager>(relaxed = true)

    @Test
    fun `SPI local search follows graph hits and observes published changes`() =
        runBlocking {
            InMemoryPersistenceBackend().open().use { session ->
                val model = TestEmbeddings.mockEmbeddingModel(default = listOf(1f, 0f))
                val stores = StorageManager(".", model, globalConfig = emptyMap())
                val queries = QueryService(stores, mockk(), null, emptyMap(), { emptyList() }, { "" })
                val rag = LightRAG(ingestionService, queries, stores, model, session, true)
                val graph = session.graph("lightrag_graph")
                graph.upsertNode("Alice", mapOf("source_id" to "c", "description" to "graph descriptor"))
                session.kv("lightrag_chunks").putAll(
                    mapOf(
                        "c" to mapOf("content" to "linked evidence"),
                        "unused" to mapOf("content" to "question"),
                    ),
                )
                session.vector("lightrag_vector", 2).upsert(
                    VectorRecord("entity:Alice", listOf(1.0, 0.0), mapOf("kind" to "entity", "entity_name" to "Alice")),
                )
                graph.upsertNode("Bob", mapOf("description" to "other endpoint"))
                graph.upsertEdge("Alice", "Bob", mapOf("description" to "published relation"))
                session.vector("lightrag_vector", 2).upsert(
                    listOf(
                        VectorRecord("edge", listOf(1.0, 0.0), mapOf("kind" to "relation", "src_id" to "Alice", "tgt_id" to "Bob")),
                        VectorRecord(
                            "chunk:c",
                            listOf(1.0, 0.0),
                            mapOf("kind" to "chunk", "chunk_id" to "c", "content" to "linked evidence"),
                        ),
                    ),
                )
                val naive = rag.aquery("question", QueryParam(mode = "naive", onlyNeedContext = true, chunkTopK = 1))
                assertEquals("question", naive?.content)
                val global = rag.aquery("question", QueryParam(mode = "global", onlyNeedContext = true, hlKeywords = listOf("topic")))
                assertTrue(global?.content.orEmpty().contains("published relation"))
                assertTrue(global?.content.orEmpty().contains("other endpoint"))
                val param = QueryParam(mode = "local", llKeywords = listOf("term"), onlyNeedContext = true)
                val context = rag.aquery("question", param)?.content.orEmpty()
                assertTrue(context.contains("linked evidence"))
                assertTrue(context.contains("graph descriptor"))
                assertFalse(context.contains("question"))
                graph.upsertNode("Alice", mapOf("source_id" to "unused", "description" to "updated"))
                val updated = rag.aquery("question", param)?.content.orEmpty()
                assertTrue(updated.contains("updated"))
                assertFalse(updated.contains("linked evidence"))
            }
        }

    @Test
    fun `query delegates to QueryService`() =
        runBlocking {
            val expected = QueryResult(content = "hello")
            coEvery { queryService.query(any(), any()) } returns expected

            val rag = LightRAG(ingestionService, queryService, storageManager)
            val result = rag.query("Hi", QueryParam(mode = "naive"))

            assertSame(expected, result)
            coVerify { queryService.query("Hi", any()) }
        }

    @Test
    fun `insert delegates to IngestionService`() =
        runBlocking {
            coEvery { ingestionService.insert(any<String>(), any()) } returns "track-id"

            val rag = LightRAG(ingestionService, queryService, storageManager)
            val trackId = rag.insert("text")

            assertEquals("track-id", trackId)
            coVerify { ingestionService.insert("text", null) }
        }

    @Test
    fun `deletion cannot overtake an in-progress unified index publication`() {
        runBlocking {
            withTimeout(5_000) {
                val graph = mockk<BaseGraphStorage>(relaxed = true)
                coEvery { storageManager.chunkEntityRelationGraph } returns graph
                coEvery { storageManager.textChunks.isEmpty() } returns true
                var pauseNextRead = false
                var deleted = false
                val publicationStarted = CompletableDeferred<Unit>()
                val resumePublication = CompletableDeferred<Unit>()
                coEvery { graph.getAllNodes() } coAnswers {
                    if (pauseNextRead) {
                        pauseNextRead = false
                        publicationStarted.complete(Unit)
                        resumePublication.await()
                    }
                    if (deleted) emptyList() else listOf(mapOf("entity_id" to "OLD"))
                }
                coEvery { ingestionService.insert(any<String>(), any()) } returns "track-id"
                coEvery { ingestionService.deleteByDocId("doc") } coAnswers {
                    deleted = true
                    mapOf("status" to "success")
                }

                InMemoryPersistenceBackend().open().use { session ->
                    val rag =
                        LightRAG(
                            ingestionService,
                            queryService,
                            storageManager,
                            persistenceSession = session,
                            useUnifiedSpiForRetrievalAndIndex = true,
                        )
                    pauseNextRead = true
                    val upsert = async { rag.aupsert("document") }
                    publicationStarted.await()
                    val deletion = async(start = CoroutineStart.UNDISPATCHED) { rag.deleteByDocId("doc") }
                    coVerify(exactly = 0) { ingestionService.deleteByDocId("doc") }
                    resumePublication.complete(Unit)

                    upsert.await()
                    assertEquals("success", deletion.await()["status"])
                    assertEquals(emptyList(), session.graph("lightrag_graph").snapshot().nodes)
                }
            }
        }
    }
}
