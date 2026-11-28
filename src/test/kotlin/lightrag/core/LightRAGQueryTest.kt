package lightrag.core

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import lightrag.core.types.BaseGraphStorage
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager
import org.junit.Test
import shared.rag.spi.persistence.InMemoryPersistenceBackend
import kotlin.test.assertEquals
import kotlin.test.assertSame

class LightRAGQueryTest {
    private val ingestionService = mockk<IngestionService>(relaxed = true)
    private val queryService = mockk<QueryService>(relaxed = true)
    private val storageManager = mockk<StorageManager>(relaxed = true)

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
