package lightrag.core

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager
import org.junit.Test
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
}
