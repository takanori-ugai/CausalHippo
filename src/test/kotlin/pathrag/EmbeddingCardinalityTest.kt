package pathrag

import kotlinx.coroutines.runBlocking
import org.junit.Test
import pathrag.storage.NanoVectorDBStorage
import pathrag.utils.EmbeddingFunc
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class EmbeddingCardinalityTest {
    @Test
    fun `short embedding response is rejected before upsert writes any vectors`() {
        runBlocking {
            val embed = EmbeddingFunc(embeddingDim = 1, maxTokenSize = 100, func = { listOf(doubleArrayOf(1.0)) })
            val storage = NanoVectorDBStorage("chunks", emptyMap(), embed)

            val failure =
                assertFailsWith<IllegalStateException> {
                    storage.upsert(mapOf("a" to mapOf("content" to "alpha"), "b" to mapOf("content" to "beta")))
                }

            assertEquals("Expected 2 embeddings, received 1", failure.message)
            assertTrue(storage.query("alpha", 2).isEmpty())
        }
    }
}
