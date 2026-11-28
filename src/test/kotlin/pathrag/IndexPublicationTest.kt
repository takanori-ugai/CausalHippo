package pathrag

import io.mockk.coEvery
import io.mockk.mockkStatic
import io.mockk.unmockkStatic
import kotlinx.coroutines.runBlocking
import pathrag.base.QueryParam
import pathrag.storage.NanoVectorDBStorage
import pathrag.utils.EmbeddingFunc
import shared.rag.spi.persistence.GraphSnapshot
import shared.rag.spi.persistence.GraphStore
import shared.rag.spi.persistence.InMemoryPersistenceBackend
import shared.rag.spi.persistence.PersistenceSession
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class IndexPublicationTest {
    private val document = "The evidence passage about Alpha and Beta."
    private val extraction =
        """{"entities":[],"relationships":[{"src_id":"ALPHA","tgt_id":"BETA","description":"related","keywords":"link","weight":1}]}"""

    @Test
    fun failedChunkEmbeddingDoesNotPublishGraphEvidence() =
        exercise { directory ->
            PathRAG(workingDir = directory).use { rag ->
                coEvery { pathrag.llm.openAiEmbedding(match { document in it }, any()) } throws IllegalStateException("embedding failed")
                assertFailsWith<IllegalStateException> { rag.ainsert(document) }
                assertTrue(rag.graph().nodes().isEmpty())
                assertTrue(rag.graph().edges().isEmpty())
            }
        }

    @Test
    fun retryPublishesCommittedDocumentAfterSpiWriteFails() =
        exercise { directory ->
            InMemoryPersistenceBackend().open(emptyMap()).use { backed ->
                var fail = true
                val session =
                    object : PersistenceSession by backed {
                        override fun graph(namespace: String): GraphStore {
                            val storage = backed.graph(namespace)
                            return object : GraphStore by storage {
                                override fun restore(snapshot: GraphSnapshot) {
                                    if (fail) {
                                        fail = false
                                        error("injected publication failure")
                                    }
                                    storage.restore(snapshot)
                                }
                            }
                        }
                    }
                PathRAG(workingDir = directory, persistenceSession = session, useUnifiedSpiForRetrievalAndIndex = true).use { rag ->
                    assertFailsWith<IllegalStateException> { rag.ainsert(document) }
                    rag.ainsert(document)
                    assertTrue(backed.graph("pathrag_graph").edges().isNotEmpty())
                    assertTrue(document in rag.aquery("evidence", QueryParam(onlyNeedContext = true)))
                }
            }
        }

    @Test
    fun shortBatchDoesNotPartiallyWriteVectorsAndCanBeRetried() =
        runBlocking {
            var short = true
            val embed = EmbeddingFunc(2, 100, { texts -> List(if (short) 1 else texts.size) { doubleArrayOf(1.0, 0.0) } })
            val store = NanoVectorDBStorage("test", emptyMap(), embed)
            val rows = mapOf("a" to mapOf("content" to "first"), "b" to mapOf("content" to "second"))
            assertFailsWith<IllegalStateException> { store.upsert(rows) }
            assertTrue(store.query("query", 2).isEmpty())
            short = false
            store.upsert(rows)
            assertEquals(setOf("first", "second"), store.query("query", 2).map { it["content"] }.toSet())
        }

    @Test
    fun collidingEndpointConcatenationsRetainBothRelationships() =
        exercise { directory ->
            coEvery { pathrag.llm.openAiComplete(any(), any(), any(), any(), any(), any(), any(), any(), any()) } returns
                """{"entities":[],"relationships":[
                {"src_id":"A","tgt_id":"BC","description":"first evidence","keywords":"link","weight":1},
                {"src_id":"AB","tgt_id":"C","description":"second evidence","keywords":"link","weight":1}
                ]}"""
            PathRAG(workingDir = directory).use { rag ->
                rag.ainsert(document)
                coEvery { pathrag.llm.openAiComplete(any(), any(), any(), any(), any(), any(), any(), any(), any()) } returns
                    """{"high_level_keywords":["link"],"low_level_keywords":[]}"""
                val query = QueryParam(mode = "global", onlyNeedContext = true)
                val original = rag.aquery("link", query)
                assertTrue("first evidence" in original)
                assertTrue("second evidence" in original)
                rag.adeleteEdge("A", "BC")
                val retained = rag.aquery("link", query)
                assertFalse("first evidence" in retained)
                assertTrue("second evidence" in retained)
            }
        }

    private fun exercise(test: suspend (String) -> Unit) =
        runBlocking {
            val dir = Files.createTempDirectory("pathrag-publication")
            mockkStatic("pathrag.llm.LlmKt")
            try {
                coEvery { pathrag.llm.openAiComplete(any(), any(), any(), any(), any(), any(), any(), any(), any()) } returns extraction
                coEvery { pathrag.llm.openAiEmbedding(any(), any()) } answers {
                    firstArg<List<String>>().map { DoubleArray(1536) { 0.02 } }
                }
                test(dir.toString())
            } finally {
                unmockkStatic("pathrag.llm.LlmKt")
                dir.toFile().deleteRecursively()
            }
        }
}
