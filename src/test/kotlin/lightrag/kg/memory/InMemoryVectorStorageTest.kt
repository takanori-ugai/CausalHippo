package lightrag.kg.memory

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.output.Response
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.runBlocking
import lightrag.TestEmbeddings
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Tests for in-memory vector storage covering upsert, embedding, persistence, and deletion. */
class InMemoryVectorStorageTest {
    /** Confirms explicit vectors support similarity querying. */
    @Test
    fun `upsert with provided vector supports similarity query`() {
        runBlocking {
            val tempDir = File("build/tmp/vector_storage_query_${System.currentTimeMillis()}").apply { mkdirs() }
            try {
                val embeddingModel =
                    TestEmbeddings.mockEmbeddingModel(
                        vectors =
                            mapOf(
                                "first" to listOf(1.0f, 0.0f),
                                "second" to listOf(0.0f, 1.0f),
                            ),
                        default = listOf(0.0f, 0.0f),
                    )
                val storage =
                    InMemoryVectorStorage(
                        namespace = "ns",
                        workspace = "ws",
                        globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                        embeddingFunc = embeddingModel,
                    )

                val data =
                    mapOf(
                        "id-1" to mapOf("vector" to listOf(1.0f, 0.0f), "content" to "first"),
                        "id-2" to mapOf("vector" to listOf(0.0f, 1.0f), "content" to "second"),
                    )

                storage.upsert(data)

                val results = storage.query("ignore", topK = 1, queryEmbedding = listOf(1.0f, 0.0f))

                assertEquals(1, results.size)
                assertEquals("id-1", results.first()["id"])
                val score = results.first()["score"] as Double
                assertEquals(1.0, score, 1e-6)
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }

    /** Ensures content-only entries trigger embedding model usage during upsert. */
    @Test
    fun `content is embedded when embedding model provided`() {
        runBlocking {
            val mockEmbeddingModel = mockk<EmbeddingModel>()
            every { mockEmbeddingModel.embed("hello world") } returns Response.from(Embedding(floatArrayOf(0.2f, 0.2f)))

            val storage =
                InMemoryVectorStorage(
                    namespace = "ns",
                    workspace = "ws",
                    globalConfig = emptyMap(),
                    embeddingFunc = mockEmbeddingModel,
                )

            storage.upsert(mapOf("doc-1" to mapOf("content" to "hello world")))

            val vectors = storage.getVectorsByIds(listOf("doc-1"))
            assertEquals(listOf(0.2f, 0.2f), vectors["doc-1"])
            verify(exactly = 1) { mockEmbeddingModel.embed("hello world") }
        }
    }

    /** Verifies vectors and metadata persist across initialize/indexDoneCallback. */
    @Test
    fun `vectors and metadata persist across initialize`() {
        runBlocking {
            val tempDir = File("build/tmp/vector_storage_persist_${System.currentTimeMillis()}").apply { mkdirs() }
            try {
                val embeddingModel =
                    TestEmbeddings.mockEmbeddingModel(
                        vectors = mapOf("persist me" to listOf(0.3f, 0.4f)),
                        default = listOf(0.0f, 0.0f),
                    )
                val initial =
                    InMemoryVectorStorage(
                        namespace = "persist-ns",
                        workspace = "ws",
                        globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                        embeddingFunc = embeddingModel,
                    )

                initial.upsert(
                    mapOf(
                        "persist-id" to mapOf("vector" to listOf(0.3f, 0.4f), "content" to "persist me"),
                    ),
                )
                initial.indexDoneCallback()

                val reloaded =
                    InMemoryVectorStorage(
                        namespace = "persist-ns",
                        workspace = "ws",
                        globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                        embeddingFunc = embeddingModel,
                    )

                reloaded.initialize()

                val loadedMeta = reloaded.getById("persist-id")
                assertNotNull(loadedMeta)
                assertEquals("persist me", loadedMeta["content"])

                val vectors = reloaded.getVectorsByIds(listOf("persist-id"))
                assertEquals(listOf(0.3f, 0.4f), vectors["persist-id"])

                val results = reloaded.query("any", topK = 1, queryEmbedding = listOf(0.3f, 0.4f))
                assertEquals("persist-id", results.first()["id"])
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }

    @Test
    fun `delete functions remove matching entities and relations`() {
        runBlocking {
            val embeddingModel = TestEmbeddings.mockEmbeddingModel()
            val storage =
                InMemoryVectorStorage(
                    namespace = "delete-ns",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

            val data =
                mapOf(
                    "entity-1" to mapOf("vector" to listOf(1.0f, 1.0f), "entity_name" to "Alice"),
                    "relation-1" to mapOf("vector" to listOf(0.5f, 0.5f), "src_id" to "Alice", "tgt_id" to "Bob"),
                    "relation-2" to mapOf("vector" to listOf(0.2f, 0.9f), "src_id" to "Charlie", "tgt_id" to "Alice"),
                )

            storage.upsert(data)

            storage.deleteEntity("Alice")
            assertNull(storage.getById("entity-1"))
            assertTrue(storage.getVectorsByIds(listOf("entity-1")).isEmpty())

            storage.deleteEntityRelation("Alice")
            assertTrue(storage.getByIds(listOf("relation-1", "relation-2")).isEmpty())
            assertTrue(storage.getVectorsByIds(listOf("relation-1", "relation-2")).isEmpty())
        }
    }
}
