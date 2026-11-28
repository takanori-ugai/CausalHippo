package hipporag

import hipporag.config.BaseConfig
import hipporag.embeddingmodel.BaseEmbeddingModel
import hipporag.embeddingmodel.getEmbeddingModel
import hipporag.llm.BaseLLM
import hipporag.llm.getLlm
import hipporag.utils.LlmResult
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkStatic
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import shared.rag.unified.HippoRagUnifiedAdapter
import shared.rag.unified.UnifiedMode
import shared.rag.unified.UnifiedQuery
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertTrue

class IndexIntegrityTest {
    private val first = "Alpha connects to Beta."
    private val second = "Alpha connects to Gamma."

    @Test
    fun rejectsOverlappingSnapshotAndPreservesIndex() =
        withRag { config, _ ->
            val rag = HippoRAG(config)
            rag.upsert(first)
            val working = Path.of(config.saveDir, "${config.llmName}_${config.embeddingModelName}")
            assertFails { rag.saveGraph(working.toString()) }
            assertTrue(Files.isRegularFile(working.resolve("graph.json")))
            assertFails { rag.loadGraph(working.toString()) }
            assertEquals(listOf(first), HippoRAG(config).query("query", QueryParam("dpr", includeAnswer = false)).docs)
        }

    @Test
    fun rejectsIncompleteSnapshotAndPreservesIndex() =
        withRag { config, _ ->
            val rag = HippoRAG(config)
            rag.upsert(first)
            val snapshot = Files.createDirectories(Path.of(config.saveDir, "invalid", "working_dir")).parent
            assertFails { rag.loadGraph(snapshot.toString()) }
            assertEquals(listOf(first), HippoRAG(config).query("query", QueryParam("dpr", includeAnswer = false)).docs)
        }

    @Test
    fun restoresValidSnapshotAndRejectsCorruptEmbeddings() =
        withRag { config, _ ->
            val rag = HippoRAG(config)
            rag.upsert(first)
            val snapshot = Path.of(config.saveDir, "snapshot")
            rag.saveGraph(snapshot.toString())
            rag.upsert(second)
            rag.loadGraph(snapshot.toString())
            val query = QueryParam("dpr", includeAnswer = false)
            assertEquals(listOf(first), rag.query("query", query).docs)

            Files.writeString(snapshot.resolve("working_dir/chunk_embeddings/vdb_chunk.json"), "invalid")
            assertFails { rag.loadGraph(snapshot.toString()) }
            assertEquals(listOf(first), HippoRAG(config).query("query", query).docs)
        }

    @Test
    fun retainsSharedEntityWhenDeletingAfterRestart() =
        withRag { config, _ ->
            HippoRag(config).index(listOf(first, second))
            val restarted = HippoRag(config)
            restarted.delete(listOf(first))
            val graph = HippoRAG(restarted).inspectGraph()
            val nodes = graph["nodes"] as List<*>
            assertTrue(nodes.any { (it as Map<*, *>)["content"] == "alpha" })
            assertEquals(3, nodes.size)
        }

    @Test
    fun embeddingFailureKeepsCommittedIndexAndCanBeRetried() =
        withRag { config, embedding ->
            val rag = HippoRAG(config)
            rag.upsert(first)
            every { embedding.batchEncode(match { it.any { text -> text.startsWith("[") } }, any(), any()) } throws
                IllegalStateException("fact encoder down")
            assertFails { rag.upsert(second) }
            assertEquals(listOf(first), rag.query("query", QueryParam("dpr", includeAnswer = false)).docs)
            assertEquals(listOf(first), HippoRAG(config).query("query", QueryParam("dpr", includeAnswer = false)).docs)
            every { embedding.batchEncode(any(), any(), any()) } answers { vectors(firstArg()) }
            rag.upsert(second)
            assertEquals(setOf(first, second), rag.query("query", QueryParam("dpr", includeAnswer = false)).docs.toSet())
            assertEquals(5, (rag.inspectGraph()["nodes"] as List<*>).size)
        }

    @Test
    fun rebuildsMissingPassageEdgesOnRetry() =
        withRag { config, _ ->
            HippoRag(config).index(listOf(first))
            val graph = Path.of(config.saveDir, "test_test", "graph.json")
            val original = Json.parseToJsonElement(Files.readString(graph)) as JsonObject
            Files.writeString(graph, JsonObject(original + ("edges" to JsonArray(emptyList()))).toString())

            val rag = HippoRag(config)
            rag.index(listOf(first))
            assertTrue((HippoRAG(rag).inspectGraph()["edges"] as List<*>).isNotEmpty())
        }

    @Test
    fun reportsOnlyPassagesIncludedInAnswerPrompt() =
        withRag { config, _ ->
            val rag = HippoRag(config.copy(qaTopK = 1))
            rag.index(listOf(first, second))
            val adapter = HippoRagUnifiedAdapter(HippoRAG(rag))
            val query = UnifiedQuery(mode = UnifiedMode.DPR, topK = 2)
            val response = adapter.query("query", query)
            assertEquals(1, response.context.size)
            assertEquals(1, response.references.size)
            assertEquals(2, adapter.query("query", query.copy(includeAnswer = false)).context.size)
        }

    @Test
    fun invalidatesRetrievalAfterIncrementalIndex() =
        withRag { config, _ ->
            val rag = HippoRag(config)
            rag.index(listOf(first))
            rag.retrieveDpr(listOf("query"))
            rag.index(listOf(second))
            assertEquals(
                setOf(first, second),
                rag
                    .retrieveDpr(listOf("query"))
                    .first
                    .single()
                    .docs
                    .toSet(),
            )
        }

    @Test
    fun retriesFailedOpenIeForAnExistingChunk() =
        withRag { config, _ ->
            var fail = true
            every { getLlm(any()) } returns
                object : BaseLLM {
                    override fun infer(messages: List<hipporag.utils.Message>): LlmResult {
                        check(!fail) { "OpenIE temporarily unavailable" }
                        return LlmResult("""{"named_entities":["Alpha","Beta"],"triples":[["Alpha","connects","Beta"]]}""", emptyMap())
                    }
                }
            val rag = HippoRag(config)
            rag.index(listOf(first))
            assertEquals(1, (HippoRAG(rag).inspectGraph()["nodes"] as List<*>).size)
            fail = false
            rag.index(listOf(first))
            assertEquals(3, (HippoRAG(rag).inspectGraph()["nodes"] as List<*>).size)
        }

    private fun vectors(texts: List<String>): Array<DoubleArray> = Array(texts.size) { doubleArrayOf(1.0, 0.0) }

    private fun withRag(test: (BaseConfig, BaseEmbeddingModel) -> Unit) {
        val dir = Files.createTempDirectory("hipporag-integrity")
        mockkStatic(::getEmbeddingModel, ::getLlm)
        try {
            val embedding = mockk<BaseEmbeddingModel>()
            every { embedding.batchEncode(any(), any(), any()) } answers { vectors(firstArg()) }
            every { getEmbeddingModel() } returns mockk { every { create(any(), any()) } returns embedding }
            val llm = mockk<BaseLLM>()
            every { llm.infer(any()) } answers {
                val messages = firstArg<List<hipporag.utils.Message>>()
                val prompt = messages.last().content
                val target = if ("Gamma" in prompt) "Gamma" else "Beta"
                val reply =
                    when {
                        "Question:" in prompt -> "Answer: Alpha"
                        else -> """{"named_entities":["Alpha","$target"],"triples":[["Alpha","connects","$target"]]}"""
                    }
                LlmResult(reply, emptyMap())
            }
            every { getLlm(any()) } returns llm
            test(BaseConfig(saveDir = dir.toString(), llmName = "test", embeddingModelName = "test"), embedding)
        } finally {
            unmockkStatic(::getEmbeddingModel, ::getLlm)
            dir.toFile().deleteRecursively()
        }
    }
}
