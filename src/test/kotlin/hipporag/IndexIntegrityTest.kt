package hipporag

import hipporag.config.BaseConfig
import hipporag.embeddingmodel.BaseEmbeddingModel
import hipporag.embeddingmodel.EmbeddingModelFactory
import hipporag.embeddingmodel.getEmbeddingModel
import hipporag.llm.BaseLLM
import hipporag.llm.getLlm
import hipporag.utils.LlmResult
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkStatic
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
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
    fun preservesUnsavedProvenanceInCheckpoint() =
        withRag { base, _ ->
            val config = base.copy(saveOpenie = false)
            val original = HippoRag(config)
            original.index(listOf(first, second))
            val snapshot = Path.of(config.saveDir, "provenance")
            HippoRAG(original).saveGraph(snapshot.toString())
            HippoRAG(config).loadGraph(snapshot.toString())
            val restored = HippoRag(config)
            restored.delete(listOf(first))
            assertEquals(listOf(second), HippoRAG(restored).query("query", QueryParam("dpr", includeAnswer = false)).docs)
        }

    @Test
    fun incompatibleRestorePreservesLiveIndex() =
        withRag { config, encoder ->
            val original = HippoRAG(config)
            original.upsert(first)
            val snapshot = Path.of(config.saveDir, "dimensions")
            original.saveGraph(snapshot.toString())
            every { encoder.batchEncode(any(), any(), any()) } answers {
                Array(firstArg<List<String>>().size) { doubleArrayOf(1.0, 0.0, 0.0) }
            }
            val live = HippoRAG(config.copy(saveDir = Path.of(config.saveDir, "other").toString()))
            live.upsert(second)
            assertFails { live.loadGraph(snapshot.toString()) }
            assertEquals(listOf(second), live.query("query", QueryParam("dpr", includeAnswer = false)).docs)
        }

    @Test
    fun initializationFailureAfterSnapshotPublicationRestoresLiveFiles() =
        withRag { config, encoder ->
            val rag = HippoRAG(config)
            rag.upsert(first)
            val snapshot = Path.of(config.saveDir, "rollback")
            rag.saveGraph(snapshot.toString())
            rag.upsert(second)
            val before = rag.inspectGraph()
            val originalFactory = getEmbeddingModel()
            val failingFactory = mockk<EmbeddingModelFactory>()
            every { failingFactory.create(any(), any()) } throws IllegalStateException("injected initialization failure")
            every { getEmbeddingModel() } returns failingFactory

            val failure = assertFails { rag.loadGraph(snapshot.toString()) }
            assertEquals("injected initialization failure", failure.message)
            assertEquals(before, rag.inspectGraph())
            assertEquals(setOf(first, second), rag.query("query", QueryParam("dpr", includeAnswer = false)).docs.toSet())

            every { originalFactory.create(any(), any()) } returns encoder
            every { getEmbeddingModel() } returns originalFactory
            assertEquals(before, HippoRAG(config).inspectGraph())
            rag.loadGraph(snapshot.toString())
            assertEquals(listOf(first), rag.query("query", QueryParam("dpr", includeAnswer = false)).docs)
        }

    @Test
    fun deletingUnknownThenLastDocumentKeepsStoresConsistent() =
        withRag { config, _ ->
            val rag = HippoRag(config)
            rag.index(listOf(first))
            rag.delete(listOf("missing document"))
            assertEquals(listOf(first), HippoRAG(rag).query("query", QueryParam("dpr", includeAnswer = false)).docs)

            rag.delete(listOf(first))
            val openie = Path.of(config.saveDir, "openie_results_ner_${config.llmName}.json")
            assertTrue(Files.notExists(openie))
            assertTrue(HippoRAG(config).query("query", QueryParam("dpr", includeAnswer = false)).docs.isEmpty())
        }

    @Test
    fun rejectsMismatchedEmbeddingStoresBeforePublishing() =
        withRag { config, _ ->
            val rag = HippoRAG(config)
            rag.upsert(first)
            val snapshot = Path.of(config.saveDir, "mixed-dimensions")
            rag.saveGraph(snapshot.toString())
            val file = snapshot.resolve("working_dir/entity_embeddings/vdb_entity.json")
            val data = Json.parseToJsonElement(Files.readString(file)) as JsonObject
            val rows = data.getValue("embeddings") as JsonArray
            val mismatched =
                JsonObject(data + ("embeddings" to JsonArray(rows.map { vector -> JsonArray((vector as JsonArray).take(1)) })))
            Files.writeString(file, mismatched.toString())
            val error = assertFails { rag.loadGraph(snapshot.toString()) }
            assertTrue(error.message.orEmpty().contains("inconsistent embedding dimensions"))
            assertEquals(listOf(first), rag.query("query", QueryParam("dpr", includeAnswer = false)).docs)
        }

    @Test
    fun rejectsGraphSnapshotsWithUnnamedVerticesOrDanglingEdges() =
        withRag { config, _ ->
            val rag = HippoRAG(config)
            rag.upsert(first)
            val snapshot = Path.of(config.saveDir, "invalid-graph")
            rag.saveGraph(snapshot.toString())
            val file = snapshot.resolve("working_dir/graph.json")
            val graph = Json.parseToJsonElement(Files.readString(file)) as JsonObject
            val vertices = graph.getValue("vertices") as JsonArray
            val edges = graph.getValue("edges") as JsonArray
            val unnamed = JsonArray(listOf(JsonObject(emptyMap())) + vertices.drop(1))
            val dangling = JsonArray(listOf(JsonObject((edges.first() as JsonObject) + ("target" to JsonPrimitive(999)))) + edges.drop(1))

            Files.writeString(file, JsonObject(graph + ("vertices" to unnamed)).toString())
            assertTrue(assertFails { rag.loadGraph(snapshot.toString()) }.message.orEmpty().contains("unnamed vertices"))
            Files.writeString(file, JsonObject(graph + ("edges" to dangling)).toString())
            assertTrue(assertFails { rag.loadGraph(snapshot.toString()) }.message.orEmpty().contains("invalid edges"))
            assertEquals(listOf(first), rag.query("query", QueryParam("dpr", includeAnswer = false)).docs)
        }

    @Test
    fun deletingWithoutProvenancePreservesCommittedEvidence() =
        withRag { config, _ ->
            HippoRag(config).index(listOf(first, second))
            Files.delete(Path.of(config.saveDir, "openie_results_ner_${config.llmName}.json"))
            val restarted = HippoRag(config)
            val rag = HippoRAG(restarted)
            val before = rag.inspectGraph()
            val error = assertFails { restarted.delete(listOf(first)) }
            assertTrue(error.message.orEmpty().contains("without OpenIE provenance"))
            assertEquals(before, rag.inspectGraph())
            assertEquals(setOf(first, second), rag.query("query", QueryParam("dpr", includeAnswer = false)).docs.toSet())
        }

    @Test
    fun dropClearsPersistentIndexAndAllowsRebuilding() =
        withRag { config, _ ->
            val rag = HippoRAG(config)
            rag.upsert(first)
            assertTrue(assertFails { rag.query("query", QueryParam("unsupported")) }.message.orEmpty().contains("Unsupported"))
            rag.drop()
            assertTrue(rag.query("query", QueryParam("dpr", includeAnswer = false)).docs.isEmpty())
            assertTrue(Files.notExists(Path.of(config.saveDir, "openie_results_ner_${config.llmName}.json")))
            rag.upsert(second)
            val result = rag.query("query", QueryParam("hippo", includeAnswer = true))
            assertEquals(listOf(second), result.answerContext)
            assertEquals("Alpha", result.answer)
        }

    @Test
    fun metadataOnlyAsyncQueryDoesNotRequireAnIndex() =
        withRag { config, _ ->
            val adapter = HippoRagUnifiedAdapter(HippoRAG(config))
            val response =
                runBlocking {
                    adapter.aquery(
                        "query",
                        UnifiedQuery(mode = UnifiedMode.DPR, includeAnswer = false, includeContext = false, includeReferences = false),
                    )
                }
            assertTrue(response.context.isEmpty())
            assertTrue(response.references.isEmpty())
            assertEquals("dpr", response.metadata["modeUsed"])
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
