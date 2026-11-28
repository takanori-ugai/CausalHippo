package causalhippo

import causalrag.causalgraph.builder.CausalGraphBuilder
import causalrag.generator.llm.LLMInterface
import causalrag.generator.promptbuilder.PromptBuilder
import hipporag.config.BaseConfig
import hipporag.embeddingmodel.BaseEmbeddingModel
import hipporag.embeddingmodel.getEmbeddingModel
import hipporag.llm.getLlm
import hipporag.utils.LlmResult
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkConstructor
import io.mockk.mockkStatic
import io.mockk.unmockkConstructor
import io.mockk.unmockkStatic
import shared.chunking.countTokensWithJTokKit
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class HybridIntegrityTest {
    private val first = "Rainfall causes river flooding."
    private val second = "River flooding causes soil erosion."
    private val third = "Heat causes ice melting."

    @Test
    fun indexesOneCorpusAcrossIncrementalWritesAndReplacement() =
        fixture { config, hippo, _ ->
            val pipeline = CausalHippoPipeline(configPath = config, hippoConfig = hippo)
            pipeline.index(listOf(first))
            pipeline.index(listOf(second))
            assertTrue(pipeline.bm25Retriever.retrieve("rainfall", topK = 5).any { it["passage"] == first })
            pipeline.reindex(listOf(third))
            assertEquals(
                listOf(third),
                pipeline.hippoRag
                    .retrieveDpr(listOf("query"))
                    .first
                    .single()
                    .docs,
            )
        }

    @Test
    fun failedReplacementKeepsPreviousCorpus() =
        fixture { config, hippo, encoder ->
            val pipeline = CausalHippoPipeline(configPath = config, hippoConfig = hippo)
            pipeline.index(listOf(first))
            every { encoder.batchEncode(match { third in it }, any(), any()) } throws IllegalStateException("encoder unavailable")
            assertFails { pipeline.reindex(listOf(third)) }
            assertEquals(
                listOf(first),
                pipeline.hippoRag
                    .retrieveDpr(listOf("query"))
                    .first
                    .single()
                    .docs,
            )
            assertTrue(pipeline.bm25Retriever.retrieve("rainfall", topK = 5).isNotEmpty())
        }

    @Test
    fun rejectsSnapshotSaveOverLiveIndex() =
        fixture { config, hippo, _ ->
            val rag = CausalHippoRAG(configPath = config, hippoConfig = hippo)
            rag.upsert(first)
            val live = Files.list(Path.of(hippo.saveDir)).use { paths -> paths.filter(Files::isDirectory).findFirst().orElseThrow() }
            assertFails { rag.saveGraph(live.toString()) }
            assertTrue(Files.isRegularFile(live.resolve("graph.json")))
        }

    @Test
    fun fusesIdenticalPassagesDespiteDifferentHippoChunkSettings() =
        fixture { config, hippo, _ ->
            val pipeline = CausalHippoPipeline(configPath = config, hippoConfig = hippo.copy(preprocessChunkMaxTokenSize = 2))
            pipeline.index(listOf(first))
            assertEquals(
                listOf(first),
                pipeline.hippoRag
                    .retrieveDpr(listOf("query"))
                    .first
                    .single()
                    .docs,
            )
        }

    @Test
    fun rejectsCorruptCausalSnapshotWithoutReplacingLiveIndex() =
        fixture { config, hippo, _ ->
            val rag = CausalHippoRAG(configPath = config, hippoConfig = hippo)
            rag.upsert(first)
            val snapshot = Path.of(hippo.saveDir).resolveSibling("snapshot")
            rag.saveGraph(snapshot.toString())
            rag.upsert(second)
            val previous = rag.inspectGraph()
            Files.writeString(snapshot.resolve("causal_graph.json"), "invalid")
            assertFails { rag.loadGraph(snapshot.toString()) }
            assertEquals(previous, rag.inspectGraph())
        }

    @Test
    fun rejectsCausalGraphFromAnotherSnapshot() =
        fixture { config, hippo, _ ->
            val rag = CausalHippoRAG(configPath = config, hippoConfig = hippo)
            rag.upsert(first)
            val firstSnapshot = Path.of(hippo.saveDir).resolveSibling("snapshot-a")
            rag.saveGraph(firstSnapshot.toString())
            rag.upsert(second)
            val secondSnapshot = firstSnapshot.resolveSibling("snapshot-b")
            rag.saveGraph(secondSnapshot.toString())
            val previous = rag.inspectGraph()
            Files.copy(
                secondSnapshot.resolve("causal_graph.json"),
                firstSnapshot.resolve("causal_graph.json"),
                java.nio.file.StandardCopyOption.REPLACE_EXISTING,
            )
            val failure = assertFails { rag.loadGraph(firstSnapshot.toString()) }
            assertTrue(failure.message.orEmpty().contains("manifest"))
            assertEquals(previous, rag.inspectGraph())
        }

    @Test
    fun restoredCorpusCanBeReplacedAfterRestart() =
        fixture { config, hippo, _ ->
            val rag = CausalHippoRAG(configPath = config, hippoConfig = hippo)
            rag.upsert(first)
            val snapshot = Path.of(hippo.saveDir).resolveSibling("snapshot")
            rag.saveGraph(snapshot.toString())
            rag.upsert(second)
            rag.loadGraph(snapshot.toString())
            val restarted = CausalHippoPipeline(configPath = config, hippoConfig = hippo)
            assertEquals(listOf(first), restarted.hippoRag.indexedPassages())
            assertTrue(restarted.bm25Retriever.retrieve("rainfall", 3).isNotEmpty())
            restarted.reindex(listOf(third))
            assertEquals(listOf(third), restarted.hippoRag.indexedPassages())
        }

    @Test
    fun reportedCausalExtractionFailureCanBeRetried() =
        fixture { config, hippo, _ ->
            val pipeline = CausalHippoPipeline(configPath = config, hippoConfig = hippo)
            pipeline.index(listOf(first))
            mockkConstructor(CausalGraphBuilder::class)
            try {
                every { anyConstructed<CausalGraphBuilder>().lastIndexingReport } returns
                    shared.rag.IndexingReport(listOf(shared.rag.IndexingFailure("causal", "0", "temporary failure")))
                every { anyConstructed<CausalGraphBuilder>().indexDocuments(match { second in it }, any()) } returns 0
                pipeline.index(listOf(second))
                assertTrue(pipeline.lastIndexingReport.failures.isNotEmpty())
            } finally {
                unmockkConstructor(CausalGraphBuilder::class)
            }
            pipeline.index(listOf(second))
            assertTrue(pipeline.graphBuilder.hasCompletedPassage(second))
            assertEquals(setOf(first, second), pipeline.hippoRag.indexedPassages().toSet())
        }

    @Test
    fun failureAfterHippoIndexingKeepsAllCommittedRetrievalState() =
        fixture { config, hippo, _ ->
            val pipeline = CausalHippoPipeline(configPath = config, hippoConfig = hippo)
            pipeline.index(listOf(first))
            mockkConstructor(CausalGraphBuilder::class)
            try {
                every { anyConstructed<CausalGraphBuilder>().indexDocuments(match { second in it }, any()) } throws
                    IllegalStateException("causal encoder unavailable")
                assertFails { pipeline.index(listOf(second)) }
                assertEquals(listOf(first), pipeline.hippoRag.indexedPassages())
                assertEquals(listOf(first), CausalHippoPipeline(configPath = config, hippoConfig = hippo).hippoRag.indexedPassages())
            } finally {
                unmockkConstructor(CausalGraphBuilder::class)
            }
        }

    @Test
    fun finalRetrievalPreservesAdaptiveRankingAndRespectsTopKAndUpdates() =
        fixture { config, hippo, _ ->
            val pipeline = CausalHippoPipeline(configPath = config, hippoConfig = hippo)
            val documents = listOf(first, second, third, "Sunlight causes plant growth.", "Cold causes water to freeze.")
            pipeline.index(documents)
            val query = "What causes flooding?"
            val firstResult = pipeline.hybridRetriever.retrieveWithDetails(query, 1)
            val details = firstResult.single().getValue("details") as Map<*, *>
            val diagnostics = details["weight_diagnostics"] as Map<*, *>
            assertEquals("dynamic", diagnostics["mode"])
            assertEquals("enabled", diagnostics["confidence_switch"])
            assertEquals("enabled", diagnostics["two_pass_adaptive"])
            val expectedRanking = pipeline.hybridRetriever.retrieve(query, topK = 5)
            assertEquals(5, expectedRanking.size)
            assertEquals(expectedRanking, pipeline.retrieveContext(query, topK = 5))
            pipeline.index(listOf("Wind causes waves."))
            assertEquals(6, pipeline.retrieveContext(query, topK = 6).size)
        }

    @Test
    fun generationReceivesOnlyTheReportedPaths() =
        fixture { config, hippo, _ ->
            val prompts = mutableListOf<String>()
            every { anyConstructed<LLMInterface>().generate(any(), any(), any(), any(), any(), any()) } answers {
                prompts.add(firstArg())
                "[]"
            }
            val rag = CausalHippoRAG(configPath = config, hippoConfig = hippo)
            rag.upsert(listOf(first, second, third))
            val question = "What causes river flooding and soil erosion?"
            val result = rag.query(question, CausalHippoQueryParam(maxPaths = 1))
            val evidence = rag.query(question, CausalHippoQueryParam(maxPaths = 1, onlyNeedContext = true, onlyNeedCausalPaths = true))
            assertEquals(1, result.causalPaths.size)
            assertEquals(result.causalPaths, evidence.causalPaths)
            assertEquals(result.context, evidence.context)
            assertTrue(result.causalPaths.single().joinToString(" → ") in prompts.last())
        }

    @Test
    fun directedEvidenceDistinguishesPassiveNegationAndCooccurrence() {
        assertTrue(supportsDirectedRelation("Flooding was caused by rainfall.", "rainfall", "flooding"))
        assertTrue(supportsDirectedRelation("Rainfall causes flooding.", "rainfall", "flooding"))
        assertFalse(supportsDirectedRelation("Rainfall does not cause flooding.", "rainfall", "flooding"))
        assertFalse(supportsDirectedRelation("It is false that rainfall causes flooding.", "rainfall", "flooding"))
        assertFalse(supportsDirectedRelation("Rainfall and flooding were reported.", "rainfall", "flooding"))
        assertFalse(supportsDirectedRelation("Flooding causes rainfall.", "rainfall", "flooding"))
    }

    @Test
    fun budgetsRenderedPromptAndKeepsShortEvidence() {
        val builder = PromptBuilder(templateStyle = "detailed")
        val question = "What causes flooding?"
        val path = listOf("rainfall", "flooding")
        val budget = countTokensWithJTokKit(builder.buildPrompt(question, listOf(first), listOf(path)))
        val oversized = "unrelated material ".repeat(budget)
        val result = builder.buildBudgetedPrompt(question, listOf(oversized, first), listOf(path), budget, summarize = false)
        assertEquals(listOf(first), result.passages)
        assertEquals(listOf(path), result.paths)
        assertTrue(countTokensWithJTokKit(result.text) <= budget)
    }

    private fun fixture(test: (String, BaseConfig, BaseEmbeddingModel) -> Unit) {
        val root = Files.createTempDirectory("causalhippo-integrity")
        mockkStatic(::getLlm, ::getEmbeddingModel)
        mockkConstructor(LLMInterface::class)
        try {
            val config = root.resolve("config.json")
            Files.writeString(config, """{"modelName":"test","embeddingModel":"test","llmApiKey":"","embeddingApiKey":""}""")
            every { anyConstructed<LLMInterface>().generate(any(), any(), any(), any(), any(), any()) } returns "[]"
            val encoder = mockk<BaseEmbeddingModel>()
            every { encoder.batchEncode(any(), any(), any()) } answers {
                Array(firstArg<List<String>>().size) { doubleArrayOf(1.0, 0.0) }
            }
            every { getEmbeddingModel() } returns mockk { every { create(any(), any()) } returns encoder }
            every { getLlm(any()) } returns
                mockk {
                    every { infer(any()) } returns LlmResult("""{"named_entities":[],"triples":[]}""", emptyMap())
                }
            test(config.toString(), BaseConfig(saveDir = root.resolve("hippo").toString()), encoder)
        } finally {
            unmockkConstructor(LLMInterface::class)
            unmockkStatic(::getLlm, ::getEmbeddingModel)
            root.toFile().deleteRecursively()
        }
    }
}
