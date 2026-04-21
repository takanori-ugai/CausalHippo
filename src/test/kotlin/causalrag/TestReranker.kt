package causalrag

import causalrag.causalgraph.retriever.CausalPathRetriever
import causalrag.reranker.BaseReranker
import causalrag.reranker.CausalPathReranker
import causalrag.retriever.Bm25Retriever
import causalrag.retriever.HybridRetriever
import causalrag.retriever.SemanticRetriever
import causalrag.retriever.VectorStoreRetriever
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests causal reranking behavior and score ordering.
 */
class TestReranker {
    private lateinit var mockRetriever: CausalPathRetriever
    private lateinit var reranker: CausalPathReranker

    private val testPaths =
        listOf(
            listOf("climate change", "rising sea levels", "coastal flooding"),
            listOf("deforestation", "carbon capture reduction", "CO2 increase"),
        )

    private val testNodes =
        listOf(
            "climate change",
            "rising sea levels",
            "coastal flooding",
            "deforestation",
            "carbon capture",
            "CO2",
        )

    private val testDocs =
        listOf(
            "Climate patterns have been changing in recent decades.",
            "Rising sea levels caused by climate change lead to coastal flooding.",
            "Deforestation reduces the ability of forests to capture carbon.",
            "Coral reefs are affected by ocean temperature changes.",
            "Coastal cities are implementing flood protection measures.",
        )

    /**
     * Initializes the mocked retriever and reranker under test.
     */
    @BeforeTest
    fun setUp() {
        mockRetriever = mockk()
        every { mockRetriever.retrievePathNodes(any(), any(), any(), any()) } returns testNodes
        every { mockRetriever.retrievePaths(any(), any(), any(), any()) } returns testPaths
        reranker = CausalPathReranker(mockRetriever)
    }

    /**
     * Verifies baseline reranking behavior for a simple overlap scorer.
     */
    @Test
    fun testBaseReranker() {
        class SimpleReranker : BaseReranker("simple") {
            override fun rerank(
                query: String,
                candidates: List<String>,
                metadata: List<Map<String, Any>>?,
            ): List<Pair<String, Double>> {
                val queryWords =
                    query
                        .lowercase()
                        .split(Regex("\\s+"))
                        .filter { it.isNotBlank() }
                        .toSet()
                return candidates
                    .map { doc ->
                        val docWords =
                            doc
                                .lowercase()
                                .split(Regex("\\s+"))
                                .filter { it.isNotBlank() }
                                .toSet()
                        doc to queryWords.intersect(docWords).size.toDouble()
                    }.sortedByDescending { it.second }
            }
        }

        val reranker = SimpleReranker()
        val ranked = reranker.rerank("climate change effects", testDocs)
        assertEquals(testDocs.size, ranked.size)
        assertTrue(
            ranked
                .first()
                .first
                .lowercase()
                .contains("climate"),
        )
    }

    /**
     * Verifies that causal path signals promote relevant passages.
     */
    @Test
    fun testCausalPathReranker() {
        val query = "How does climate change affect coastal areas?"
        val rankedDocs = reranker.rerank(query, testDocs)

        verify { mockRetriever.retrievePaths(query, any(), any(), any()) }
        assertTrue(
            rankedDocs
                .first()
                .first
                .lowercase()
                .contains("rising sea levels"),
        )
        assertTrue(
            rankedDocs
                .first()
                .first
                .lowercase()
                .contains("coastal flooding"),
        )
    }

    /**
     * Verifies that documents containing the full causal chain rank highest.
     */
    @Test
    fun testDocumentScoring() {
        every { mockRetriever.retrievePaths(any(), any(), any(), any()) } returns
            listOf(listOf("climate change", "rising sea levels", "coastal flooding"))

        val docs =
            listOf(
                "Climate change is causing rising sea levels.",
                "Coastal flooding is a problem in many cities.",
                "Climate change leads to rising sea levels, which causes coastal flooding.",
            )
        val ranked = reranker.rerank("What causes coastal flooding?", docs)
        assertTrue(ranked.first().first.contains("climate change leads to rising sea levels", ignoreCase = true))
    }

    /**
     * Verifies that custom weights still favor passages preserving the full chain.
     */
    @Test
    fun testRerankingWithWeights() {
        val weighted =
            CausalPathReranker(
                mockRetriever,
                nodeMatchWeight = 1.0,
                pathMatchWeight = 3.0,
            )

        every { mockRetriever.retrievePaths(any(), any(), any(), any()) } returns
            listOf(listOf("climate change", "rising sea levels", "coastal flooding"))

        val docs =
            listOf(
                "Climate change and flooding are environmental issues.",
                "Rising sea levels are causing coastal flooding worldwide.",
                "Climate change leads to rising sea levels, causing coastal flooding.",
            )

        val ranked = weighted.rerank("How does climate change cause flooding?", docs)
        assertTrue(ranked.first().first.contains("climate change leads to rising sea levels", ignoreCase = true))
    }

    /**
     * Verifies that BM25 returns no passages when a query has no lexical matches.
     */
    @Test
    fun testBm25ReturnsNoResultsForMiss() {
        val retriever = Bm25Retriever()
        retriever.indexCorpus(testDocs)

        val results = retriever.retrieve("quasar nebula pulsar", topK = 3)

        assertTrue(results.isEmpty())
    }

    /**
     * Verifies that BM25-only candidates can still be returned when semantic retrieval is empty.
     */
    @Test
    fun testHybridRetrieverSupportsBm25Fallback() {
        val vectorRetriever = mockk<VectorStoreRetriever>()
        val graphRetriever = mockk<CausalPathRetriever>()
        val bm25Retriever = Bm25Retriever()
        bm25Retriever.indexCorpus(testDocs)

        every { vectorRetriever.searchWithScores(any(), any()) } returns emptyList()
        every { graphRetriever.retrievePathNodes(any(), any(), any(), any()) } returns emptyList()
        every { graphRetriever.retrievePaths(any(), any(), any(), any()) } returns emptyList()

        val retriever =
            HybridRetriever(
                semanticRetriever = vectorRetriever,
                graphRetriever = graphRetriever,
                semanticWeight = 0.0,
                causalWeight = 0.0,
                bm25Weight = 1.0,
                bm25Retriever = bm25Retriever,
            )

        val results = retriever.retrieveWithDetails("climate change", topK = 3)

        assertTrue(results.isNotEmpty())
        assertTrue(results.any { (it["passage"] as String).contains("Climate", ignoreCase = true) })
    }

    /**
     * Verifies that query-aware gating increases causal weight for causally rich queries.
     */
    @Test
    fun testHybridRetrieverDynamicWeightingBoostsCausalSignal() {
        val semanticRetriever = mockk<SemanticRetriever>()
        val graphRetriever = mockk<CausalPathRetriever>()
        val neutralPassage = "General weather discussion without a clear cause chain."
        val causalPassage = "Rainfall a increases runoff and a leads to b flooding impacts."

        every { semanticRetriever.searchWithScores(any(), any()) } returns
            listOf(
                neutralPassage to 0.95,
                causalPassage to 0.35,
            )
        every { graphRetriever.retrievePathNodes(any(), any(), any(), any()) } returns listOf("a", "b")
        every { graphRetriever.retrievePaths(any(), any(), any(), any()) } returns
            listOf(
                listOf("a", "b"),
                listOf("x", "y"),
                listOf("m", "n"),
            )
        every { graphRetriever.retrieveNodes(any(), any(), any()) } returns
            listOf(
                "a" to 0.95,
                "x" to 0.10,
                "m" to 0.05,
            )

        val retriever =
            HybridRetriever(
                semanticRetriever = semanticRetriever,
                graphRetriever = graphRetriever,
                semanticWeight = 0.8,
                causalWeight = 0.2,
                bm25Weight = 0.0,
                dynamicWeightingEnabled = true,
                minCausalMatches = 0,
            )

        val results = retriever.retrieveWithDetails("How does a cause b?", topK = 2)

        assertTrue(results.isNotEmpty())
        assertEquals(causalPassage, results.first()["passage"])
        val details = results.first()["details"] as Map<*, *>
        val semanticWeight = details["semantic_weight"] as Double
        val causalWeight = details["causal_weight"] as Double
        assertTrue(causalWeight > semanticWeight)
    }

    /**
     * Verifies two-pass adaptive weighting can upweight causal signal after semantic-first ranking.
     */
    @Test
    fun testHybridRetrieverTwoPassAdaptiveRebalancesWeights() {
        val semanticRetriever = mockk<SemanticRetriever>()
        val graphRetriever = mockk<CausalPathRetriever>()
        val neutralPassage = "General weather discussion without a clear cause chain."
        val causalPassage = "Rainfall a increases runoff and a leads to b flooding impacts."

        every { semanticRetriever.searchWithScores(any(), any()) } returns
            listOf(
                neutralPassage to 0.95,
                causalPassage to 0.35,
            )
        every { graphRetriever.retrievePathNodes(any(), any(), any(), any()) } returns listOf("a", "b")
        every { graphRetriever.retrievePaths(any(), any(), any(), any()) } returns listOf(listOf("a", "b"))

        val retriever =
            HybridRetriever(
                semanticRetriever = semanticRetriever,
                graphRetriever = graphRetriever,
                semanticWeight = 0.8,
                causalWeight = 0.2,
                bm25Weight = 0.0,
                twoPassAdaptiveEnabled = true,
                minCausalMatches = 0,
            )

        val results = retriever.retrieveWithDetails("How does a cause b?", topK = 1)

        assertTrue(results.isNotEmpty())
        assertEquals(causalPassage, results.first()["passage"])
        val details = results.first()["details"] as Map<*, *>
        val diagnostics = details["weight_diagnostics"] as Map<*, *>
        assertEquals("enabled", diagnostics["two_pass_adaptive"])
    }

    /**
     * Verifies confidence-based switching raises semantic weight when causal evidence is weak.
     */
    @Test
    fun testHybridRetrieverConfidenceSwitchBoostsSemanticWhenCausalIsWeak() {
        val semanticRetriever = mockk<SemanticRetriever>()
        val graphRetriever = mockk<CausalPathRetriever>()
        val docA = "A broad explanation with high semantic relevance."
        val docB = "Another semantically relevant passage."

        every { semanticRetriever.searchWithScores(any(), any()) } returns
            listOf(
                docA to 0.90,
                docB to 0.70,
            )
        every { graphRetriever.retrievePathNodes(any(), any(), any(), any()) } returns emptyList()
        every { graphRetriever.retrievePaths(any(), any(), any(), any()) } returns emptyList()

        val retriever =
            HybridRetriever(
                semanticRetriever = semanticRetriever,
                graphRetriever = graphRetriever,
                semanticWeight = 0.2,
                causalWeight = 0.8,
                bm25Weight = 0.0,
                confidenceBasedSwitchEnabled = true,
                minCausalMatches = 0,
            )

        val results = retriever.retrieveWithDetails("Explain impacts", topK = 2)

        assertTrue(results.isNotEmpty())
        val details = results.first()["details"] as Map<*, *>
        val semanticWeight = details["semantic_weight"] as Double
        val causalWeight = details["causal_weight"] as Double
        val diagnostics = details["weight_diagnostics"] as Map<*, *>
        assertTrue(semanticWeight > causalWeight)
        assertEquals("enabled", diagnostics["confidence_switch"])
    }
}
