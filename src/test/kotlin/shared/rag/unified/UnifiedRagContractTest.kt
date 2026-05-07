package shared.rag.unified

import causalhippo.CausalHippoRAG
import causalrag.CausalRAG
import causalrag.CausalRagRunResult
import com.microsoft.graphrag.GraphRAG
import com.microsoft.graphrag.query.QueryResult as GraphQueryResult
import hipporag.HippoRAG
import hipporag.utils.QuerySolution
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryResult as LightQueryResult
import pathrag.PathRAG
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UnifiedRagContractTest {
    @Test
    fun `all adapters expose normalized inspect graph shape`() {
        val adapters = buildAdapters()

        adapters.forEach { (_, rag) ->
            val inspection = rag.inspectGraph()
            assertTrue(inspection["nodes"] is List<*>)
            assertTrue(inspection["edges"] is List<*>)
            assertTrue(inspection["metadata"] is Map<*, *>)
            val metadata = inspection["metadata"] as Map<*, *>
            assertTrue(metadata.containsKey("nodeCount"))
            assertTrue(metadata.containsKey("edgeCount"))
        }
    }

    @Test
    fun `all adapters return unified response metadata with rag id`() =
        runBlocking {
            val adapters = buildAdapters()

            adapters.forEach { (ragId, rag) ->
                val response =
                    rag.aquery(
                        "What is the relation?",
                        UnifiedQuery(
                            text = "What is the relation?",
                            mode = UnifiedMode.HYBRID,
                            includeAnswer = true,
                            includeContext = true,
                            includeReferences = true,
                            includeGraphPaths = true,
                            includeFollowUps = true,
                            streaming = true,
                        ),
                    )

                assertEquals(ragId.name, response.metadata["ragId"])
                assertTrue(response.metadata.containsKey("modeUsed"))
            }
        }

    private fun buildAdapters(): List<Pair<RagId, UnifiedRag>> {
        val graphDelegate =
            mockk<GraphRAG>().also { delegate ->
                coEvery { delegate.aquery(any(), any()) } returns GraphQueryResult(answer = "graph-answer", context = emptyList())
                every { delegate.inspectGraph() } returns
                    mapOf(
                        "nodes" to listOf(mapOf("id" to "g1")),
                        "edges" to listOf(mapOf("source" to "g1", "target" to "g2")),
                    )
            }

        val lightDelegate =
            mockk<LightRAG>().also { delegate ->
                coEvery { delegate.aquery(any(), any()) } returns
                    LightQueryResult(
                        content = "light-answer",
                        rawData = mapOf("data" to mapOf("context" to listOf("light-context"))),
                    )
                every { delegate.inspectGraph() } returns
                    mapOf(
                        "nodes" to listOf(mapOf("id" to "l1")),
                        "edges" to listOf(mapOf("source" to "l1", "target" to "l2")),
                    )
            }

        val pathDelegate =
            mockk<PathRAG>().also { delegate ->
                coEvery { delegate.aquery(any(), any()) } returns "path-payload"
                every { delegate.inspectGraph() } returns
                    mapOf(
                        "nodes" to mapOf("P1" to mapOf("entity_name" to "P1")),
                        "edges" to listOf(mapOf("source" to "P1", "target" to "P2")),
                    )
            }

        val hippoDelegate =
            mockk<HippoRAG>().also { delegate ->
                coEvery { delegate.aquery(any(), any()) } returns
                    QuerySolution(
                        question = "q",
                        docs = listOf("hippo-doc"),
                        answer = "hippo-answer",
                    )
                every { delegate.inspectGraph() } returns
                    mapOf(
                        "nodes" to listOf(mapOf("id" to "h1")),
                        "edges" to listOf(mapOf("source" to "h1", "target" to "h2")),
                    )
            }

        val causalDelegate =
            mockk<CausalRAG>().also { delegate ->
                coEvery { delegate.aquery(any(), any()) } returns
                    CausalRagRunResult(
                        answer = "causal-answer",
                        context = listOf("causal-context"),
                        causalPaths = listOf(listOf("A", "B")),
                    )
                every { delegate.inspectGraph() } returns
                    mapOf(
                        "nodes" to listOf(mapOf("id" to "c1")),
                        "edges" to listOf(mapOf("source" to "c1", "target" to "c2")),
                    )
            }

        val causalHippoDelegate =
            mockk<CausalHippoRAG>().also { delegate ->
                coEvery { delegate.aquery(any(), any()) } returns
                    CausalRagRunResult(
                        answer = "causal-hippo-answer",
                        context = listOf("causal-hippo-context"),
                        causalPaths = listOf(listOf("X", "Y")),
                    )
                every { delegate.inspectGraph() } returns
                    mapOf(
                        "causal_graph" to
                            mapOf(
                                "nodes" to listOf(mapOf("id" to "cg1")),
                                "edges" to listOf(mapOf("source" to "cg1", "target" to "cg2")),
                            ),
                        "hippo_graph" to
                            mapOf(
                                "nodes" to listOf(mapOf("id" to "hg1")),
                                "edges" to listOf(mapOf("source" to "hg1", "target" to "hg2")),
                            ),
                    )
            }

        return listOf(
            RagId.GRAPH_RAG to GraphRagUnifiedAdapter(graphDelegate),
            RagId.LIGHT_RAG to LightRagUnifiedAdapter(lightDelegate),
            RagId.PATH_RAG to PathRagUnifiedAdapter(pathDelegate),
            RagId.HIPPO_RAG to HippoRagUnifiedAdapter(hippoDelegate),
            RagId.CAUSAL_RAG to CausalRagUnifiedAdapter(causalDelegate),
            RagId.CAUSAL_HIPPO_RAG to CausalHippoRagUnifiedAdapter(causalHippoDelegate),
        )
    }
}
