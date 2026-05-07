package shared.rag.unified

import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import pathrag.PathRAG
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UnifiedGraphSnapshotTest {
    @Test
    fun `normalizeGraphInspection converts map nodes to list with ids`() {
        val payload =
            mapOf(
                "nodes" to
                    mapOf(
                        "ALPHA" to mapOf("description" to "alpha"),
                        "BETA" to mapOf("description" to "beta"),
                    ),
                "edges" to
                    listOf(
                        mapOf("source" to "ALPHA", "target" to "BETA"),
                    ),
            )

        val normalized = normalizeGraphInspection(payload)
        val nodes = normalized["nodes"] as List<*>
        val edges = normalized["edges"] as List<*>
        val metadata = normalized["metadata"] as Map<*, *>

        assertEquals(2, nodes.size)
        assertEquals(1, edges.size)
        assertTrue((nodes.first() as Map<*, *>).containsKey("id"))
        assertEquals(2, metadata["nodeCount"])
        assertEquals(1, metadata["edgeCount"])
    }

    @Test
    fun `normalizeGraphInspection flattens causal hippo nested graphs`() {
        val payload =
            mapOf(
                "causal_graph" to
                    mapOf(
                        "nodes" to listOf(mapOf("id" to "C1")),
                        "edges" to listOf(mapOf("source" to "C1", "target" to "C2")),
                    ),
                "hippo_graph" to
                    mapOf(
                        "nodes" to listOf(mapOf("id" to "H1")),
                        "edges" to listOf(mapOf("source" to "H1", "target" to "H2")),
                    ),
            )

        val normalized = normalizeGraphInspection(payload)
        val nodes = normalized["nodes"] as List<*>
        val edges = normalized["edges"] as List<*>

        assertEquals(2, nodes.size)
        assertEquals(2, edges.size)
        val nodeGraphs = nodes.map { (it as Map<*, *>)["graph"] as String }.toSet()
        assertTrue("causal_graph" in nodeGraphs)
        assertTrue("hippo_graph" in nodeGraphs)
    }

    @Test
    fun `path adapter inspectGraph returns normalized shape`() {
        val delegate = mockk<PathRAG>()
        every {
            delegate.inspectGraph()
        } returns
            mapOf(
                "nodes" to mapOf("A" to mapOf("entity" to "A")),
                "edges" to listOf(mapOf("source" to "A", "target" to "B")),
            )
        coEvery { delegate.ainspectGraph() } returns
            mapOf(
                "nodes" to mapOf("A" to mapOf("entity" to "A")),
                "edges" to listOf(mapOf("source" to "A", "target" to "B")),
            )

        val adapter = PathRagUnifiedAdapter(delegate)
        val normalized = adapter.inspectGraph()

        assertTrue(normalized["nodes"] is List<*>)
        assertTrue(normalized["edges"] is List<*>)
        assertTrue(normalized["metadata"] is Map<*, *>)
    }
}
