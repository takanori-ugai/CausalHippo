package com.youtu.graphrag.shared.treecomm

import com.youtu.graphrag.shared.graph.GraphNode
import com.youtu.graphrag.shared.graph.GraphRelationship
import com.youtu.graphrag.shared.llm.LlmClient
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FastTreeCommTest {
    @Test
    fun `detectCommunities builds deterministic community metadata`() {
        val nodeA = GraphNode("A", emptyMap())
        val nodeB = GraphNode("B", emptyMap())
        val nodeC = GraphNode("C", emptyMap())

        val relationships =
            listOf(
                GraphRelationship(nodeA, "links_to", nodeB),
                GraphRelationship(nodeB, "links_to", nodeC),
                GraphRelationship(nodeA, "links_to", nodeC),
            )

        val communities = FastTreeComm().detectCommunities(relationships)

        assertEquals(1, communities.size)
        assertEquals("community_1", communities[0]["community_id"])
        assertEquals(3, communities[0]["node_count"]) // 3 unique nodes: A, B, C
        assertEquals(3, communities[0]["edge_count"])
        assertEquals(listOf("A", "B", "C"), communities[0]["nodes"])
    }

    @Test
    fun `detectCommunities returns empty list for empty relationships`() {
        val communities = FastTreeComm().detectCommunities(emptyList())
        assertEquals(0, communities.size)
    }

    @Test
    fun `detectCommunities separates disconnected components and extracts keywords`() {
        val nodeA = GraphNode("entity", mapOf("name" to "Project Alpha"))
        val nodeB = GraphNode("entity", mapOf("name" to "Tokyo Office"))
        val nodeC = GraphNode("entity", mapOf("name" to "Bob Team"))
        val nodeD = GraphNode("entity", mapOf("name" to "Carol Team"))

        val relationships =
            listOf(
                GraphRelationship(nodeA, "located_in", nodeB),
                GraphRelationship(nodeC, "paired_with", nodeD),
            )

        val communities = FastTreeComm().detectCommunities(relationships)

        assertEquals(2, communities.size)
        assertEquals("community_1", communities[0]["community_id"])
        assertEquals(2, communities[0]["node_count"])
        assertEquals("community_2", communities[1]["community_id"])
        assertEquals(2, communities[1]["node_count"])

        val keywords =
            communities
                .flatMap { community -> (community["keywords"] as? List<*>)?.map { it.toString() }.orEmpty() }
                .toSet()
        assertTrue("Project Alpha" in keywords)
        assertTrue("Bob Team" in keywords || "Carol Team" in keywords)
    }

    @Test
    fun `detectCommunities supports hierarchical mode and max community cutoff`() {
        val nodeA = GraphNode("entity", mapOf("name" to "Alpha"))
        val nodeB = GraphNode("entity", mapOf("name" to "Beta"))
        val nodeC = GraphNode("entity", mapOf("name" to "Gamma"))
        val nodeD = GraphNode("entity", mapOf("name" to "Delta"))

        val relationships =
            listOf(
                GraphRelationship(nodeA, "links_to", nodeB),
                GraphRelationship(nodeB, "links_to", nodeC),
                GraphRelationship(nodeC, "links_to", nodeD),
            )

        val communities =
            FastTreeComm().detectCommunities(
                relationships = relationships,
                options =
                    TreeCommOptions(
                        enableFastMode = false,
                        maxTotalCommunities = 1,
                        mergeThreshold = 0.0,
                        maxIterations = 10,
                    ),
            )

        assertEquals(1, communities.size)
        assertEquals(4, communities.first()["node_count"])
    }

    @Test
    fun `detectCommunities generates summary with deterministic fallback when llm unavailable`() {
        val nodeA = GraphNode("entity", mapOf("name" to "Alice"))
        val nodeB = GraphNode("entity", mapOf("name" to "Acme"))
        val relationships =
            listOf(
                GraphRelationship(nodeA, "works_at", nodeB),
            )

        val llmClient =
            object : LlmClient {
                override fun complete(prompt: String): String = ""
            }

        val communities =
            FastTreeComm(llmClient).detectCommunities(
                relationships = relationships,
                options =
                    TreeCommOptions(
                        enableSummary = true,
                        summaryMaxWords = 16,
                    ),
            )

        val summary = communities.first()["summary"]?.toString().orEmpty()
        assertTrue(summary.isNotBlank())
        assertTrue(summary.split(Regex("\\s+")).size <= 16)
    }

    @Test
    fun `treeComm options reject invalid ranges`() {
        assertFailsWith<IllegalArgumentException> { TreeCommOptions(structWeight = -0.1) }
        assertFailsWith<IllegalArgumentException> { TreeCommOptions(mergeThreshold = 1.1) }
        assertFailsWith<IllegalArgumentException> { TreeCommOptions(maxTotalCommunities = 0) }
        assertFailsWith<IllegalArgumentException> { TreeCommOptions(maxIterations = -1) }
        assertFailsWith<IllegalArgumentException> { TreeCommOptions(summaryMaxWords = 0) }
    }

    @Test
    fun `community cap does not drop disconnected components in fast mode`() {
        val nodeA = GraphNode("entity", mapOf("name" to "A"))
        val nodeB = GraphNode("entity", mapOf("name" to "B"))
        val nodeC = GraphNode("entity", mapOf("name" to "C"))
        val nodeD = GraphNode("entity", mapOf("name" to "D"))

        val relationships =
            listOf(
                GraphRelationship(nodeA, "links_to", nodeB),
                GraphRelationship(nodeC, "links_to", nodeD),
            )

        val communities =
            FastTreeComm().detectCommunities(
                relationships = relationships,
                options = TreeCommOptions(enableFastMode = true, maxTotalCommunities = 1),
            )

        assertEquals(2, communities.size)
        val members =
            communities
                .flatMap { community -> (community["nodes"] as? List<*>)?.mapNotNull { node -> node?.toString() }.orEmpty() }
                .toSet()
        assertEquals(setOf("A", "B", "C", "D"), members)
    }
}
