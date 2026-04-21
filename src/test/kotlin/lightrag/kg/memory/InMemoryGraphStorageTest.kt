package lightrag.kg.memory

import kotlinx.coroutines.runBlocking
import lightrag.TestEmbeddings
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Tests for in-memory graph storage operations including CRUD, batching, and edge semantics. */
class InMemoryGraphStorageTest {
    /** Verifies undirected edge storage and deduplication in edge listings. */
    @Test
    fun `upsert edge stores undirected relation and deduplicates getAllEdges`() {
        runBlocking {
            val storage =
                InMemoryGraphStorage(
                    namespace = "ns",
                    workspace = "ws",
                    embeddingFunc = TestEmbeddings.mockEmbeddingModel(),
                )
            storage.upsertNode("A", mapOf("label" to "A"))
            storage.upsertNode("B", mapOf("label" to "B"))

            val edgeData = mapOf("type" to "knows")
            storage.upsertEdge("A", "B", edgeData)

            assertTrue(storage.hasEdge("A", "B"))
            assertTrue(storage.hasEdge("B", "A"))

            val edges = storage.getAllEdges()
            assertEquals(1, edges.size, "Edges should be deduped for undirected storage")
            val stored = edges.first()
            assertEquals("A", stored["source"])
            assertEquals("B", stored["target"])
            assertEquals("knows", stored["type"])

            val nodeEdges = storage.getNodeEdges("A")
            assertNotNull(nodeEdges)
            assertEquals(listOf("A" to "B"), nodeEdges)
            assertNull(storage.getNodeEdges("missing"))
        }
    }

    private fun createStorage(): InMemoryGraphStorage =
        InMemoryGraphStorage(
            namespace = "test_graph",
            workspace = "test_workspace",
            embeddingFunc = TestEmbeddings.mockEmbeddingModel(),
        )

    /** Covers basic graph CRUD: node insertion, edge creation, and reverse edge lookup. */
    @Test
    fun `test basic graph operations`() {
        runBlocking {
            val storage = createStorage()

            // 1. Insert the first node
            val node1Id = "Artificial Intelligence"
            val node1Data =
                mapOf(
                    "entity_id" to node1Id,
                    "description" to "Artificial intelligence is a branch of computer science...",
                    "keywords" to "AI,Machine Learning,Deep Learning",
                    "entity_type" to "Technology Field",
                )
            storage.upsertNode(node1Id, node1Data)

            // 2. Insert the second node
            val node2Id = "Machine Learning"
            val node2Data =
                mapOf(
                    "entity_id" to node2Id,
                    "description" to "Machine learning is a branch of artificial intelligence...",
                    "keywords" to "Supervised Learning,Unsupervised Learning",
                    "entity_type" to "Technology Field",
                )
            storage.upsertNode(node2Id, node2Data)

            // 3. Insert the connecting edge
            val edgeData =
                mapOf(
                    "relationship" to "includes",
                    "weight" to "1.0",
                    "description" to "The field of artificial intelligence includes the subfield of machine learning.",
                )
            storage.upsertEdge(node1Id, node2Id, edgeData)

            // 4. Read node properties
            val node1Props = storage.getNode(node1Id)
            assertNotNull(node1Props)
            assertEquals(node1Id, node1Props["entity_id"])
            assertEquals(node1Data["description"], node1Props["description"])
            assertEquals(node1Data["entity_type"], node1Props["entity_type"])

            // 5. Read edge properties
            val edgeProps = storage.getEdge(node1Id, node2Id)
            assertNotNull(edgeProps)
            assertEquals(edgeData["relationship"], edgeProps["relationship"])
            assertEquals(edgeData["description"], edgeProps["description"])
            assertEquals(edgeData["weight"], edgeProps["weight"])

            // 5.1 Verify undirected graph property - read reverse edge properties
            val reverseEdgeProps = storage.getEdge(node2Id, node1Id)
            assertNotNull(reverseEdgeProps)
            assertEquals(edgeProps, reverseEdgeProps)
        }
    }

    /** Confirms deleting a node removes all connected edges and adjacency results. */
    @Test
    fun `delete node removes all connected edges`() {
        runBlocking {
            val storage =
                InMemoryGraphStorage(
                    namespace = "ns",
                    workspace = "ws",
                    embeddingFunc = TestEmbeddings.mockEmbeddingModel(),
                )
            storage.upsertNode("A", mapOf())
            storage.upsertNode("B", mapOf())
            storage.upsertNode("C", mapOf())
            storage.upsertEdge("A", "B", mapOf())
            storage.upsertEdge("B", "C", mapOf())

            storage.deleteNode("B")

            assertFalse(storage.hasNode("B"))
            assertFalse(storage.hasEdge("A", "B"))
            assertFalse(storage.hasEdge("B", "A"))
            assertFalse(storage.hasEdge("B", "C"))
            assertFalse(storage.hasEdge("C", "B"))
            assertEquals(emptyList(), storage.getNodeEdges("A"))
        }
    }

    /** Exercises degree calculations and connectivity metrics for a small graph. */
    @Test
    fun `test advanced graph operations`() {
        runBlocking {
            val storage = createStorage()

            // 1. Insert test data
            val node1Id = "Artificial Intelligence"
            val node2Id = "Machine Learning"
            val node3Id = "Deep Learning"

            storage.upsertNode(node1Id, mapOf("entity_id" to node1Id))
            storage.upsertNode(node2Id, mapOf("entity_id" to node2Id))
            storage.upsertNode(node3Id, mapOf("entity_id" to node3Id))

            storage.upsertEdge(node1Id, node2Id, mapOf("relationship" to "includes"))
            storage.upsertEdge(node2Id, node3Id, mapOf("relationship" to "includes"))

            // 2. Test nodeDegree
            assertEquals(1, storage.nodeDegree(node1Id))
            assertEquals(2, storage.nodeDegree(node2Id))
            assertEquals(1, storage.nodeDegree(node3Id))

            // 3. Test edgeDegree
            assertEquals(3, storage.edgeDegree(node1Id, node2Id))

            // 3.1 Test reverse edge degree
            assertEquals(3, storage.edgeDegree(node2Id, node1Id))

            // 4. Test getNodeEdges
            val node2Edges = storage.getNodeEdges(node2Id)
            assertNotNull(node2Edges)
            assertEquals(2, node2Edges.size)

            // 5. Test getAllLabels
            val allLabels = storage.getAllLabels()
            assertEquals(3, allLabels.size)
            assertTrue(allLabels.contains(node1Id))
            assertTrue(allLabels.contains(node2Id))
            assertTrue(allLabels.contains(node3Id))

            // 6. Test getKnowledgeGraph
            val kg = storage.getKnowledgeGraph("*", 2, 10)
            assertEquals(3, kg.nodes.size)
            assertEquals(2, kg.edges.size)
            assertFalse(kg.isTruncated)

            // 7. Test deleteNode
            storage.deleteNode(node3Id)
            assertNull(storage.getNode(node3Id))

            // Re-insert for next tests
            storage.upsertNode(node3Id, mapOf("entity_id" to node3Id))
            storage.upsertEdge(node2Id, node3Id, mapOf("relationship" to "includes"))

            // 8. Test removeEdges
            storage.removeEdges(listOf(node2Id to node3Id))
            assertNull(storage.getEdge(node2Id, node3Id))
            assertNull(storage.getEdge(node3Id, node2Id)) // Reverse should also be gone

            // 9. Test removeNodes
            storage.removeNodes(listOf(node2Id, node3Id))
            assertNull(storage.getNode(node2Id))
            assertNull(storage.getNode(node3Id))
        }
    }

    /** Ensures removing edges clears both directions without deleting nodes. */
    @Test
    fun `removeEdges clears both directions without deleting nodes`() {
        runBlocking {
            val storage =
                InMemoryGraphStorage(
                    namespace = "ns",
                    workspace = "ws",
                    embeddingFunc = TestEmbeddings.mockEmbeddingModel(),
                )
            storage.upsertNode("A", mapOf())
            storage.upsertNode("B", mapOf())
            storage.upsertEdge("A", "B", mapOf("weight" to "1"))

            storage.removeEdges(listOf("A" to "B"))

            assertTrue(storage.hasNode("A"))
            assertTrue(storage.hasNode("B"))
            assertFalse(storage.hasEdge("A", "B"))
            assertFalse(storage.hasEdge("B", "A"))
            assertEquals(emptyList(), storage.getAllEdges())
        }
    }

    /** Validates batch retrieval helpers for nodes, edges, and degrees. */
    @Test
    fun `test graph batch operations`() {
        runBlocking {
            val storage = createStorage()

            val node1Id = "Artificial Intelligence"
            val node2Id = "Machine Learning"
            val node3Id = "Deep Learning"

            storage.upsertNode(node1Id, mapOf("entity_id" to node1Id))
            storage.upsertNode(node2Id, mapOf("entity_id" to node2Id))
            storage.upsertNode(node3Id, mapOf("entity_id" to node3Id))

            storage.upsertEdge(node1Id, node2Id, mapOf("relationship" to "includes"))
            storage.upsertEdge(node2Id, node3Id, mapOf("relationship" to "includes"))

            // 2. Test getNodesBatch
            val nodesBatch = storage.getNodesBatch(listOf(node1Id, node2Id, node3Id))
            assertEquals(3, nodesBatch.size)
            assertNotNull(nodesBatch[node1Id])
            assertNotNull(nodesBatch[node2Id])
            assertNotNull(nodesBatch[node3Id])

            // 3. Test nodeDegreesBatch
            val nodeDegrees = storage.nodeDegreesBatch(listOf(node1Id, node2Id, node3Id))
            assertEquals(3, nodeDegrees.size)
            assertEquals(1, nodeDegrees[node1Id])
            assertEquals(2, nodeDegrees[node2Id])
            assertEquals(1, nodeDegrees[node3Id])

            // 4. Test edgeDegreesBatch
            val edgePairs = listOf(node1Id to node2Id, node2Id to node3Id)
            val edgeDegrees = storage.edgeDegreesBatch(edgePairs)
            assertEquals(2, edgeDegrees.size)
            assertEquals(3, edgeDegrees[node1Id to node2Id])
            assertEquals(3, edgeDegrees[node2Id to node3Id])

            // 5. Test getEdgesBatch
            val edgesBatch = storage.getEdgesBatch(listOf(mapOf("src" to node1Id, "tgt" to node2Id)))
            assertEquals(1, edgesBatch.size)
            assertNotNull(edgesBatch[node1Id to node2Id])

            // 6. Test getNodesEdgesBatch
            val nodesEdgesBatch = storage.getNodesEdgesBatch(listOf(node1Id, node2Id))
            assertEquals(2, nodesEdgesBatch.size)
            assertEquals(1, nodesEdgesBatch[node1Id]?.size)
            assertEquals(2, nodesEdgesBatch[node2Id]?.size)
        }
    }

    /** Checks label popularity and search utilities reflect degree counts and query filters. */
    @Test
    fun `popular and search labels reflect degrees and query matching`() {
        runBlocking {
            val storage =
                InMemoryGraphStorage(
                    namespace = "ns",
                    workspace = "ws",
                    embeddingFunc = TestEmbeddings.mockEmbeddingModel(),
                )
            storage.upsertNode("alpha", mapOf())
            storage.upsertNode("beta", mapOf())
            storage.upsertNode("gamma", mapOf())
            storage.upsertEdge("alpha", "beta", mapOf())
            storage.upsertEdge("alpha", "gamma", mapOf())

            val popular = storage.getPopularLabels(limit = 2)
            assertEquals("alpha", popular.first(), "Node with highest degree should be first")
            assertEquals(2, popular.size)

            val matches = storage.searchLabels("a", limit = 3)
            assertTrue(matches.containsAll(listOf("alpha", "gamma")))
        }
    }

    /** Knowledge graph traversal should honor depth/max_nodes and report truncation. */
    @Test
    fun `knowledge graph traversal returns connected component`() {
        runBlocking {
            val storage =
                InMemoryGraphStorage(
                    namespace = "ns",
                    workspace = "ws",
                    embeddingFunc = TestEmbeddings.mockEmbeddingModel(),
                )
            storage.upsertNode("A", mapOf("entity_id" to "A"))
            storage.upsertNode("B", mapOf("entity_id" to "B"))
            storage.upsertNode("C", mapOf("entity_id" to "C"))
            storage.upsertNode("D", mapOf("entity_id" to "D"))
            storage.upsertEdge("A", "B", mapOf("type" to "rel"))
            storage.upsertEdge("B", "C", mapOf("type" to "rel"))
            storage.upsertEdge("C", "D", mapOf("type" to "rel"))

            val depthLimited = storage.getKnowledgeGraph(nodeLabel = "A", maxDepth = 1, maxNodes = 10)
            assertEquals(2, depthLimited.nodes.size)
            assertEquals(1, depthLimited.edges.size)
            assertFalse(depthLimited.isTruncated)

            val truncated = storage.getKnowledgeGraph(nodeLabel = "A", maxDepth = 5, maxNodes = 2)
            assertEquals(2, truncated.nodes.size)
            assertEquals(1, truncated.edges.size)
            assertTrue(truncated.isTruncated)
        }
    }

    /** Confirms graph operations tolerate special characters in node identifiers. */
    @Test
    fun `test graph special characters`() {
        runBlocking {
            val storage = createStorage()

            val node1Id = "Node with 'single quotes'"
            val node1Data =
                mapOf(
                    "entity_id" to node1Id,
                    "description" to "This description contains 'single quotes', \"double quotes\", and \\backslashes",
                )
            storage.upsertNode(node1Id, node1Data)

            val nodeProps = storage.getNode(node1Id)
            assertNotNull(nodeProps)
            assertEquals(node1Id, nodeProps["entity_id"])
            assertEquals(node1Data["description"], nodeProps["description"])
        }
    }
}
