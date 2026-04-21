package lightrag.kg.memory

import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import kotlinx.coroutines.runBlocking
import lightrag.kg.neo4j.Neo4jGraphStorage
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.String
import kotlin.test.Ignore
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Integration-style checks for Neo4j graph storage (ignored unless a database is available). */
@Ignore("Requires Neo4j database")
class Neo4jGraphStorageTest {
    val apiKey = "***************************************************"

    init {
        if (apiKey.isNullOrBlank()) {
            println("Error: OPENAI_API_KEY environment variable is not set.")
        }
    }

    private lateinit var storage: Neo4jGraphStorage

    /** Sets up a Neo4jGraphStorage instance and clears existing data. */
    @Before
    fun setUp() {
        storage =
            Neo4jGraphStorage(
                namespace = "chunk_entity_relation_graph",
                globalConfig =
                    mapOf<String, Any>(
                        "neo4j" to
                            mapOf(
                                "uri" to "neo4j://localhost:7687",
                                "username" to "neo4j",
                                "password" to "Takasan0",
                            ),
                    ),
                embeddingFunc =
                    OpenAiEmbeddingModel
                        .builder()
                        .apiKey(apiKey)
                        .modelName("text-embedding-3-large")
                        .dimensions(3072)
                        .build(),
            )
        runBlocking {
            storage.initialize()
            storage.drop()
        }
    }

    /** Closes the Neo4j driver after each test run. */
    @After
    fun tearDown() {
        runBlocking {
            storage.finalize()
        }
    }

    /** Validates basic graph CRUD operations against Neo4j storage. */
    @Test
    fun `test basic graph operations`() {
        runBlocking {
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

    /** Exercises degree calculations, adjacency, and label retrieval against Neo4j storage. */
    @Test
    fun `test advanced graph operations`() {
        runBlocking {
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

            // 6. Test getKnowledgeGraph (placeholder in InMemoryGraphStorage, but we can test it returns something)
            val kg = storage.getKnowledgeGraph("*", 2, 10)
            assertNotNull(kg)
            // InMemoryGraphStorage implementation returns empty KG currently, so we just check type/existence
            // If implementation changes, update assertions

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

    //  @Ignore("Requires Neo4j database")
    @Test
    fun `test graph batch operations`() {
        runBlocking {
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

    @Test
    fun `test graph special characters`() {
        runBlocking {
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
