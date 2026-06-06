package com.youtu.graphrag.shared.treecomm

import com.youtu.graphrag.shared.graph.GraphNode
import com.youtu.graphrag.shared.graph.GraphRelationship
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

private data class TreeCommEdgeFixture(
    val source: String,
    val relation: String = "related_to",
    val target: String,
)

private data class TreeCommOptionsFixture(
    val embeddingModel: String? = null,
    val structWeight: Double? = null,
    val enableFastMode: Boolean? = null,
    val maxTotalCommunities: Int? = null,
    val enableSummary: Boolean? = null,
    val mergeThreshold: Double? = null,
    val maxIterations: Int? = null,
)

private data class TreeCommExpectedFixture(
    val communityCount: Int,
    val partitions: List<List<String>>,
)

private data class TreeCommParityFixture(
    val name: String,
    val edges: List<TreeCommEdgeFixture>,
    val options: TreeCommOptionsFixture = TreeCommOptionsFixture(),
    val expected: TreeCommExpectedFixture,
)

class TreeCommFixtureParityTest {
    @Test
    fun `treecomm fixtures lock community partitions in hierarchical mode`() {
        val fixtures = loadFixtures()
        assertTrue(fixtures.isNotEmpty())

        fixtures.forEach { fixture ->
            val relationships =
                fixture.edges.map { edge ->
                    GraphRelationship(
                        startNode = GraphNode(label = "entity", properties = mapOf("name" to edge.source)),
                        relation = edge.relation,
                        endNode = GraphNode(label = "entity", properties = mapOf("name" to edge.target)),
                    )
                }

            val options = fixture.options.toTreeCommOptions()
            val communities = FastTreeComm().detectCommunities(relationships, options)
            val partitions =
                communities
                    .mapNotNull { community ->
                        (community["nodes"] as? List<*>)?.mapNotNull { node -> node?.toString() }?.sorted()
                    }.sortedWith(compareBy<List<String>> { it.size }.thenBy { it.firstOrNull().orEmpty() })

            val expectedPartitions =
                fixture.expected.partitions
                    .map { partition -> partition.sorted() }
                    .sortedWith(compareBy<List<String>> { it.size }.thenBy { it.firstOrNull().orEmpty() })

            assertEquals(
                fixture.expected.communityCount,
                communities.size,
                "community count parity failed for fixture '${fixture.name}'",
            )
            assertEquals(
                expectedPartitions,
                partitions,
                "partition parity failed for fixture '${fixture.name}'",
            )
        }
    }

    private fun loadFixtures(): List<TreeCommParityFixture> =
        listOf(
            TreeCommParityFixture(
                name = "hierarchical_multi_component_partition",
                edges =
                    listOf(
                        TreeCommEdgeFixture(source = "Alice", relation = "works_at", target = "Acme"),
                        TreeCommEdgeFixture(source = "Acme", relation = "located_in", target = "Tokyo"),
                        TreeCommEdgeFixture(source = "Bob", relation = "paired_with", target = "Carol"),
                    ),
                options =
                    TreeCommOptionsFixture(
                        enableFastMode = false,
                        structWeight = 0.3,
                        mergeThreshold = 0.0,
                        maxIterations = 6,
                        maxTotalCommunities = 2,
                    ),
                expected =
                    TreeCommExpectedFixture(
                        communityCount = 2,
                        partitions =
                            listOf(
                                listOf("Alice", "Acme", "Tokyo"),
                                listOf("Bob", "Carol"),
                            ),
                    ),
            ),
        )

    private fun TreeCommOptionsFixture.toTreeCommOptions(): TreeCommOptions =
        TreeCommOptions(
            embeddingModel = embeddingModel ?: TreeCommOptions().embeddingModel,
            structWeight = structWeight ?: TreeCommOptions().structWeight,
            enableFastMode = enableFastMode ?: TreeCommOptions().enableFastMode,
            maxTotalCommunities = maxTotalCommunities ?: TreeCommOptions().maxTotalCommunities,
            enableSummary = enableSummary ?: TreeCommOptions().enableSummary,
            mergeThreshold = mergeThreshold ?: TreeCommOptions().mergeThreshold,
            maxIterations = maxIterations ?: TreeCommOptions().maxIterations,
        )
}
