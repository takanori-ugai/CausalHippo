package ragas.testset.transforms

import kotlinx.coroutines.test.runTest
import ragas.testset.graph.KnowledgeGraph
import ragas.testset.graph.Node
import ragas.testset.graph.NodeType
import ragas.testset.graph.Relationship
import kotlin.test.Test
import kotlin.test.assertEquals

class RelationshipBuilderDedupTest {
    @Test
    fun dedupDistinguishesBidirectionalFlag() =
        runTest {
            val source = Node(id = "source", type = NodeType.DOCUMENT)
            val target = Node(id = "target", type = NodeType.CHUNK)
            val kg = KnowledgeGraph(nodes = mutableListOf(source, target))

            val builder =
                object : RelationshipBuilder(name = "test_builder") {
                    override suspend fun build(
                        kg: KnowledgeGraph,
                        filtered: KnowledgeGraph,
                    ): List<Relationship> =
                        listOf(
                            Relationship(
                                type = "semantic_overlap",
                                sourceId = source.id,
                                targetId = target.id,
                                bidirectional = false,
                                properties = mapOf("shared_keyword_count" to "2"),
                            ),
                            Relationship(
                                type = "semantic_overlap",
                                sourceId = source.id,
                                targetId = target.id,
                                bidirectional = true,
                                properties = mapOf("shared_keyword_count" to "2"),
                            ),
                        )
                }

            applyTransforms(kg, SingleTransform(builder))

            assertEquals(2, kg.relationships.size)
            assertEquals(setOf(false, true), kg.relationships.map { rel -> rel.bidirectional }.toSet())
        }
}
