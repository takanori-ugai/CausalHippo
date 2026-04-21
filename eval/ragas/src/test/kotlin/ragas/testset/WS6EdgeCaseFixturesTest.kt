package ragas.testset

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.model.SingleTurnSample
import ragas.testset.graph.NodeType
import kotlin.test.Test
import kotlin.test.assertTrue

class WS6EdgeCaseFixturesTest {
    @Test
    fun sparseOverlapFixtureConformance() =
        runBlocking {
            val fixture = WS6TestFixtures.readFixture("ws6_sparse_overlap_fixture.json")
            val run = WS6TestFixtures.executeFixture(fixture)
            val expected = fixture.jsonObject.getValue("expected").jsonObject

            assertTrue(run.chunkCount >= expected.getValue("min_chunk_nodes").jsonPrimitive.int)
            assertTrue(run.overlapRelationshipCount <= expected.getValue("max_overlap_relationships").jsonPrimitive.int)
            assertTrue(run.nextRelationshipCount >= expected.getValue("min_next_relationships").jsonPrimitive.int)

            val requiredPrefixes =
                expected
                    .getValue("required_synthesizer_prefixes")
                    .jsonArray
                    .map { item -> item.jsonPrimitive.content }
            val synthesizerSet =
                run.testset.samples
                    .map { sample -> sample.synthesizerName }
                    .toSet()
            assertTrue(requiredPrefixes.all { prefix -> synthesizerSet.any { name -> name.startsWith(prefix) } })

            val minResponseChars = expected.getValue("min_response_characters").jsonPrimitive.int
            run.testset.samples.forEach { sample ->
                val response = (sample.evalSample as SingleTurnSample).response.orEmpty()
                assertTrue(response.length >= minResponseChars)
            }
        }

    @Test
    fun longDocumentFixtureConformance() =
        runBlocking {
            val fixture = WS6TestFixtures.readFixture("ws6_long_document_fixture.json")
            val run = WS6TestFixtures.executeFixture(fixture)
            val expected = fixture.jsonObject.getValue("expected").jsonObject

            assertTrue(run.chunkCount >= expected.getValue("min_chunk_nodes").jsonPrimitive.int)
            assertTrue(run.chunkCount <= expected.getValue("max_chunk_nodes").jsonPrimitive.int)
            assertTrue(run.childRelationshipCount >= expected.getValue("min_child_relationships").jsonPrimitive.int)
            assertTrue(run.nextRelationshipCount >= expected.getValue("min_next_relationships").jsonPrimitive.int)

            val multiHopCount = run.testset.samples.count { sample -> sample.synthesizerName.startsWith("multi_hop_") }
            assertTrue(multiHopCount <= expected.getValue("max_multi_hop_samples").jsonPrimitive.int)

            val avgChunkChars =
                run.graph.nodes
                    .filter { node -> node.type == NodeType.CHUNK }
                    .map { node ->
                        node
                            .getProperty("page_content")
                            .orEmpty()
                            .length
                            .toDouble()
                    }.average()
            assertTrue(avgChunkChars >= expected.getValue("min_avg_chunk_characters").jsonPrimitive.double)
        }

    @Test
    fun relationshipDensityFixtureConformance() =
        runBlocking {
            val fixture = WS6TestFixtures.readFixture("ws6_relationship_density_fixture.json")
            val run = WS6TestFixtures.executeFixture(fixture)
            val expected = fixture.jsonObject.getValue("expected").jsonObject

            assertTrue(run.chunkCount >= expected.getValue("min_chunk_nodes").jsonPrimitive.int)
            assertTrue(run.overlapRelationshipCount >= expected.getValue("min_overlap_relationships").jsonPrimitive.int)

            val totalPairs = run.chunkCount * (run.chunkCount - 1) / 2
            val density =
                if (totalPairs == 0) {
                    0.0
                } else {
                    run.overlapRelationshipCount.toDouble() / totalPairs.toDouble()
                }
            assertTrue(density >= expected.getValue("semantic_overlap_density_min").jsonPrimitive.double)
            assertTrue(density <= expected.getValue("semantic_overlap_density_max").jsonPrimitive.double)

            val multiHopCount = run.testset.samples.count { sample -> sample.synthesizerName.startsWith("multi_hop_") }
            assertTrue(multiHopCount >= expected.getValue("min_multi_hop_samples").jsonPrimitive.int)
        }
}
