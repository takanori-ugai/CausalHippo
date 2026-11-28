package lightrag.operate

import dev.langchain4j.model.embedding.EmbeddingModel
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import lightrag.core.Constants
import lightrag.core.types.BaseVectorStorage
import lightrag.kg.memory.InMemoryGraphStorage
import lightrag.utils.computeMd5
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class GraphMergeTest {
    @Test
    fun `reversed relations share one vector ID and missing endpoints are created`() {
        runBlocking {
            val graph = InMemoryGraphStorage("graph", "test", embeddingFunc = mockk<EmbeddingModel>())
            val entities = mockk<BaseVectorStorage>(relaxed = true)
            val relations = mockk<BaseVectorStorage>(relaxed = true)
            for ((source, target, chunk) in listOf(Triple("A", "B", "first"), Triple("B", "A", "second"))) {
                val extraction = ExtractionResult(relations = listOf(ExtractedRelation(source, target, "related", "Evidence $chunk")))
                val (nodes, edges) = processExtractionResult(extraction, chunk)
                mergeNodesAndEdges(nodes, edges, graph, entities, relations)
            }

            assertEquals("UNKNOWN", graph.getNode("A")?.get("entity_type"))
            assertEquals("UNKNOWN", graph.getNode("B")?.get("entity_type"))
            val edge = assertNotNull(graph.getEdge("A", "B"))
            assertEquals(setOf("first", "second"), edge.getValue("source_id").split(Constants.GRAPH_FIELD_SEP).toSet())
            assertEquals("2.0", edge["weight"])
            assertEquals(1, graph.getAllEdges().size)
            coVerify(exactly = 2) { relations.upsert(match { it.keys == setOf(computeMd5("A#B")) }) }
            coVerify(exactly = 0) { relations.upsert(match { computeMd5("B#A") in it }) }
        }
    }
}
