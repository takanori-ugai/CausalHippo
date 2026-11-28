package causalrag.causalgraph.builder

import causalrag.generator.llm.LLMInterface
import causalrag.utils.SimpleHashEmbedding
import io.mockk.every
import io.mockk.mockk
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class IndexingReportTest {
    @Test
    fun `indexing reports extraction failures while retaining successful triples`() {
        val llm = mockk<LLMInterface>()
        every { llm.generate(any(), 1.0, 800, false, true, true) } returnsMany
            listOf("malformed", """[{"cause":"heavy rain","effect":"flooding","confidence":0.9}]""", "[]")
        val builder =
            CausalGraphBuilder(extractorMethod = "llm", llmInterface = llm, embeddingModel = SimpleHashEmbedding())

        val edges = builder.indexDocuments(listOf("Extraction fails for this document.", "Heavy rain causes flooding."))
        assertEquals(1, edges)
        assertEquals(1, builder.lastIndexingReport.failedChunkCount)
        val failure = builder.lastIndexingReport.failures.single()
        assertEquals("0:0", failure.chunk)
        assertTrue(failure.reason.contains("JSON"))

        builder.indexDocuments(listOf("A valid response may contain no relationships."))
        assertEquals(0, builder.lastIndexingReport.failedChunkCount)
    }
}
