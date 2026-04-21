package causalrag

import causalrag.retriever.HippoRagSemanticMode
import causalrag.retriever.HippoRagSemanticRetrieverAdapter
import hipporag.HippoRag
import hipporag.utils.QuerySolution
import io.mockk.every
import io.mockk.mockk
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TestHippoRagSemanticRetrieverAdapter {
    @Test
    fun testGraphModeMapsHippoResultsToPassageScores() {
        val hippoRag = mockk<HippoRag>()
        every { hippoRag.retrieve(listOf("what causes flooding?"), 3, null) } returns
            (
                listOf(
                    QuerySolution(
                        question = "what causes flooding?",
                        docs = listOf("Heavy rainfall causes flooding.", "Overflowing rivers increase flood risk."),
                        docScores = doubleArrayOf(0.9, 0.6),
                    ),
                ) to null
            )

        val adapter = HippoRagSemanticRetrieverAdapter(hippoRag, mode = HippoRagSemanticMode.GRAPH)
        val results = adapter.searchWithScores("what causes flooding?", topK = 3)

        assertEquals(2, results.size)
        assertEquals("Heavy rainfall causes flooding." to 0.9, results[0])
        assertEquals("Overflowing rivers increase flood risk." to 0.6, results[1])
    }

    @Test
    fun testGraphModeFallsBackToDprWhenConfigured() {
        val hippoRag = mockk<HippoRag>()
        every { hippoRag.retrieve(listOf("how to reduce flooding?"), 2, null) } returns
            (listOf(QuerySolution(question = "how to reduce flooding?", docs = emptyList(), docScores = doubleArrayOf())) to null)
        every { hippoRag.retrieveDpr(listOf("how to reduce flooding?"), 2, null) } returns
            (
                listOf(
                    QuerySolution(
                        question = "how to reduce flooding?",
                        docs = listOf("Retention basins reduce runoff."),
                        docScores = doubleArrayOf(0.75),
                    ),
                ) to null
            )

        val adapter = HippoRagSemanticRetrieverAdapter(hippoRag, mode = HippoRagSemanticMode.GRAPH)
        val results = adapter.searchWithScores("how to reduce flooding?", topK = 2)

        assertEquals(1, results.size)
        assertTrue(results.first().first.contains("Retention basins"))
        assertEquals(0.75, results.first().second)
    }
}
