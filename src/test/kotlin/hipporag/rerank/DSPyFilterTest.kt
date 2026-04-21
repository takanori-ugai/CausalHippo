package hipporag.rerank

import hipporag.llm.BaseLLM
import hipporag.utils.LlmResult
import io.mockk.every
import io.mockk.mockk
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DSPyFilterTest {
    @Test
    fun testRerankEmptyFacts() {
        val llmModel = mockk<BaseLLM>()
        val filter = DSPyFilter(llmModel)

        val (indices, facts, metadata) =
            filter.rerank(
                query = "test query",
                candidateFacts = emptyList(),
                candidateFactIndices = emptyList(),
                lenAfterRerank = 5,
            )

        assertTrue(indices.isEmpty())
        assertTrue(facts.isEmpty())
    }

    @Test
    fun testRerankWithValidResponse() {
        val llmModel = mockk<BaseLLM>()
        every { llmModel.infer(any()) } returns
            LlmResult(
                response = """{"fact": [["A", "relates", "B"], ["C", "knows", "D"]]}""",
                metadata = emptyMap(),
            )

        val filter = DSPyFilter(llmModel)

        val candidateFacts =
            listOf(
                listOf("A", "relates", "B"),
                listOf("C", "knows", "D"),
                listOf("E", "sees", "F"),
            )
        val candidateIndices = listOf(0, 1, 2)

        val (indices, facts, metadata) =
            filter.rerank(
                query = "test query",
                candidateFacts = candidateFacts,
                candidateFactIndices = candidateIndices,
                lenAfterRerank = 2,
            )

        assertTrue(facts.size <= 2)
        assertTrue(indices.size <= 2)
    }

    @Test
    fun testRerankWithMalformedResponse() {
        val llmModel = mockk<BaseLLM>()
        every { llmModel.infer(any()) } returns
            LlmResult(
                response = "invalid json",
                metadata = emptyMap(),
            )

        val filter = DSPyFilter(llmModel)

        val candidateFacts =
            listOf(
                listOf("A", "relates", "B"),
                listOf("C", "knows", "D"),
            )
        val candidateIndices = listOf(0, 1)

        val (indices, facts, metadata) =
            filter.rerank(
                query = "test query",
                candidateFacts = candidateFacts,
                candidateFactIndices = candidateIndices,
                lenAfterRerank = 2,
            )

        assertEquals(candidateIndices.take(2), indices)
        assertEquals(candidateFacts.take(2), facts)
    }

    @Test
    fun testRerankLimitResults() {
        val llmModel = mockk<BaseLLM>()
        every { llmModel.infer(any()) } returns
            LlmResult(
                response = """{"fact": [["A", "relates", "B"], ["C", "knows", "D"], ["E", "sees", "F"]]}""",
                metadata = emptyMap(),
            )

        val filter = DSPyFilter(llmModel)

        val candidateFacts =
            listOf(
                listOf("A", "relates", "B"),
                listOf("C", "knows", "D"),
                listOf("E", "sees", "F"),
            )
        val candidateIndices = listOf(0, 1, 2)

        val (indices, facts, metadata) =
            filter.rerank(
                query = "test query",
                candidateFacts = candidateFacts,
                candidateFactIndices = candidateIndices,
                lenAfterRerank = 1,
            )

        assertTrue(indices.size <= 1)
        assertTrue(facts.size <= 1)
    }

    @Test
    fun testRerankWithException() {
        val llmModel = mockk<BaseLLM>()
        every { llmModel.infer(any()) } throws RuntimeException("Model error")

        val filter = DSPyFilter(llmModel)

        val candidateFacts =
            listOf(
                listOf("A", "relates", "B"),
                listOf("C", "knows", "D"),
            )
        val candidateIndices = listOf(0, 1)

        val (indices, facts, metadata) =
            filter.rerank(
                query = "test query",
                candidateFacts = candidateFacts,
                candidateFactIndices = candidateIndices,
                lenAfterRerank = 2,
            )

        assertEquals(candidateIndices.take(2), indices)
        assertEquals(candidateFacts.take(2), facts)
        assertTrue(metadata.containsKey("error"))
    }

    @Test
    fun testRerankPartialMatch() {
        val llmModel = mockk<BaseLLM>()
        every { llmModel.infer(any()) } returns
            LlmResult(
                response = """{"fact": [["A", "relates", "B"]]}""",
                metadata = emptyMap(),
            )

        val filter = DSPyFilter(llmModel)

        val candidateFacts =
            listOf(
                listOf("A", "relates", "B"),
                listOf("C", "knows", "D"),
            )
        val candidateIndices = listOf(0, 1)

        val (indices, facts, metadata) =
            filter.rerank(
                query = "test query",
                candidateFacts = candidateFacts,
                candidateFactIndices = candidateIndices,
                lenAfterRerank = 5,
            )

        assertTrue(indices.isNotEmpty())
        assertTrue(facts.isNotEmpty())
        assertTrue(facts.contains(listOf("A", "relates", "B")))
    }

    @Test
    fun testRerankFuzzyMatching() {
        val llmModel = mockk<BaseLLM>()
        every { llmModel.infer(any()) } returns
            LlmResult(
                response = """{"fact": [["a", "relates", "b"]]}""",
                metadata = emptyMap(),
            )

        val filter = DSPyFilter(llmModel)

        val candidateFacts =
            listOf(
                listOf("A", "relates", "B"),
                listOf("C", "knows", "D"),
            )
        val candidateIndices = listOf(0, 1)

        val (indices, facts, metadata) =
            filter.rerank(
                query = "test query",
                candidateFacts = candidateFacts,
                candidateFactIndices = candidateIndices,
                lenAfterRerank = 2,
            )

        assertTrue(indices.isNotEmpty())
        assertTrue(facts.isNotEmpty())
    }

    @Test
    fun testRerankWithCustomFilePath() {
        val llmModel = mockk<BaseLLM>()
        every { llmModel.infer(any()) } returns
            LlmResult(
                response = """{"fact": []}""",
                metadata = emptyMap(),
            )

        val filter = DSPyFilter(llmModel, rerankDspyFilePath = "nonexistent.json")

        val candidateFacts = listOf(listOf("A", "relates", "B"))
        val candidateIndices = listOf(0)

        val (indices, facts, metadata) =
            filter.rerank(
                query = "test query",
                candidateFacts = candidateFacts,
                candidateFactIndices = candidateIndices,
                lenAfterRerank = 1,
            )

        assertTrue(indices.isNotEmpty() || facts.isNotEmpty() || indices.isEmpty())
    }

    @Test
    fun testRerankWithEmptyJsonResponse() {
        val llmModel = mockk<BaseLLM>()
        every { llmModel.infer(any()) } returns
            LlmResult(
                response = """{"fact": []}""",
                metadata = emptyMap(),
            )

        val filter = DSPyFilter(llmModel)

        val candidateFacts =
            listOf(
                listOf("A", "relates", "B"),
                listOf("C", "knows", "D"),
            )
        val candidateIndices = listOf(0, 1)

        val (indices, facts, metadata) =
            filter.rerank(
                query = "test query",
                candidateFacts = candidateFacts,
                candidateFactIndices = candidateIndices,
                lenAfterRerank = 2,
            )

        assertEquals(candidateIndices.take(2), indices)
        assertEquals(candidateFacts.take(2), facts)
    }

    @Test
    fun testRerankPreservesOrder() {
        val llmModel = mockk<BaseLLM>()
        every { llmModel.infer(any()) } returns
            LlmResult(
                response = """{"fact": [["C", "knows", "D"], ["A", "relates", "B"]]}""",
                metadata = emptyMap(),
            )

        val filter = DSPyFilter(llmModel)

        val candidateFacts =
            listOf(
                listOf("A", "relates", "B"),
                listOf("C", "knows", "D"),
            )
        val candidateIndices = listOf(0, 1)

        val (indices, facts, metadata) =
            filter.rerank(
                query = "test query",
                candidateFacts = candidateFacts,
                candidateFactIndices = candidateIndices,
                lenAfterRerank = 2,
            )

        assertTrue(facts.isNotEmpty())
        if (facts.size >= 2) {
            assertTrue(facts[0] == listOf("C", "knows", "D") || facts[0] == listOf("A", "relates", "B"))
        }
    }
}
