package ragas.testset.transforms

import kotlinx.coroutines.runBlocking
import ragas.testset.graph.Node
import ragas.testset.graph.NodeType
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SentenceChunkSplitterTest {
    @Test
    fun splitKeepsAbbreviationsAndDecimalsIntact() =
        runBlocking {
            val splitter = SentenceChunkSplitter(maxSentencesPerChunk = 1)
            val node =
                Node(
                    type = NodeType.DOCUMENT,
                    properties =
                        mutableMapOf(
                            "page_content" to "Mr. Smith paid 3.14 dollars. He arrived at 5 p.m. sharp.",
                        ),
                )

            val (chunks, _) = splitter.split(node)
            val chunkTexts = chunks.map { chunk -> chunk.getProperty("page_content").orEmpty() }

            assertTrue(chunkTexts.any { text -> "3.14 dollars." in text })
            assertFalse(chunkTexts.any { text -> text == "Mr." })
            assertFalse(chunkTexts.any { text -> text == "14 dollars." })
        }

    @Test
    fun splitHandlesTrailingAbbreviationSentence() =
        runBlocking {
            val splitter = SentenceChunkSplitter(maxSentencesPerChunk = 1)
            val node =
                Node(
                    type = NodeType.DOCUMENT,
                    properties =
                        mutableMapOf(
                            "page_content" to "I talked to Mr. Smith today. We met at noon.",
                        ),
                )

            val (chunks, _) = splitter.split(node)
            val chunkTexts = chunks.map { chunk -> chunk.getProperty("page_content").orEmpty() }

            assertTrue(chunkTexts.any { text -> "Mr. Smith today." in text })
            assertTrue(chunkTexts.any { text -> "We met at noon." in text })
            assertFalse(chunkTexts.any { text -> text == "Mr." })
        }

    @Test
    fun splitHandlesMultipleConsecutiveAbbreviations() =
        runBlocking {
            val splitter = SentenceChunkSplitter(maxSentencesPerChunk = 1)
            val node =
                Node(
                    type = NodeType.DOCUMENT,
                    properties =
                        mutableMapOf(
                            "page_content" to "He lives in the U.S.A. It is large.",
                        ),
                )

            val (chunks, _) = splitter.split(node)
            val chunkTexts = chunks.map { chunk -> chunk.getProperty("page_content").orEmpty() }

            assertTrue(chunkTexts.any { text -> text.contains("U.S.A.") })
            assertFalse(chunkTexts.any { text -> text == "U." || text == "S." || text == "A." })
        }

    @Test
    fun splitReturnsNoChunksForBlankContent() =
        runBlocking {
            val splitter = SentenceChunkSplitter(maxSentencesPerChunk = 1)
            val node =
                Node(
                    type = NodeType.DOCUMENT,
                    properties = mutableMapOf("page_content" to "   "),
                )

            val (chunks, relationships) = splitter.split(node)

            assertTrue(chunks.isEmpty())
            assertTrue(relationships.isEmpty())
        }

    @Test
    fun splitGroupsSentencesWhenMaxSentencesPerChunkIsGreaterThanOne() =
        runBlocking {
            val splitter = SentenceChunkSplitter(maxSentencesPerChunk = 2)
            val node =
                Node(
                    type = NodeType.DOCUMENT,
                    properties =
                        mutableMapOf(
                            "page_content" to "First sentence. Second sentence! Third sentence?",
                        ),
                )

            val (chunks, _) = splitter.split(node)
            val chunkTexts = chunks.map { chunk -> chunk.getProperty("page_content").orEmpty() }

            assertTrue(chunkTexts.contains("First sentence. Second sentence!"))
            assertTrue(chunkTexts.contains("Third sentence?"))
        }
}
