package ragas.prompt

import kotlinx.coroutines.runBlocking
import ragas.embeddings.BaseRagasEmbedding
import kotlin.test.Test
import kotlin.test.assertEquals

class DynamicFewShotPromptCachingTest {
    private class CountingEmbedding : BaseRagasEmbedding {
        var embedTextCount = 0
        var embedTextsCount = 0

        override suspend fun embedText(text: String): List<Float> {
            embedTextCount++
            return listOf(0f)
        }

        override suspend fun embedTexts(texts: List<String>): List<List<Float>> {
            embedTextsCount++
            return texts.map { listOf(0f) }
        }
    }

    @Test
    fun testRedundantEmbeddingCalls() =
        runBlocking {
            val embedding = CountingEmbedding()
            val prompt =
                DynamicFewShotPrompt(
                    instruction = "test",
                    examples =
                        listOf(
                            PromptExample(mapOf("input" to "1"), mapOf("output" to "1")),
                            PromptExample(mapOf("input" to "2"), mapOf("output" to "2")),
                        ),
                    embeddings = embedding,
                )

            // First call to format
            prompt.format(mapOf("input" to "query1"))
            assertEquals(1, embedding.embedTextsCount, "embedTexts should be called once on first format")

            // Second call to format
            prompt.format(mapOf("input" to "query2"))
            // Currently this will fail because it's called twice
            assertEquals(1, embedding.embedTextsCount, "embedTexts should NOT be called again on second format")
        }

    @Test
    fun testCacheInvalidationOnEmbeddingChange() =
        runBlocking {
            val embedding1 = CountingEmbedding()
            val embedding2 = CountingEmbedding()
            val prompt =
                DynamicFewShotPrompt(
                    instruction = "test",
                    examples =
                        listOf(
                            PromptExample(mapOf("input" to "1"), mapOf("output" to "1")),
                        ),
                    embeddings = embedding1,
                )

            prompt.format(mapOf("input" to "query1"))
            assertEquals(1, embedding1.embedTextsCount)

            prompt.embeddings = embedding2
            prompt.format(mapOf("input" to "query2"))
            assertEquals(1, embedding2.embedTextsCount, "embedTexts should be called once for new embedding model")
        }
}
