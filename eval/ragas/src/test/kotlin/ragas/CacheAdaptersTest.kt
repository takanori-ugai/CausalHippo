package ragas

import kotlinx.coroutines.runBlocking
import ragas.cache.InMemoryCacheBackend
import ragas.cache.stableCacheKey
import ragas.embeddings.BaseRagasEmbedding
import ragas.embeddings.CachedRagasEmbedding
import ragas.llms.BaseRagasLlm
import ragas.llms.CachedRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals

class CacheAdaptersTest {
    @Test
    fun cachedLlmReturnsCachedResultOnRepeatedRequest() =
        runBlocking {
            val base = CountingLlm("hello")
            val cached = CachedRagasLlm(base, InMemoryCacheBackend())

            cached.generateText("prompt")
            cached.generateText("prompt")

            assertEquals(1, base.calls)
        }

    @Test
    fun cachedEmbeddingReturnsCachedVectorOnRepeatedRequest() =
        runBlocking {
            val base = CountingEmbedding(listOf(0.1f, 0.2f))
            val cached = CachedRagasEmbedding(base, InMemoryCacheBackend())

            cached.embedText("text")
            cached.embedText("text")

            assertEquals(1, base.calls)
        }

    @Test
    fun cachedEmbeddingAcceptsNumericCachePayloads() =
        runBlocking {
            val cache = InMemoryCacheBackend()
            cache.put(stableCacheKey("embedding|text"), listOf(0.1, 0.2))

            val base = CountingEmbedding(listOf(9.9f, 9.9f))
            val cached = CachedRagasEmbedding(base, cache)
            val result = cached.embedText("text")

            assertEquals(listOf(0.1f, 0.2f), result)
            assertEquals(0, base.calls)
        }

    @Test
    fun cachedEmbeddingBatchesCacheMissesInEmbedTexts() =
        runBlocking {
            val cache = InMemoryCacheBackend()
            cache.put(stableCacheKey("embedding|hit"), listOf(1.0, 2.0))
            val base = BatchCountingEmbedding()
            val cached = CachedRagasEmbedding(base, cache)

            val result = cached.embedTexts(listOf("hit", "miss1", "miss2"))

            assertEquals(listOf(1.0f, 2.0f), result[0])
            assertEquals(listOf(5.0f), result[1])
            assertEquals(listOf(5.0f), result[2])
            assertEquals(0, base.embedTextCalls)
            assertEquals(1, base.embedTextsCalls)
            assertEquals(listOf("miss1", "miss2"), base.lastBatchInputs)
        }
}

private class CountingLlm(
    private val text: String,
) : BaseRagasLlm {
    var calls: Int = 0

    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        calls += 1
        return LlmResult(generations = listOf(LlmGeneration(text = text)))
    }
}

private class CountingEmbedding(
    private val vector: List<Float>,
) : BaseRagasEmbedding {
    var calls: Int = 0

    override suspend fun embedText(text: String): List<Float> {
        calls += 1
        return vector
    }
}

private class BatchCountingEmbedding : BaseRagasEmbedding {
    var embedTextCalls: Int = 0
    var embedTextsCalls: Int = 0
    var lastBatchInputs: List<String> = emptyList()

    override suspend fun embedText(text: String): List<Float> {
        embedTextCalls += 1
        return listOf(text.length.toFloat())
    }

    override suspend fun embedTexts(texts: List<String>): List<List<Float>> {
        embedTextsCalls += 1
        lastBatchInputs = texts
        return texts.map { text -> listOf(text.length.toFloat()) }
    }
}
