package ragas.llms

import kotlinx.coroutines.runBlocking
import ragas.cache.InMemoryCacheBackend
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CachedStructuredLlmTest {
    @Test
    fun cachedLlmExposesStructuredOnlyIfDelegateDoes() =
        runBlocking {
            val base = PlainLlm()
            val cache = InMemoryCacheBackend()
            val cached = CachedRagasLlm(base, cache)

            assertFalse(
                cached is StructuredOutputRagasLlm,
                "CachedRagasLlm should not implement StructuredOutputRagasLlm if delegate doesn't",
            )
        }

    @Test
    fun cachedLlmExposesStructuredIfDelegateDoes() =
        runBlocking {
            val base = StructuredLlm()
            val cache = InMemoryCacheBackend()

            val cached = CachedRagasLlm(base, cache)

            assertTrue(
                cached is StructuredOutputRagasLlm,
                "CachedRagasLlm should implement StructuredOutputRagasLlm if delegate does",
            )
        }
}

private class PlainLlm : BaseRagasLlm {
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult = LlmResult(emptyList())
}

private class StructuredLlm :
    BaseRagasLlm,
    StructuredOutputRagasLlm {
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult = LlmResult(emptyList())

    override suspend fun generateNumericValue(prompt: String): Double? = 0.0

    override suspend fun generateDiscreteValue(prompt: String): String? = ""

    override suspend fun generateRankingItems(prompt: String): List<String> = emptyList()
}
