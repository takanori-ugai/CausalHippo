package ragas.llms

import ragas.cache.CacheBackend
import ragas.cache.stableCacheKey
import ragas.runtime.RunConfig

open class CachedRagasLlm private constructor(
    internal val delegate: BaseRagasLlm,
    internal val cache: CacheBackend,
) : BaseRagasLlm {
    override var runConfig: RunConfig
        get() = delegate.runConfig
        set(value) {
            delegate.runConfig = value
        }

    /**
     * Generates one or more text completions for the supplied prompt.
     */
    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        val key =
            stableCacheKey(
                buildString {
                    append("llm|")
                    append(prompt)
                    append("|n=")
                    append(n)
                    append("|temp=")
                    append(temperature)
                    append("|stop=")
                    append(stop?.joinToString(",") ?: "")
                },
            )

        val cached = cache.get(key)
        if (cached is LlmResult) {
            return cached
        }

        val result = delegate.generateText(prompt, n, temperature, stop)
        cache.put(key, result)
        return result
    }

    companion object {
        operator fun invoke(
            delegate: BaseRagasLlm,
            cache: CacheBackend,
        ): CachedRagasLlm =
            if (delegate is StructuredOutputRagasLlm) {
                CachedStructuredRagasLlm(delegate, cache)
            } else {
                CachedRagasLlm(delegate, cache)
            }
    }

    private class CachedStructuredRagasLlm(
        delegate: BaseRagasLlm,
        cache: CacheBackend,
    ) : CachedRagasLlm(delegate, cache),
        StructuredOutputRagasLlm {
        private val structuredDelegate =
            delegate as? StructuredOutputRagasLlm
                ?: error("Delegate LLM does not support structured output.")

        override suspend fun generateNumericValue(prompt: String): Double? {
            val key = stableCacheKey("llm-structured|numeric|$prompt")
            val cached = cache.get(key)
            if (cached is Double) {
                return cached
            }
            val result = structuredDelegate.generateNumericValue(prompt)
            if (result != null) {
                cache.put(key, result)
            }
            return result
        }

        override suspend fun generateDiscreteValue(prompt: String): String? {
            val key = stableCacheKey("llm-structured|discrete|$prompt")
            val cached = cache.get(key)
            if (cached is String) {
                return cached
            }
            val result = structuredDelegate.generateDiscreteValue(prompt)
            if (result != null) {
                cache.put(key, result)
            }
            return result
        }

        override suspend fun generateRankingItems(prompt: String): List<String>? {
            val key = stableCacheKey("llm-structured|ranking|$prompt")
            val cached = cache.get(key)
            if (cached is List<*>) {
                @Suppress("UNCHECKED_CAST")
                return cached.filterIsInstance<String>()
            }
            val result = structuredDelegate.generateRankingItems(prompt)
            if (result != null) {
                cache.put(key, result)
            }
            return result
        }
    }
}
