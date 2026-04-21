package ragas.cache

/**
 * Cache backend for storing arbitrary values by string key.
 *
 * Implementations are expected to be safe for concurrent access because callers
 * may read/write cache entries from multiple coroutines or threads.
 */
interface CacheBackend {
    /**
     * Returns the cached value for a key, or null when missing.
     *
     * @param key Lookup key.
     */
    fun get(key: String): Any?

    /**
     * Stores a value in cache under the provided key.
     *
     * @param key Lookup key.
     * @param value Value payload.
     */
    fun put(
        key: String,
        value: Any?,
    )
}
