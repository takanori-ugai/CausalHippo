package ragas.cache

import java.util.concurrent.ConcurrentHashMap

/**
 * Simple in-memory key/value cache backend.
 */
class InMemoryCacheBackend : CacheBackend {
    private val store = ConcurrentHashMap<String, Any?>()

    /**
     * Returns the cached value for a key, or null when missing.
     */
    override fun get(key: String): Any? = store[key]

    /**
     * Stores a value in cache under the provided key.
     */
    override fun put(
        key: String,
        value: Any?,
    ) {
        store[key] = value
    }
}
