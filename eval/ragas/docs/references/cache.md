# Cache

## Interface

```kotlin
interface CacheBackend {
    fun get(key: String): Any?
    fun put(key: String, value: Any?)
}
```

## Included backend

- `InMemoryCacheBackend`

## Utilities

- `stableCacheKey(raw: String): String` (SHA-256)
- `withCache(llm, cache)`
- `withCache(embedding, cache)`
