# Embeddings

## Core interface

```kotlin
interface BaseRagasEmbedding {
    suspend fun embedText(text: String): List<Float>
    suspend fun embedTexts(texts: List<String>): List<List<Float>>
}
```

## Built-in adapters

- `LangChain4jEmbedding(EmbeddingModel)`
- `CachedRagasEmbedding(delegate, cache)`

Use top-level helper `withCache(embedding, cache)` for convenience.
