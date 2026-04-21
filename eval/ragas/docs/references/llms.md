# LLMs

## Core interface

```kotlin
interface BaseRagasLlm {
    var runConfig: RunConfig
    suspend fun generateText(prompt: String, n: Int = 1, temperature: Double? = 0.01, stop: List<String>? = null): LlmResult
}
```

## Built-in adapters

- `LangChain4jLlm(ChatModel, runConfig)`
- `CachedRagasLlm(delegate, cache)`

## Structured output support

`StructuredOutputRagasLlm` adds:

- `generateNumericValue(...)`
- `generateDiscreteValue(...)`
- `generateRankingItems(...)`

`LangChain4jLlm` implements both text and structured-output interfaces.
