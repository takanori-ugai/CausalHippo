# Synthesizers

The Kotlin port currently provides scaffold-level synthesis via `TestsetGenerator`.

```kotlin
class TestsetGenerator(var knowledgeGraph: KnowledgeGraph = KnowledgeGraph())
```

Current behavior generates simple single-hop style samples from document text.
