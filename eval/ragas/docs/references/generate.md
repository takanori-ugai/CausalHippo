# Generation API

Use `TestsetGenerator.generateFromDocuments(...)`.

```kotlin
suspend fun generateFromDocuments(
    documents: List<String>,
    testsetSize: Int,
    transforms: Transforms? = null,
): Testset
```

This is a scaffold implementation and not full feature parity with Python synthesizer pipelines.
