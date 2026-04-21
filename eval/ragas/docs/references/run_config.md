# RunConfig

```kotlin
data class RunConfig(
    val timeoutSeconds: Long = 180,
    val maxRetries: Int = 10,
    val maxWaitSeconds: Long = 60,
    val maxWorkers: Int = 16,
    val retryOn: Set<KClass<out Exception>> = setOf(Exception::class),
    val logRetries: Boolean = false,
    val seed: Int = 42,
)
```

`RunConfig` controls timeout/retry/concurrency behavior used by evaluation and executor paths.
