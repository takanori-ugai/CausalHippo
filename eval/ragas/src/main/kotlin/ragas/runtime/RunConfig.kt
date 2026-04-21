package ragas.runtime

import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * Runtime settings for retries, concurrency, and timeout behavior.
 *
 * @property timeoutSeconds Timeout in seconds.
 * @property maxRetries Maximum retries.
 * @property maxWaitSeconds Maximum backoff wait in seconds.
 * @property maxWorkers Maximum concurrent workers.
 * @property retryOn Retryable exception classes.
 * @property logRetries Whether retry attempts are logged.
 * @property seed Random seed used for deterministic jitter.
 */
data class RunConfig(
    val timeoutSeconds: Long = 180,
    val maxRetries: Int = 10,
    val maxWaitSeconds: Long = 60,
    val maxWorkers: Int = 16,
    val retryOn: Set<KClass<out Exception>> = setOf(Exception::class),
    val logRetries: Boolean = false,
    val seed: Int = 42,
) {
    val random: Random = Random(seed)

    /**
     * Returns true when the exception type matches configured retryable types.
     *
     * @param error Error to inspect or report.
     */
    fun shouldRetry(error: Exception): Boolean = retryOn.any { type -> type.isInstance(error) }
}
