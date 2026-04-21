package ragas.runtime

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlin.math.min
import kotlin.math.pow

/**
 * Runs [block] with retry and exponential backoff as configured by [runConfig].
 *
 * @param runConfig Runtime retry/concurrency configuration.
 * @param block Suspending block to execute.
 */
@Suppress("TooGenericExceptionCaught")
suspend fun <T> retryAsync(
    runConfig: RunConfig,
    block: suspend () -> T,
): T {
    var attempt = 0

    while (attempt < runConfig.maxRetries) {
        try {
            return block()
        } catch (error: CancellationException) {
            throw error
        } catch (error: Exception) {
            attempt += 1
            if (!runConfig.shouldRetry(error) || attempt >= runConfig.maxRetries) {
                throw error
            }

            val exponential = 2.0.pow((attempt - 1).toDouble()).toLong()
            val waitSeconds = min(exponential, runConfig.maxWaitSeconds)
            val jitterMillis = runConfig.random.nextLong(0, 1_000)
            delay(waitSeconds * 1_000 + jitterMillis)
        }
    }

    error("Retry loop exited unexpectedly")
}
