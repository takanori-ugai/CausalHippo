package hipporag.utils

import io.github.oshai.kotlinlogging.KLogger
import io.github.oshai.kotlinlogging.KotlinLogging

/**
 * Returns a KotlinLogging logger with the given [name].
 */
fun getLogger(name: String): KLogger = KotlinLogging.logger(name)
