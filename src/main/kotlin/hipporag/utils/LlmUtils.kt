package hipporag.utils

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlin.math.min
import kotlin.random.Random

/**
 * Filters duplicate or malformed triples to unique 3-item lists.
 */
fun filterInvalidTriples(triples: List<List<String>>): List<List<String>> {
    val uniqueTriples = mutableSetOf<List<String>>()
    val validTriples = mutableListOf<List<String>>()

    for (triple in triples) {
        if (triple.size != 3) continue
        val validTriple = triple.toList()
        if (uniqueTriples.add(validTriple)) {
            validTriples.add(validTriple)
        }
    }

    return validTriples
}

/**
 * Decodes unicode escape sequences (e.g., `\\uXXXX`) in [content].
 */
fun safeUnicodeDecode(content: String): String {
    if (!content.contains("\\u")) return content

    val builder = StringBuilder(content.length)
    var index = 0
    while (index < content.length) {
        val char = content[index]
        if (char == '\\' && index + 5 < content.length && content[index + 1] == 'u') {
            val hex = content.substring(index + 2, index + 6)
            val codeUnit = hex.toIntOrNull(16)
            if (codeUnit != null) {
                val decoded = codeUnit.toChar()
                if (Character.isHighSurrogate(decoded) &&
                    index + 11 < content.length &&
                    content[index + 6] == '\\' &&
                    content[index + 7] == 'u'
                ) {
                    val lowHex = content.substring(index + 8, index + 12)
                    val lowUnit = lowHex.toIntOrNull(16)
                    if (lowUnit != null) {
                        val low = lowUnit.toChar()
                        if (Character.isLowSurrogate(low)) {
                            val codePoint = Character.toCodePoint(decoded, low)
                            builder.append(Character.toChars(codePoint))
                            index += 12
                            continue
                        }
                    }
                }
                builder.append(decoded)
                index += 6
                continue
            }
        }
        builder.append(char)
        index += 1
    }
    return builder.toString()
}

/**
 * Decodes unicode escape sequences (e.g., `\\uXXXX`) in [content].
 */
fun safeUnicodeDecode(content: ByteArray): String = safeUnicodeDecode(content.toString(Charsets.UTF_8))

/**
 * Attempts to repair truncated JSON by closing unbalanced brackets.
 */
fun fixBrokenGeneratedJson(jsonStr: String): String {
    data class UnclosedResult(
        val stack: List<Char>,
        val insideString: Boolean,
    )

    fun findUnclosed(input: String): UnclosedResult {
        val unclosed = mutableListOf<Char>()
        var insideString = false
        var escapeNext = false
        for (char in input) {
            if (insideString) {
                when {
                    escapeNext -> escapeNext = false
                    char == '\\' -> escapeNext = true
                    char == '"' -> insideString = false
                }
            } else {
                when (char) {
                    '"' -> {
                        insideString = true
                    }

                    '{', '[' -> {
                        unclosed.add(char)
                    }

                    '}', ']' -> {
                        if (unclosed.isNotEmpty()) {
                            val last = unclosed.last()
                            if ((char == '}' && last == '{') || (char == ']' && last == '[')) {
                                unclosed.removeAt(unclosed.lastIndex)
                            }
                        }
                    }
                }
            }
        }
        return UnclosedResult(unclosed, insideString)
    }

    val unclosedOriginal = findUnclosed(jsonStr)
    if (unclosedOriginal.stack.isEmpty()) return jsonStr

    val lastCommaIndex = findLastCommaOutsideString(jsonStr)
    val truncated = if (lastCommaIndex != -1) jsonStr.substring(0, lastCommaIndex) else jsonStr
    val unclosed = findUnclosed(truncated)
    if (unclosed.stack.isEmpty()) return truncated

    val closingMap = mapOf('{' to '}', '[' to ']')
    val builder = StringBuilder(truncated)
    if (unclosed.insideString) {
        builder.append('"')
    }
    for (openChar in unclosed.stack.asReversed()) {
        builder.append(closingMap.getValue(openChar))
    }
    return builder.toString()
}

private fun findLastCommaOutsideString(input: String): Int {
    var insideString = false
    var escapeNext = false
    var lastCommaIndex = -1
    input.forEachIndexed { index, char ->
        if (insideString) {
            when {
                escapeNext -> escapeNext = false
                char == '\\' -> escapeNext = true
                char == '"' -> insideString = false
            }
        } else {
            when (char) {
                '"' -> insideString = true
                ',' -> lastCommaIndex = index
            }
        }
    }
    return lastCommaIndex
}

/**
 * Executes [block] with exponential backoff and jitter.
 */
@Suppress("TooGenericExceptionCaught")
fun <T> retryWithBackoff(
    maxAttempts: Int,
    baseDelayMillis: Long = 250,
    maxDelayMillis: Long = 4000,
    jitterMillis: Long = 100,
    retryOn: (Exception) -> Boolean = { true },
    block: () -> T,
): T {
    require(maxAttempts >= 1) { "maxAttempts must be >= 1" }
    var attempt = 0
    while (attempt < maxAttempts) {
        try {
            return block()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Error) {
            throw e
        } catch (e: Exception) {
            if (!retryOn(e) || attempt == maxAttempts - 1) {
                throw e
            }
        }
        val exponent = 1 shl attempt.coerceAtMost(10)
        val delay = min(maxDelayMillis, baseDelayMillis * exponent.toLong())
        val jitter = if (jitterMillis > 0) Random.nextLong(0, jitterMillis) else 0
        Thread.sleep(delay + jitter)
        attempt += 1
    }
    error("retryWithBackoff: unreachable")
}

/**
 * Executes [block] with exponential backoff and jitter without blocking the thread.
 */
@Suppress("TooGenericExceptionCaught")
suspend fun <T> retryWithBackoffSuspend(
    maxAttempts: Int,
    baseDelayMillis: Long = 250,
    maxDelayMillis: Long = 4000,
    jitterMillis: Long = 100,
    retryOn: (Exception) -> Boolean = { true },
    block: suspend () -> T,
): T {
    require(maxAttempts >= 1) { "maxAttempts must be >= 1" }
    var attempt = 0
    while (attempt < maxAttempts) {
        try {
            return block()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Error) {
            throw e
        } catch (e: Exception) {
            if (!retryOn(e) || attempt == maxAttempts - 1) {
                throw e
            }
        }
        val exponent = 1 shl attempt.coerceAtMost(10)
        val delayMillis = min(maxDelayMillis, baseDelayMillis * exponent.toLong())
        val jitter = if (jitterMillis > 0) Random.nextLong(0, jitterMillis) else 0
        delay(delayMillis + jitter)
        attempt += 1
    }
    error("retryWithBackoffSuspend: unreachable")
}

/**
 * Converts `{placeholder}` style strings to `${placeholder}` templates.
 */
fun convertFormatToTemplate(
    originalString: String,
    placeholderMapping: Map<String, String>? = null,
    staticValues: Map<String, Any>? = null,
): String {
    val mapping = placeholderMapping ?: emptyMap()
    val statics = staticValues ?: emptyMap()
    return TEMPLATE_PLACEHOLDER_REGEX.replace(originalString) { matchResult ->
        val originalPlaceholder = matchResult.groupValues[1]
        when {
            statics.containsKey(originalPlaceholder) -> statics.getValue(originalPlaceholder).toString()
            else -> "\${${mapping[originalPlaceholder] ?: originalPlaceholder}}"
        }
    }
}

private val TEMPLATE_PLACEHOLDER_REGEX = Regex("\\{(\\w+)\\}")
