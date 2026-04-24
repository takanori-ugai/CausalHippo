package shared.chunking

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.IntArrayList
import java.util.concurrent.ConcurrentHashMap

const val DEFAULT_TIKTOKEN_MODEL: String = "gpt-4o-mini"
private const val DEFAULT_FALLBACK_ENCODING = "cl100k_base"

/**
 * Token-based chunk output with stable chunk order.
 *
 * @property tokens number of tokens in [content]
 * @property content decoded chunk content
 * @property chunkOrderIndex zero-based chunk order
 */
data class TokenChunk(
    val tokens: Int,
    val content: String,
    val chunkOrderIndex: Int,
)

private object JTokKitCodec {
    private val registry = Encodings.newDefaultEncodingRegistry()
    private val encodingCache = ConcurrentHashMap<String, Encoding>()

    fun encoding(model: String = DEFAULT_TIKTOKEN_MODEL): Encoding {
        val resolvedModel = model.ifBlank { DEFAULT_TIKTOKEN_MODEL }
        return encodingCache.computeIfAbsent(resolvedModel) { key ->
            registry.getEncodingForModel(key).orElseGet {
                registry.getEncoding(DEFAULT_FALLBACK_ENCODING).orElseThrow {
                    IllegalStateException("Unable to resolve fallback encoding '$DEFAULT_FALLBACK_ENCODING'")
                }
            }
        }
    }
}

/**
 * Encode [text] into token IDs using JTokKit.
 */
fun encodeWithJTokKit(
    text: String,
    model: String = DEFAULT_TIKTOKEN_MODEL,
): List<Int> {
    val encoded = JTokKitCodec.encoding(model).encode(text)
    val arr = IntArray(encoded.size())
    for (i in 0 until encoded.size()) {
        arr[i] = encoded.get(i)
    }
    return arr.toList()
}

/**
 * Decode token IDs back into text using JTokKit.
 */
fun decodeWithJTokKit(
    tokens: List<Int>,
    model: String = DEFAULT_TIKTOKEN_MODEL,
): String {
    val arr = tokens.toIntArray()
    val intArrayList = IntArrayList(arr.size)
    arr.forEach { intArrayList.add(it) }
    return JTokKitCodec.encoding(model).decode(intArrayList)
}

/**
 * Count tokens in [text] for the target [model].
 */
fun countTokensWithJTokKit(
    text: String,
    model: String = DEFAULT_TIKTOKEN_MODEL,
): Int = JTokKitCodec.encoding(model).countTokens(text)

/**
 * Token-based chunking with overlap.
 */
fun chunkByTokenSizeWithOverlap(
    content: String,
    chunkTokenSize: Int = 1200,
    chunkOverlapTokenSize: Int = 100,
    model: String = DEFAULT_TIKTOKEN_MODEL,
): List<TokenChunk> {
    require(chunkTokenSize > 0) { "chunkTokenSize must be positive." }
    require(chunkOverlapTokenSize >= 0) { "chunkOverlapTokenSize must be non-negative." }
    require(chunkTokenSize > chunkOverlapTokenSize) {
        "chunkTokenSize ($chunkTokenSize) must be greater than chunkOverlapTokenSize ($chunkOverlapTokenSize)"
    }

    if (content.isBlank()) return emptyList()

    val tokens = encodeWithJTokKit(content, model)
    if (tokens.isEmpty()) return emptyList()

    val chunks = mutableListOf<TokenChunk>()
    val step = chunkTokenSize - chunkOverlapTokenSize
    var start = 0
    var index = 0
    while (start < tokens.size) {
        val end = minOf(start + chunkTokenSize, tokens.size)
        val slice = tokens.subList(start, end)
        val decoded = decodeWithJTokKit(slice, model).trim()
        chunks +=
            TokenChunk(
                tokens = slice.size,
                content = decoded,
                chunkOrderIndex = index,
            )
        index += 1
        start += step
    }
    return chunks
}

/**
 * Hard-truncate [text] to at most [maxTokenSize] tokens.
 */
fun truncateTextByTokenSize(
    text: String,
    maxTokenSize: Int,
    model: String = DEFAULT_TIKTOKEN_MODEL,
): String {
    if (maxTokenSize <= 0 || text.isBlank()) return ""
    val tokens = encodeWithJTokKit(text, model)
    if (tokens.size <= maxTokenSize) return text
    return decodeWithJTokKit(tokens.subList(0, maxTokenSize), model)
}

/**
 * Hard token-budget truncation for ordered text items.
 *
 * When [includePartialLastItem] is true, a partial final item may be included.
 */
fun hardTruncateStringsByTokenBudget(
    items: List<String>,
    maxTokenSize: Int,
    model: String = DEFAULT_TIKTOKEN_MODEL,
    includePartialLastItem: Boolean = false,
): List<String> {
    if (maxTokenSize <= 0 || items.isEmpty()) return emptyList()

    val result = mutableListOf<String>()
    var total = 0
    for (item in items) {
        val content = item
        val itemTokens = countTokensWithJTokKit(content, model)
        if (total + itemTokens <= maxTokenSize) {
            result += content
            total += itemTokens
            continue
        }

        if (includePartialLastItem) {
            val remaining = maxTokenSize - total
            if (remaining > 0) {
                val partial = truncateTextByTokenSize(content, remaining, model).trim()
                if (partial.isNotEmpty()) {
                    result += partial
                }
            }
        }
        break
    }

    return result
}
