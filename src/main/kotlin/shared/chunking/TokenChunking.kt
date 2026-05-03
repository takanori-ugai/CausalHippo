package shared.chunking

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.IntArrayList
import java.util.concurrent.ConcurrentHashMap

/** Default model name used for JTokKit tokenization when no model is specified. */
const val DEFAULT_TIKTOKEN_MODEL: String = "gpt-4o-mini"

/** Default token size used for ingestion chunking. */
const val DEFAULT_INGEST_CHUNK_TOKEN_SIZE: Int = 1200

/** Default overlap (in tokens) between consecutive ingestion chunks. */
const val DEFAULT_INGEST_CHUNK_OVERLAP_TOKEN_SIZE: Int = 100

/** Default maximum token budget used when building prompt context. */
const val DEFAULT_PROMPT_CONTEXT_TOKEN_BUDGET: Int = 4000
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
 * Encodes [text] into token IDs using the tokenizer associated with [model].
 *
 * @param text source text to encode.
 * @param model model name used to resolve the tokenizer; blank values use [DEFAULT_TIKTOKEN_MODEL].
 * @return token IDs in model-tokenizer order.
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
 * Decodes token [tokens] into text using the tokenizer associated with [model].
 *
 * @param tokens token IDs to decode.
 * @param model model name used to resolve the tokenizer; blank values use [DEFAULT_TIKTOKEN_MODEL].
 * @return decoded text.
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
 * Counts the number of tokens in [text] for the tokenizer associated with [model].
 *
 * @param text source text to measure.
 * @param model model name used to resolve the tokenizer; blank values use [DEFAULT_TIKTOKEN_MODEL].
 * @return number of tokens produced by encoding [text].
 */
fun countTokensWithJTokKit(
    text: String,
    model: String = DEFAULT_TIKTOKEN_MODEL,
): Int = JTokKitCodec.encoding(model).countTokens(text)

/**
 * Splits [content] into overlapping token-based chunks and preserves source order.
 *
 * Each returned [TokenChunk.content] is decoded from token slices and trimmed.
 *
 * @param content source content to chunk.
 * @param chunkTokenSize maximum tokens per chunk; must be positive.
 * @param chunkOverlapTokenSize overlap between adjacent chunks; must be non-negative and smaller than [chunkTokenSize].
 * @param model model name used to resolve the tokenizer.
 * @return ordered token chunks, or an empty list when [content] is blank.
 */
fun chunkByTokenSizeWithOverlap(
    content: String,
    chunkTokenSize: Int = DEFAULT_INGEST_CHUNK_TOKEN_SIZE,
    chunkOverlapTokenSize: Int = DEFAULT_INGEST_CHUNK_OVERLAP_TOKEN_SIZE,
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
 * Hard-truncates [text] to at most [maxTokenSize] tokens.
 *
 * @param text source text.
 * @param maxTokenSize maximum number of tokens to keep.
 * @param model model name used to resolve the tokenizer.
 * @return decoded truncated text, or an empty string when [maxTokenSize] is non-positive or [text] is blank.
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
 * Items are consumed in input order until the cumulative token count would exceed [maxTokenSize].
 * When [includePartialLastItem] is true, a truncated suffix item is included when budget remains.
 *
 * @param items ordered text items.
 * @param maxTokenSize overall token budget for all returned items.
 * @param model model name used to resolve the tokenizer.
 * @param includePartialLastItem whether to include a token-truncated version of the first overflowing item.
 * @return retained full (and optionally partial) items that fit in the budget.
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
