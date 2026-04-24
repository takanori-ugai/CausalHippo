package pathrag.utils

import shared.chunking.DEFAULT_TIKTOKEN_MODEL
import shared.chunking.countTokensWithJTokKit
import shared.chunking.decodeWithJTokKit
import shared.chunking.encodeWithJTokKit

/**
 * Minimal tokenizer wrapper backed by JTokkit encodings.
 */
object Tokenizer {
    /**
     * Convert the given text into a list of token IDs using the encoding associated with the specified model.
     *
     * The function resolves an encoding for `model` and falls back to the "cl100k_base" encoding when a model-specific
     * encoding is not available.
     *
     * @param content The text to encode into tokens.
     * @param model The model name whose encoding should be used; defaults to "gpt-4o-mini".
     * @return A list of token IDs representing the encoded `content`.
     */
    fun encode(
        content: String,
        model: String = DEFAULT_TIKTOKEN_MODEL,
    ): List<Int> = encodeWithJTokKit(content, model)

    /**
     * Decode a sequence of token IDs into text using the encoding for the specified model.
     *
     * @param tokens List of token IDs to decode.
     * @param model Model name used to select the encoding; defaults to "gpt-4o-mini".
     * @return The decoded text.
     */
    fun decode(
        tokens: List<Int>,
        model: String = DEFAULT_TIKTOKEN_MODEL,
    ): String = decodeWithJTokKit(tokens, model)

    /**
     * Count tokens for [content] using the selected [model].
     */
    fun count(
        content: String,
        model: String = DEFAULT_TIKTOKEN_MODEL,
    ): Int = countTokensWithJTokKit(content, model)
}
