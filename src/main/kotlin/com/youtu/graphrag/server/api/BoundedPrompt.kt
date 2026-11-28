package com.youtu.graphrag.server.api

import shared.chunking.countTokensWithJTokKit

internal fun tripleIdentity(value: String): String = value.trim().replace(Regex("""\s+\[score: [+-]?\d+\.\d+\]$"""), "")

internal data class BoundedPrompt(
    val text: String,
    val triples: List<String>,
    val chunks: List<String>,
)

internal fun buildBoundedPrompt(
    triples: List<String>,
    chunks: List<String>,
    maxTokens: Int,
    render: (List<String>, List<String>) -> String,
): BoundedPrompt? {
    val full = render(triples, chunks)
    if (countTokensWithJTokKit(full) <= maxTokens) return BoundedPrompt(full, triples, chunks)
    val selectedTriples = mutableListOf<String>()
    val selectedChunks = mutableListOf<String>()
    var prompt = render(selectedTriples, selectedChunks)
    if (countTokensWithJTokKit(prompt) > maxTokens) return null

    fun addWithinBudget(
        selected: MutableList<String>,
        item: String,
    ) {
        selected.add(item)
        val candidate = render(selectedTriples, selectedChunks)
        if (countTokensWithJTokKit(candidate) <= maxTokens) {
            prompt = candidate
        } else {
            selected.removeAt(selected.lastIndex)
        }
    }

    for (index in 0 until maxOf(triples.size, chunks.size)) {
        chunks.getOrNull(index)?.let { addWithinBudget(selectedChunks, it) }
        triples.getOrNull(index)?.let { addWithinBudget(selectedTriples, it) }
    }
    return BoundedPrompt(prompt, selectedTriples, selectedChunks)
}
