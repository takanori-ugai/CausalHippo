package com.youtu.graphrag.shared.retriever.nlp

import java.util.Locale

data class QueryNlpAnalysis(
    val entities: List<String> = emptyList(),
    val keyTerms: List<String> = emptyList(),
    val keywords: List<String> = emptyList(),
) {
    fun enhancedQuestion(originalQuestion: String): String {
        val trimmed = originalQuestion.trim()
        if (trimmed.isBlank()) {
            return trimmed
        }

        val enhancedParts = mutableListOf(trimmed)
        if (entities.isNotEmpty()) {
            enhancedParts.add("Entities: ${entities.joinToString(", ")}")
        }
        if (keyTerms.isNotEmpty()) {
            enhancedParts.add("Key terms: ${keyTerms.joinToString(", ")}")
        }
        return enhancedParts.joinToString(" ")
    }

    fun normalizedKeywordSet(): Set<String> =
        keywords
            .asSequence()
            .map { keyword -> keyword.trim().lowercase(Locale.ROOT) }
            .filter { keyword -> keyword.isNotBlank() }
            .toSet()
}

interface QueryNlp {
    fun analyze(question: String): QueryNlpAnalysis
}
