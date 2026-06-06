package com.youtu.graphrag.shared.retriever.nlp

import java.util.Locale

class RegexQueryNlp(
    stopwords: Collection<String>,
) : QueryNlp {
    private val stopwordSet =
        stopwords
            .asSequence()
            .map { word -> word.trim().lowercase(Locale.ROOT) }
            .filter { word -> word.isNotBlank() }
            .toSet()

    override fun analyze(question: String): QueryNlpAnalysis {
        val normalizedQuestion = question.trim()
        if (normalizedQuestion.isBlank()) {
            return QueryNlpAnalysis()
        }

        val entities = extractEntities(normalizedQuestion)
        val keyTerms = extractKeyTerms(normalizedQuestion)
        val keywords =
            linkedSetOf<String>().apply {
                keyTerms.forEach { term ->
                    val normalized = term.lowercase(Locale.ROOT)
                    if (normalized.length > 2) {
                        add(normalized)
                    }
                }
                entities.forEach { entity ->
                    val normalized = entity.lowercase(Locale.ROOT)
                    if (normalized.length > 2) {
                        add(normalized)
                    }
                }
            }

        return QueryNlpAnalysis(
            entities = entities,
            keyTerms = keyTerms,
            keywords = keywords.toList(),
        )
    }

    private fun extractEntities(question: String): List<String> {
        val collected = linkedSetOf<String>()
        val titleCaseMatcher = TITLE_CASE_ENTITY_REGEX.findAll(question)
        val acronymMatcher = ACRONYM_ENTITY_REGEX.findAll(question)

        (titleCaseMatcher + acronymMatcher).forEach { match ->
            val candidate = match.value.trim()
            if (candidate.length < 2) {
                return@forEach
            }
            val lowered = candidate.lowercase(Locale.ROOT)
            if (lowered in IGNORED_ENTITY_WORDS || lowered in stopwordSet) {
                return@forEach
            }
            collected.add(candidate)
        }
        return collected.toList()
    }

    private fun extractKeyTerms(question: String): List<String> {
        val keyTerms = mutableListOf<String>()
        TOKEN_REGEX.findAll(question).forEach { match ->
            if (keyTerms.size >= MAX_KEY_TERMS) {
                return@forEach
            }

            val token = match.value
            val normalized = token.lowercase(Locale.ROOT)
            if (normalized.length <= 2 || normalized in stopwordSet) {
                return@forEach
            }
            if (normalized.all { character -> character.isDigit() }) {
                return@forEach
            }
            keyTerms.add(token)
        }
        return keyTerms
    }

    companion object {
        private const val MAX_KEY_TERMS = 5
        private val TOKEN_REGEX = Regex("[\\p{L}\\p{N}_']+")
        private val TITLE_CASE_ENTITY_REGEX = Regex("\\b[A-Z][a-zA-Z0-9_]*(?:\\s+[A-Z][a-zA-Z0-9_]*)*\\b")
        private val ACRONYM_ENTITY_REGEX = Regex("\\b[A-Z]{2,}(?:\\s+[A-Z]{2,})*\\b")
        private val IGNORED_ENTITY_WORDS =
            setOf(
                "who",
                "what",
                "where",
                "when",
                "why",
                "how",
                "which",
            )
    }
}
