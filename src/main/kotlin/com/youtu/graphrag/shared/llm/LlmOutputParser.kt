package com.youtu.graphrag.shared.llm

import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule

object LlmOutputParser {
    private val mapper: ObjectMapper =
        ObjectMapper()
            .registerKotlinModule()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)

    private val fenceRegex =
        Regex(
            pattern = "^\\s*```(?:\\s*\\w+)?\\s*\\n(?<body>[\\s\\S]*?)\\n\\s*```\\s*$",
            options = setOf(RegexOption.MULTILINE),
        )
    private val zeroWidthRegex = Regex("[\\u200B-\\u200D\\uFEFF]")
    private val trailingCommaRegex = Regex(",\\s*([}\\]])")
    private val quotedKeyRegex = Regex("'([^'\\r\\n]+)'\\s*:")
    private val quotedValueAfterColonRegex = Regex(":\\s*'([^'\\r\\n]*)'")
    private val quotedValueInArrayRegex = Regex("([\\[,])\\s*'([^'\\r\\n]*)'")

    fun cleanLlmContent(text: String?): String {
        if (text == null) {
            return ""
        }

        var cleaned = text.replace("\r\n", "\n").replace("\r", "\n").trim()
        cleaned = zeroWidthRegex.replace(cleaned, "")

        val fenced = fenceRegex.matchEntire(cleaned)
        cleaned =
            if (fenced != null) {
                fenced.groups["body"]
                    ?.value
                    ?.trim()
                    .orEmpty()
            } else if (cleaned.startsWith("```") && cleaned.endsWith("```") && cleaned.length >= 6) {
                cleaned.substring(3, cleaned.length - 3).trim()
            } else {
                cleaned
            }

        if (cleaned.lowercase().startsWith("json\n")) {
            cleaned = cleaned.substringAfter('\n').trim()
        }

        return cleaned
    }

    fun parseJsonValue(rawText: String?): Any? {
        val cleaned = cleanLlmContent(rawText)
        if (cleaned.isBlank()) {
            return null
        }

        val candidates = linkedSetOf(cleaned)
        extractFirstJsonBlock(cleaned)?.let { candidates.add(it) }

        for (candidate in candidates) {
            tryParse(candidate)?.let { return it }
            tryParse(repairJson(candidate))?.let { return it }
        }

        return null
    }

    fun parseJsonObject(rawText: String?): Map<String, Any?>? {
        val parsed = parseJsonValue(rawText) ?: return null
        if (parsed !is Map<*, *>) {
            return null
        }

        return parsed.entries.associate { (key, value) ->
            key.toString() to value
        }
    }

    private fun repairJson(candidate: String): String {
        var repaired = candidate
        repaired = repaired.replace('\u201C', '"').replace('\u201D', '"')
        repaired = repaired.replace('\u2018', '\'').replace('\u2019', '\'')
        repaired = trailingCommaRegex.replace(repaired, "$1")
        repaired =
            quotedKeyRegex.replace(repaired) { match ->
                "\"${match.groupValues[1]}\":"
            }
        repaired =
            quotedValueAfterColonRegex.replace(repaired) { match ->
                ": \"${match.groupValues[1]}\""
            }
        repaired =
            quotedValueInArrayRegex.replace(repaired) { match ->
                "${match.groupValues[1]} \"${match.groupValues[2]}\""
            }
        repaired = repaired.replace(Regex("\\bNone\\b"), "null")
        repaired = repaired.replace(Regex("\\bTrue\\b"), "true")
        repaired = repaired.replace(Regex("\\bFalse\\b"), "false")
        return repaired
    }

    private fun tryParse(candidate: String): Any? =
        runCatching {
            mapper.readValue(candidate, Any::class.java)
        }.getOrNull()

    private fun extractFirstJsonBlock(text: String): String? {
        val objectStart = text.indexOf('{').takeIf { it >= 0 }
        val arrayStart = text.indexOf('[').takeIf { it >= 0 }
        val startIndex =
            when {
                objectStart == null -> arrayStart
                arrayStart == null -> objectStart
                else -> minOf(objectStart, arrayStart)
            } ?: return null

        val stack = ArrayDeque<Char>()
        var inString = false
        var stringQuote = '"'
        var escaped = false

        for (index in startIndex until text.length) {
            val char = text[index]

            if (inString) {
                if (escaped) {
                    escaped = false
                    continue
                }
                if (char == '\\') {
                    escaped = true
                    continue
                }
                if (char == stringQuote) {
                    inString = false
                }
                continue
            }

            if (char == '"' || char == '\'') {
                inString = true
                stringQuote = char
                continue
            }

            when (char) {
                '{',
                '[',
                -> {
                    stack.addLast(char)
                }

                '}' -> {
                    if (stack.isEmpty() || stack.last() != '{') {
                        return null
                    }
                    stack.removeLast()
                }

                ']' -> {
                    if (stack.isEmpty() || stack.last() != '[') {
                        return null
                    }
                    stack.removeLast()
                }
            }

            if (stack.isEmpty()) {
                return text.substring(startIndex, index + 1)
            }
        }

        return null
    }
}
