package data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import java.io.File

class OpenAlexFullTextToMusiqueConverter {
    private val inputJson = Json { ignoreUnknownKeys = true }
    private val outputJson = Json { prettyPrint = false }

    fun convert(
        inputPath: String,
        outputPath: String,
    ) {
        val inputFile = File(inputPath)
        val outputFile = File(outputPath)
        require(inputFile.exists()) { "Input file not found: $inputPath" }

        outputFile.parentFile?.mkdirs()

        var sourceCount = 0
        var writtenCount = 0

        inputFile.bufferedReader().use { reader ->
            outputFile.bufferedWriter().use { writer ->
                reader.lineSequence().forEachIndexed { lineIndex, rawLine ->
                    val line = rawLine.trim()
                    if (line.isBlank()) return@forEachIndexed

                    sourceCount++
                    val source =
                        try {
                            inputJson.decodeFromString(OpenAlexFullTextRow.serializer(), line)
                        } catch (e: Exception) {
                            throw IllegalArgumentException("Invalid JSON at line ${lineIndex + 1}: ${e.message}", e)
                        }

                    val sourceKey = source.id.toFullTextMusiqueKey("row${lineIndex + 1}")
                    val paragraphs =
                        source.fullText
                            .toParagraphTexts()
                            .mapIndexed { paragraphIndex, paragraphText ->
                                FullTextMusiqueParagraph(
                                    idx = paragraphIndex,
                                    title = source.title.orEmpty().trim(),
                                    paragraphText = paragraphText,
                                    isSupporting = true,
                                )
                            }

                    source
                        .extractQuestions()
                        .forEachIndexed { questionIndex, question ->
                            val output =
                                FullTextMusiqueRow(
                                    id = "${sourceKey}__q$questionIndex",
                                    paragraphs = paragraphs,
                                    question = question,
                                    questionDecomposition = emptyList(),
                                    answer = "",
                                    answerAliases = emptyList(),
                                    answerable = true,
                                )
                            writer.append(outputJson.encodeToString(output))
                            writer.newLine()
                            writtenCount++
                        }
                }
            }
        }

        println("Converted $sourceCount source rows into $writtenCount MuSiQue rows.")
        println("Output written to: $outputPath")
    }
}

private fun String?.toFullTextMusiqueKey(fallback: String): String {
    if (this.isNullOrBlank()) return fallback
    return this.trimEnd('/').substringAfterLast('/').ifBlank { fallback }
}

private fun String?.toParagraphTexts(): List<String> {
    if (this.isNullOrBlank()) return listOf("")
    val normalized = this.replace("\r\n", "\n").replace('\r', '\n').trim()
    if (normalized.isBlank()) return listOf("")
    return normalized
        .split(Regex("\\n\\s*\\n+"))
        .map { it.trim() }
        .filter { it.isNotBlank() }
        .flatMap { it.chunkByMaxWords(1000) }
        .ifEmpty { listOf("") }
}

private fun String.chunkByMaxWords(maxWords: Int): List<String> {
    require(maxWords > 0) { "maxWords must be greater than 0" }
    val words = this.trim().split(Regex("\\s+")).filter { it.isNotBlank() }
    if (words.isEmpty()) return listOf("")
    if (words.size <= maxWords) return listOf(this.trim())
    return words
        .chunked(maxWords)
        .map { it.joinToString(" ") }
}

@Serializable
private data class OpenAlexFullTextRow(
    val id: String? = null,
    val title: String? = null,
    @SerialName("fullText")
    val fullText: String? = null,
    // abstract/introduction are intentionally ignored for this converter.
    val questions: List<JsonElement> = emptyList(),
)

private fun OpenAlexFullTextRow.extractQuestions(): List<String> =
    questions
        .mapNotNull { element -> element.extractQuestionText() }
        .map { it.trim() }
        .filter { it.isNotBlank() }
        .distinct()

private fun JsonElement.extractQuestionText(): String? {
    val primitive = this as? JsonPrimitive
    if (primitive != null) {
        return primitive.contentOrNull?.trim()?.takeIf { it.isNotBlank() }
    }

    val obj = this as? JsonObject ?: return null
    return (obj["question"] as? JsonPrimitive)
        ?.contentOrNull
        ?.trim()
        ?.takeIf { it.isNotBlank() }
}

@Serializable
private data class FullTextMusiqueParagraph(
    val idx: Int,
    val title: String,
    @SerialName("paragraph_text")
    val paragraphText: String,
    @SerialName("is_supporting")
    val isSupporting: Boolean,
)

@Serializable
private data class FullTextMusiqueQuestionDecomposition(
    val id: Int = 0,
    val question: String = "",
    val answer: String = "",
    @SerialName("paragraph_support_idx")
    val paragraphSupportIdx: Int = 0,
)

@Serializable
private data class FullTextMusiqueRow(
    val id: String,
    val paragraphs: List<FullTextMusiqueParagraph>,
    val question: String,
    @SerialName("question_decomposition")
    val questionDecomposition: List<FullTextMusiqueQuestionDecomposition>,
    val answer: String,
    @SerialName("answer_aliases")
    val answerAliases: List<String>,
    val answerable: Boolean,
)
