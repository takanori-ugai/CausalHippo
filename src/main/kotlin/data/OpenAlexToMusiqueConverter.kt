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

class OpenAlexToMusiqueConverter {
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
                            inputJson.decodeFromString(OpenAlexRow.serializer(), line)
                        } catch (e: Exception) {
                            throw IllegalArgumentException("Invalid JSON at line ${lineIndex + 1}: ${e.message}", e)
                        }

                    val sourceKey = source.id.toMusiqueKey("row${lineIndex + 1}")
                    val paragraph =
                        MusiqueParagraph(
                            idx = 0,
                            title = source.title.orEmpty().trim(),
                            paragraphText = source.abstract.orEmpty().trim(),
                            isSupporting = true,
                        )

                    source
                        .extractQuestions(targetSource = "abstract")
                        .forEachIndexed { questionIndex, question ->
                            val output =
                                MusiqueRow(
                                    id = "${sourceKey}__q$questionIndex",
                                    paragraphs = listOf(paragraph),
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

private fun String?.toMusiqueKey(fallback: String): String {
    if (this.isNullOrBlank()) return fallback
    return this.trimEnd('/').substringAfterLast('/').ifBlank { fallback }
}

@Serializable
private data class OpenAlexRow(
    val id: String? = null,
    val title: String? = null,
    val abstract: String? = null,
    // Keep only fields we need; introduction/fullText are intentionally ignored.
    val questions: List<JsonElement> = emptyList(),
)

private fun OpenAlexRow.extractQuestions(targetSource: String): List<String> =
    questions
        .mapNotNull { element -> element.extractQuestionAndSource() }
        .filter { (_, source) -> source == null || source == targetSource }
        .map { (question, _) -> question.trim() }
        .filter { it.isNotBlank() }
        .distinct()

private fun JsonElement.extractQuestionAndSource(): Pair<String, String?>? {
    val primitive = this as? JsonPrimitive
    if (primitive != null) {
        val value = primitive.contentOrNull?.trim().orEmpty()
        return value.takeIf { it.isNotBlank() }?.let { it to null }
    }

    val obj = this as? JsonObject ?: return null
    val question =
        (obj["question"] as? JsonPrimitive)
            ?.contentOrNull
            ?.trim()
            .orEmpty()
    if (question.isBlank()) return null
    val source =
        (obj["source"] as? JsonPrimitive)
            ?.contentOrNull
            ?.trim()
            ?.lowercase()
    return question to source
}

@Serializable
private data class MusiqueParagraph(
    val idx: Int,
    val title: String,
    @SerialName("paragraph_text")
    val paragraphText: String,
    @SerialName("is_supporting")
    val isSupporting: Boolean,
)

@Serializable
private data class MusiqueQuestionDecomposition(
    val id: Int = 0,
    val question: String = "",
    val answer: String = "",
    @SerialName("paragraph_support_idx")
    val paragraphSupportIdx: Int = 0,
)

@Serializable
private data class MusiqueRow(
    val id: String,
    val paragraphs: List<MusiqueParagraph>,
    val question: String,
    @SerialName("question_decomposition")
    val questionDecomposition: List<MusiqueQuestionDecomposition>,
    val answer: String,
    @SerialName("answer_aliases")
    val answerAliases: List<String>,
    val answerable: Boolean,
)
