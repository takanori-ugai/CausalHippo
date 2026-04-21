package data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.apache.commons.csv.CSVFormat
import java.io.File

class CsvToJsonConverter {
    private val json = Json { prettyPrint = false }

    fun convert(
        csvPath: String,
        outputPath: String,
    ) {
        val inputFile = File(csvPath)
        val outputFile = File(outputPath)

        outputFile.parentFile?.mkdirs()
        inputFile.bufferedReader().use { reader ->
            outputFile.bufferedWriter().use { writer ->
                CSVFormat.DEFAULT.parse(reader).use { records ->
                    var rowIndex = 0
                    var hasRows = false

                    for (fields in records) {
                        if ((0 until fields.size()).none { i -> fields[i].isNotBlank() }) {
                            continue
                        }

                        hasRows = true
                        require(fields.size() >= 4) {
                            "Row $rowIndex expected at least 4 CSV fields (col1,col2,paragraph,is_supporting), got ${fields.size()}"
                        }

                        val column1 = fields[0].trim()
                        val column2 = fields[1].trim()
                        val paragraph = stripEntityTags(fields[2].trim())
                        val isSupporting = parseBoolean(fields[3])

                        val document =
                            Document(
                                id = "${inputFile.nameWithoutExtension}_$rowIndex",
                                paragraphs =
                                    listOf(
                                        Paragraph(idx = 0, title = column1, paragraphText = paragraph, isSupporting = isSupporting),
                                    ),
                                question = "What is the cause of $column2?",
                                answer = column1,
                            )
                        writer.append(json.encodeToString(document))
                        writer.newLine()
                        rowIndex++
                    }

                    require(hasRows) { "CSV file is empty: $csvPath" }
                }
            }
        }
    }

    private fun stripEntityTags(text: String): String = text.replace(Regex("</?e[12]>"), "")

    private fun parseBoolean(raw: String): Boolean = raw.trim().equals("true", ignoreCase = true)

    @Serializable
    data class Paragraph(
        val idx: Int,
        val title: String,
        @SerialName("paragraph_text") val paragraphText: String,
        @SerialName("is_supporting") val isSupporting: Boolean,
    )

    @Serializable
    data class Document(
        val id: String,
        val paragraphs: List<Paragraph>,
        val question: String,
        val answer: String,
    )
}
