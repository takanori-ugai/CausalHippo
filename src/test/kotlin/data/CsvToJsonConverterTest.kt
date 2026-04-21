package data

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CsvToJsonConverterTest {
    private val json = Json { ignoreUnknownKeys = false }

    @Test
    fun convertsTestCsvToExpectedJson() {
        val tempDir = Files.createTempDirectory("csv-to-json-test").toFile()
        val csvFile = File(tempDir, "test.csv")
        val outputJsonFile = File(tempDir, "out.jsonl")
        csvFile.writeText(
            """
            Title A,Effect A,Paragraph with <e1>Entity</e1> and <e2>Cause</e2>,true
            Title B,Effect B,Second paragraph,false
            """.trimIndent(),
        )

        try {
            CsvToJsonConverter().convert(csvFile.path, outputJsonFile.path)

            val actualLines = outputJsonFile.readLines().filter { it.isNotBlank() }
            assertEquals(2, actualLines.size)

            val actual = JsonArray(actualLines.map { json.parseToJsonElement(it) })
            val expected =
                json.parseToJsonElement(
                    """
                    [
                      {
                        "id": "test_0",
                        "paragraphs": [
                          {
                            "idx": 0,
                            "title": "Title A",
                            "paragraph_text": "Paragraph with Entity and Cause",
                            "is_supporting": true
                          }
                        ],
                        "question": "What is the cause of Effect A?",
                        "answer": "Title A"
                      },
                      {
                        "id": "test_1",
                        "paragraphs": [
                          {
                            "idx": 0,
                            "title": "Title B",
                            "paragraph_text": "Second paragraph",
                            "is_supporting": false
                          }
                        ],
                        "question": "What is the cause of Effect B?",
                        "answer": "Title B"
                      }
                    ]
                    """.trimIndent(),
                )
            assertEquals(expected, actual)
        } finally {
            assertTrue(outputJsonFile.delete() || !outputJsonFile.exists())
            assertTrue(csvFile.delete() || !csvFile.exists())
            assertTrue(tempDir.delete() || !tempDir.exists())
        }
    }
}
