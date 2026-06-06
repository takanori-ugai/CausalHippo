package com.youtu.graphrag.shared.ingest

import java.io.ByteArrayOutputStream
import java.nio.file.Files
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.io.path.writeBytes
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertTrue

class DocumentParserTest {
    private val parser = BestEffortDocumentParser()

    @Test
    fun `parse docx extracts text from word document xml`() {
        val tempFile = Files.createTempFile("graphrag-docx-", ".docx")
        tempFile.writeBytes(createMinimalDocx("Alpha facts", "Beta details"))

        val text = parser.parseFile(tempFile.toString(), ".docx")

        assertTrue(text.contains("Alpha facts"))
        assertTrue(text.contains("Beta details"))
    }

    @Test
    fun `parse doc returns empty for non-word binary content`() {
        val tempFile = Files.createTempFile("graphrag-doc-", ".doc")
        tempFile.writeText("Legacy report text content")

        val text = parser.parseFile(tempFile.toString(), ".doc")

        assertTrue(text.isBlank())
    }

    private fun createMinimalDocx(vararg paragraphs: String): ByteArray {
        val xmlBody =
            paragraphs.joinToString(separator = "") { paragraph ->
                "<w:p><w:r><w:t>${escapeXml(paragraph)}</w:t></w:r></w:p>"
            }
        val contentTypesXml =
            """
            <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
              <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
              <Default Extension="xml" ContentType="application/xml"/>
              <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
            </Types>
            """.trimIndent()
        val rootRelsXml =
            """
            <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
            </Relationships>
            """.trimIndent()
        val documentXml =
            """
            <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>$xmlBody</w:body>
            </w:document>
            """.trimIndent()

        val output = ByteArrayOutputStream()
        ZipOutputStream(output).use { zip ->
            zip.putNextEntry(ZipEntry("[Content_Types].xml"))
            zip.write(contentTypesXml.encodeToByteArray())
            zip.closeEntry()

            zip.putNextEntry(ZipEntry("_rels/.rels"))
            zip.write(rootRelsXml.encodeToByteArray())
            zip.closeEntry()

            zip.putNextEntry(ZipEntry("word/document.xml"))
            zip.write(documentXml.encodeToByteArray())
            zip.closeEntry()
        }

        return output.toByteArray()
    }

    private fun escapeXml(value: String): String =
        value
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;")
            .replace("'", "&apos;")
}
