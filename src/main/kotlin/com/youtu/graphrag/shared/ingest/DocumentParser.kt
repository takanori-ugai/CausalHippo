package com.youtu.graphrag.shared.ingest

import io.github.oshai.kotlinlogging.KotlinLogging
import org.apache.pdfbox.Loader
import org.apache.pdfbox.text.PDFTextStripper
import org.apache.poi.hwpf.HWPFDocument
import org.apache.poi.hwpf.extractor.WordExtractor
import org.apache.poi.xwpf.extractor.XWPFWordExtractor
import org.apache.poi.xwpf.usermodel.XWPFDocument
import java.io.ByteArrayInputStream
import java.nio.file.Files
import java.nio.file.Path
import java.util.zip.ZipFile
import javax.xml.XMLConstants
import javax.xml.parsers.DocumentBuilderFactory

interface DocumentParser {
    fun parseFile(
        path: String,
        extension: String,
    ): String
}

class BestEffortDocumentParser : DocumentParser {
    private val logger = KotlinLogging.logger {}

    override fun parseFile(
        path: String,
        extension: String,
    ): String {
        val normalizedExtension = extension.lowercase()
        val filePath = Path.of(path)
        if (!Files.exists(filePath)) {
            return ""
        }

        return when (normalizedExtension) {
            ".pdf" -> parsePdf(filePath)
            ".docx" -> parseDocx(filePath)
            ".doc" -> parseDoc(filePath)
            else -> ""
        }
    }

    private fun parsePdf(path: Path): String =
        runCatching {
            Loader.loadPDF(path.toFile()).use { document ->
                PDFTextStripper().getText(document).normalizeWhitespace()
            }
        }.onFailure { error ->
            logger.warn(error) { "Failed to parse PDF file: $path" }
        }.getOrDefault("")

    private fun parseDocx(path: Path): String {
        val poiExtracted =
            runCatching { parseDocxWithPoi(path) }
                .onFailure { error -> logger.warn(error) { "POI DOCX parse failed for file: $path" } }
                .getOrDefault("")
        if (poiExtracted.isNotBlank()) {
            return poiExtracted
        }

        try {
            ZipFile(path.toFile()).use { zipFile ->
                val entry = zipFile.getEntry("word/document.xml") ?: return ""
                zipFile.getInputStream(entry).use { input ->
                    return parseDocxXml(input.readBytes())
                }
            }
        } catch (error: Exception) {
            logger.warn(error) { "Failed to parse DOCX file: $path" }
            return ""
        }
    }

    private fun parseDocxXml(xmlBytes: ByteArray): String {
        val factory =
            DocumentBuilderFactory.newInstance().apply {
                isNamespaceAware = true
                setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true)
                setFeature("http://apache.org/xml/features/disallow-doctype-decl", true)
                setFeature("http://xml.org/sax/features/external-general-entities", false)
                setFeature("http://xml.org/sax/features/external-parameter-entities", false)
                setAttribute(XMLConstants.ACCESS_EXTERNAL_DTD, "")
                setAttribute(XMLConstants.ACCESS_EXTERNAL_SCHEMA, "")
            }
        val documentBuilder = factory.newDocumentBuilder()
        val document =
            ByteArrayInputStream(xmlBytes).use { input ->
                documentBuilder.parse(input)
            }

        val textNodes = document.getElementsByTagNameNS("*", "t")
        val extracted = mutableListOf<String>()
        for (index in 0 until textNodes.length) {
            val text =
                textNodes
                    .item(index)
                    ?.textContent
                    ?.trim()
                    .orEmpty()
            if (text.isNotEmpty()) {
                extracted.add(text)
            }
        }

        return extracted.joinToString(" ").normalizeWhitespace()
    }

    private fun parseDoc(path: Path): String =
        runCatching { parseDocWithPoi(path) }
            .onFailure { error -> logger.warn(error) { "Failed to parse DOC file: $path" } }
            .getOrDefault("")

    private fun parseDocxWithPoi(path: Path): String =
        Files.newInputStream(path).use { input ->
            XWPFDocument(input).use { document ->
                XWPFWordExtractor(document).use { extractor ->
                    extractor.text.normalizeWhitespace()
                }
            }
        }

    private fun parseDocWithPoi(path: Path): String =
        Files.newInputStream(path).use { input ->
            HWPFDocument(input).use { document ->
                WordExtractor(document).use { extractor ->
                    extractor.text.normalizeWhitespace()
                }
            }
        }

    private fun String.normalizeWhitespace(): String = trim().replace(WHITESPACE_REGEX, " ")

    companion object {
        private val WHITESPACE_REGEX = Regex("\\s+")
    }
}

class NoopDocumentParser : DocumentParser {
    override fun parseFile(
        path: String,
        extension: String,
    ): String = ""
}
