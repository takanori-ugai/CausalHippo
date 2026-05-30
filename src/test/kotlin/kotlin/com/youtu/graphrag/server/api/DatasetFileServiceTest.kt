package com.youtu.graphrag.server.api

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.youtu.graphrag.shared.ingest.DocumentParser
import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import java.nio.file.Path
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DatasetFileServiceTest {
    private val mapper = jacksonObjectMapper()

    @Test
    fun `upload files writes corpus and returns generated dataset name`() {
        val root = createRootDir()
        val service = DatasetFileService(rootDir = root, clock = fixedClock())
        service.ensureStartupDirectories()

        val result =
            runBlocking {
                service.uploadFiles(
                    listOf(
                        IncomingFile(
                            fileName = "My file.txt",
                            bytes = "hello world".encodeToByteArray(),
                        ),
                    ),
                )
            }

        assertTrue(result.success)
        assertEquals("My_file", result.datasetName)
        assertEquals(1, result.filesCount)

        val corpusPath = root.resolve("data/uploaded/My_file/corpus.json")
        assertTrue(corpusPath.exists())

        val corpusNode = mapper.readTree(corpusPath.toFile())
        assertEquals(1, corpusNode.size())
        assertEquals("My file.txt", corpusNode[0]["title"].asText())
        assertEquals("hello world", corpusNode[0]["text"].asText())
    }

    @Test
    fun `upload files parses document formats when parser returns text`() {
        val root = createRootDir()
        val parser =
            object : DocumentParser {
                override fun parseFile(
                    path: String,
                    extension: String,
                ): String = "parsed document body"
            }
        val service = DatasetFileService(rootDir = root, clock = fixedClock(), documentParser = parser)
        service.ensureStartupDirectories()

        val result =
            runBlocking {
                service.uploadFiles(
                    listOf(
                        IncomingFile(
                            fileName = "notes.docx",
                            bytes = "placeholder".encodeToByteArray(),
                        ),
                    ),
                )
            }

        assertTrue(result.success)
        assertEquals(1, result.filesCount)

        val corpusPath = root.resolve("data/uploaded/notes/corpus.json")
        val corpusNode = mapper.readTree(corpusPath.toFile())
        assertEquals("notes.docx", corpusNode[0]["title"].asText())
        assertEquals("parsed document body", corpusNode[0]["text"].asText())
    }

    @Test
    fun `upload files skips document formats when parser returns empty text`() {
        val root = createRootDir()
        val parser =
            object : DocumentParser {
                override fun parseFile(
                    path: String,
                    extension: String,
                ): String = ""
            }
        val service = DatasetFileService(rootDir = root, clock = fixedClock(), documentParser = parser)
        service.ensureStartupDirectories()

        val error =
            assertFailsWith<IllegalArgumentException> {
                runBlocking {
                    service.uploadFiles(
                        listOf(
                            IncomingFile(
                                fileName = "report.pdf",
                                bytes = "placeholder".encodeToByteArray(),
                            ),
                        ),
                    )
                }
            }

        assertTrue(error.message?.contains("No supported files were uploaded") == true)
        assertTrue(error.message?.contains("report.pdf") == true)
        assertTrue(!root.resolve("data/uploaded/report").exists())
    }

    @Test
    fun `get datasets includes uploaded and demo entries`() {
        val root = createRootDir()
        val service = DatasetFileService(rootDir = root, clock = fixedClock())
        service.ensureStartupDirectories()

        root.resolve("data/uploaded/alpha").createDirectories()
        Files.writeString(root.resolve("data/uploaded/alpha/corpus.json"), "[]")
        root.resolve("output/graphs").createDirectories()
        Files.writeString(root.resolve("output/graphs/alpha_new.json"), "[]")

        root.resolve("data/demo").createDirectories()
        Files.writeString(root.resolve("data/demo/demo_corpus.json"), "[]")

        val datasets = service.getDatasets().datasets
        val names = datasets.map { it.name }.toSet()

        assertTrue("alpha" in names)
        assertTrue("demo" in names)
        assertEquals("ready", datasets.first { it.name == "alpha" }.status)
    }

    @Test
    fun `save schema validates file extension and json type`() {
        val root = createRootDir()
        val service = DatasetFileService(rootDir = root, clock = fixedClock())
        service.ensureStartupDirectories()

        assertFailsWith<IllegalArgumentException> {
            service.saveSchema("demo", "schema.json", "{}".encodeToByteArray().inputStream())
        }

        assertFailsWith<IllegalArgumentException> {
            service.saveSchema("ds1", "schema.txt", "{}".encodeToByteArray().inputStream())
        }

        assertFailsWith<IllegalArgumentException> {
            service.saveSchema("ds1", "schema.json", "[]".encodeToByteArray().inputStream())
        }

        val response =
            service.saveSchema("ds1", "schema.json", "{\"Nodes\":[\"person\"]}".encodeToByteArray().inputStream())
        assertEquals(true, response["success"])
        assertTrue(root.resolve("schemas/ds1.json").exists())
    }

    @Test
    fun `delete dataset removes known files and directories`() {
        val root = createRootDir()
        val service = DatasetFileService(rootDir = root, clock = fixedClock())
        service.ensureStartupDirectories()

        root.resolve("data/uploaded/ds1").createDirectories()
        Files.writeString(root.resolve("data/uploaded/ds1/corpus.json"), "[]")
        root.resolve("output/graphs").createDirectories()
        Files.writeString(root.resolve("output/graphs/ds1_new.json"), "[]")
        root.resolve("output/chunks").createDirectories()
        Files.writeString(root.resolve("output/chunks/ds1.txt"), "chunk")
        root.resolve("schemas").createDirectories()
        Files.writeString(root.resolve("schemas/ds1.json"), "{}")
        root.resolve("retriever/faiss_cache_new/ds1").createDirectories()
        Files.writeString(root.resolve("retriever/faiss_cache_new/ds1/cache.bin"), "x")

        val response = service.deleteDataset("ds1")

        assertEquals(true, response["success"])
        assertTrue(!root.resolve("data/uploaded/ds1").exists())
        assertTrue(!root.resolve("output/graphs/ds1_new.json").exists())
        assertTrue(!root.resolve("output/chunks/ds1.txt").exists())
        assertTrue(!root.resolve("schemas/ds1.json").exists())
        assertTrue(!root.resolve("retriever/faiss_cache_new/ds1").exists())
    }

    @Test
    fun `upload fails when all files are unsupported`() {
        val root = createRootDir()
        val service = DatasetFileService(rootDir = root, clock = fixedClock())
        service.ensureStartupDirectories()

        val error =
            assertFailsWith<IllegalArgumentException> {
                runBlocking {
                    service.uploadFiles(listOf(IncomingFile("binary.exe", byteArrayOf(1, 2, 3))))
                }
            }

        assertTrue(error.message?.contains("No supported files were uploaded") == true)
        assertTrue(!root.resolve("data/uploaded/binary").exists())
    }

    @Test
    fun `upload with mixed supported and unsupported files does not persist unsupported files`() {
        val root = createRootDir()
        val service = DatasetFileService(rootDir = root, clock = fixedClock())
        service.ensureStartupDirectories()

        val result =
            runBlocking {
                service.uploadFiles(
                    listOf(
                        IncomingFile("keep.txt", "hello".encodeToByteArray()),
                        IncomingFile("drop.exe", byteArrayOf(1, 2, 3)),
                    ),
                )
            }

        assertTrue(result.success)
        assertEquals("2files_20260512", result.datasetName)
        assertEquals(1, result.filesCount)
        assertTrue(root.resolve("data/uploaded/2files_20260512/keep.txt").exists())
        assertTrue(!root.resolve("data/uploaded/2files_20260512/drop.exe").exists())
    }

    @Test
    fun `upload files skips failing document parser and continues processing`() {
        val root = createRootDir()
        val parser =
            object : DocumentParser {
                override fun parseFile(
                    path: String,
                    extension: String,
                ): String = throw IllegalStateException("parse failed")
            }
        val service = DatasetFileService(rootDir = root, clock = fixedClock(), documentParser = parser)
        service.ensureStartupDirectories()

        val progressUpdates = mutableListOf<UploadProgressUpdate>()
        val result =
            runBlocking {
                service.uploadFiles(
                    files =
                        listOf(
                            IncomingFile("keep.txt", "hello".encodeToByteArray()),
                            IncomingFile("broken.docx", byteArrayOf(1, 2, 3)),
                        ),
                    onProgress = { update -> progressUpdates.add(update) },
                )
            }

        assertTrue(result.success)
        assertEquals(1, result.filesCount)
        assertTrue(progressUpdates.any { it.progress == 50 && it.message == "Processed keep.txt" })
        assertTrue(progressUpdates.any { it.progress == 90 && it.message == "Failed to parse broken.docx" })
        assertTrue(root.resolve("data/uploaded/2files_20260512/keep.txt").exists())
        assertTrue(root.resolve("data/uploaded/2files_20260512/broken.docx").exists())
    }

    private fun createRootDir(): Path = Files.createTempDirectory("youtu-graphrag-dataset-service-test")

    private fun fixedClock(): Clock = Clock.fixed(Instant.parse("2026-05-12T00:00:00Z"), ZoneOffset.UTC)
}
