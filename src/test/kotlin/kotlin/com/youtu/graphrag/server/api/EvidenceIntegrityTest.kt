package com.youtu.graphrag.server.api

import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.graph.DatasetArtifacts
import shared.chunking.countTokensWithJTokKit
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption.ATOMIC_MOVE
import java.nio.file.StandardCopyOption.REPLACE_EXISTING
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class EvidenceIntegrityTest {
    @Test
    fun `failed graph replacement restores the previous chunk file`() {
        val root = Files.createTempDirectory("youtu-artifact-recovery")
        try {
            val graph = Files.createDirectory(root.resolve("graph.json"))
            Files.writeString(graph.resolve("marker"), "old graph")
            val chunks = root.resolve("chunks.txt")
            Files.writeString(chunks, "old chunk")

            assertFailsWith<IOException> {
                DatasetArtifacts.publish(graph, "new graph".encodeToByteArray(), chunks, "new chunk".encodeToByteArray())
            }
            assertEquals("old graph", Files.readString(graph.resolve("marker")))
            assertEquals("old chunk", Files.readString(chunks))
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `failed first graph publication leaves no chunk file`() {
        val root = Files.createTempDirectory("youtu-artifact-first-publication")
        try {
            val graph = Files.createDirectory(root.resolve("graph.json"))
            Files.writeString(graph.resolve("marker"), "existing directory")
            val chunks = root.resolve("chunks.txt")
            assertFailsWith<IOException> {
                DatasetArtifacts.publish(graph, "new graph".encodeToByteArray(), chunks, "new chunk".encodeToByteArray())
            }
            assertTrue(Files.notExists(chunks))
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `a question that exceeds the budget on its own produces no prompt`() {
        val result = buildBoundedPrompt(emptyList(), emptyList(), 5) { _, _ -> "long question ".repeat(30) }
        kotlin.test.assertNull(result)
    }

    @Test
    fun `failed rollback blocks artifact reads until replacement succeeds`() {
        val root = Files.createTempDirectory("youtu-artifact-blocked")
        try {
            val graph = root.resolve("output/graphs/sample_new.json")
            val chunks = root.resolve("output/chunks/sample.txt")
            val previousGraph = "[]".encodeToByteArray()
            DatasetArtifacts.publish(graph, previousGraph, chunks, "old chunk".encodeToByteArray())
            var moves = 0

            val failure =
                assertFailsWith<IOException> {
                    DatasetArtifacts.publish(
                        graph,
                        previousGraph,
                        chunks,
                        "uncommitted chunk".encodeToByteArray(),
                        move = { source, target ->
                            if (++moves > 1) throw IOException("injected move failure")
                            Files.move(source, target, ATOMIC_MOVE, REPLACE_EXISTING)
                        },
                    )
                }
            assertEquals(1, failure.suppressed.size)
            assertEquals("uncommitted chunk", Files.readString(chunks))
            assertFailsWith<IllegalStateException> { DatasetArtifacts.readPair(graph, chunks) }

            val configFile = root.resolve("config.yaml")
            Files.writeString(configFile, "{}")
            val service = GraphConstructionService(ConfigManager(configFile.toString()), root)
            assertFailsWith<IllegalStateException> { service.loadGraphVisualization("sample") }

            assertFailsWith<IOException> {
                DatasetArtifacts.publish(
                    graph,
                    previousGraph,
                    chunks,
                    "another uncommitted chunk".encodeToByteArray(),
                    move = { source, target ->
                        if (target == graph) throw IOException("injected graph move failure")
                        Files.move(source, target, ATOMIC_MOVE, REPLACE_EXISTING)
                    },
                )
            }
            assertEquals("uncommitted chunk", Files.readString(chunks))
            assertFailsWith<IllegalStateException> { service.loadGraphVisualization("sample") }

            DatasetArtifacts.publish(graph, previousGraph, chunks, "repaired chunk".encodeToByteArray())
            assertEquals("repaired chunk", DatasetArtifacts.readPair(graph, chunks)?.second?.decodeToString())
            assertNotNull(service.loadGraphVisualization("sample"))
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `malformed graph visualization returns empty evidence`() {
        val root = Files.createTempDirectory("youtu-invalid-graph-json")
        try {
            val configFile = root.resolve("config.yaml")
            Files.writeString(configFile, "{}")
            val graph = root.resolve("output/graphs/sample_new.json")
            Files.createDirectories(graph.parent)
            Files.writeString(graph, "invalid JSON")
            val visualization =
                assertNotNull(GraphConstructionService(ConfigManager(configFile.toString()), root).loadGraphVisualization("sample"))
            assertEquals("[]", visualization["nodes"].toString())
            assertEquals("[]", visualization["links"].toString())
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `rejecting a nonpositive prompt budget preserves the original config`() {
        val config = ConfigManager("config/base_config.json")
        val previous = config.retrieval.maxPromptTokens
        val error =
            assertFailsWith<IllegalArgumentException> {
                config.withOverrides(mapOf("retrieval" to mapOf("maxPromptTokens" to 0)))
            }
        assertTrue(error.message.orEmpty().contains("max_prompt_tokens"))
        assertEquals(previous, config.retrieval.maxPromptTokens)
    }

    @Test
    fun `bounded prompt keeps a later triple after chunks are exhausted`() {
        val result =
            assertNotNull(
                buildBoundedPrompt(listOf("oversized ".repeat(100), "small triple"), listOf("short chunk"), 20) { triples, chunks ->
                    (triples + chunks).joinToString("\n")
                },
            )
        assertEquals(listOf("small triple"), result.triples)
        assertEquals(listOf("short chunk"), result.chunks)
    }

    @Test
    fun `config overrides affect only the requesting query`() {
        val config = ConfigManager("config/base_config.json")
        val originalMode = config.triggers.mode
        val originalTopK = config.retrieval.topK
        val first = config.withOverrides(mapOf("triggers" to mapOf("mode" to "agent"), "retrieval" to mapOf("topK" to 3)))
        val second = config.withOverrides(mapOf("triggers" to mapOf("mode" to "noagent"), "retrieval" to mapOf("topK" to 17)))

        assertEquals("agent", first.triggers.mode)
        assertEquals(3, first.retrieval.topK)
        assertEquals("noagent", second.triggers.mode)
        assertEquals(17, second.retrieval.topK)
        assertEquals(originalMode, config.triggers.mode)
        assertEquals(originalTopK, config.retrieval.topK)
    }

    @Test
    fun `prompt budget includes the question and excludes oversized evidence`() {
        val prompt =
            assertNotNull(
                buildBoundedPrompt(listOf("A relates to B"), listOf("long ".repeat(100), "short evidence"), 30) { triples, chunks ->
                    "Question: which evidence supports this answer?\n" + triples.joinToString("\n") + "\n" + chunks.joinToString("\n")
                },
            )
        assertEquals(listOf("short evidence"), prompt.chunks)
        assertEquals(listOf("A relates to B"), prompt.triples)
        assertTrue(countTokensWithJTokKit(prompt.text) <= 30)
    }

    @Test
    fun `triple identity is independent of ranking score`() {
        assertEquals(tripleIdentity("(A, relates_to, B) [score: 0.100]"), tripleIdentity("(A, relates_to, B) [score: 0.900]"))
    }
}
