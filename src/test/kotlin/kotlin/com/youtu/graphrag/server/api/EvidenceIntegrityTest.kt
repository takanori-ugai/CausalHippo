package com.youtu.graphrag.server.api

import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.graph.DatasetArtifacts
import shared.chunking.countTokensWithJTokKit
import java.io.IOException
import java.nio.file.Files
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
