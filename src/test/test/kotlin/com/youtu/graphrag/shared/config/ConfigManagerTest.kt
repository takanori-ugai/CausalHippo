package com.youtu.graphrag.shared.config

import java.nio.file.Files
import kotlin.io.path.createTempDirectory
import kotlin.io.path.readText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ConfigManagerTest {
    @Test
    fun `loads base config and key dataset contracts`() {
        val config = ConfigManager("config/base_config.json")

        assertTrue(config.datasets.containsKey("demo"))
        assertEquals("agent", config.triggers.mode)
        assertEquals(
            "output/graphs/demo_new.json",
            config.getDatasetConfig("demo").graphOutput,
        )
        assertTrue(config.prompts["construction"]?.containsKey("general") == true)
    }

    @Test
    fun `applies nested overrides with python-compatible keys`() {
        val config = ConfigManager("config/base_config.json")

        config.overrideConfig(
            mapOf(
                "triggers" to mapOf("mode" to "noagent"),
                "retrieval" to
                    mapOf(
                        "top_k" to 9,
                        "strategy" to
                            mapOf(
                                "enable_parallel" to false,
                                "enabled" to listOf("lexical_triple", "lexical_chunk"),
                                "weights" to mapOf("lexical_triple" to 1.5),
                                "timeout_ms" to 900,
                                "max_concurrency" to 2,
                            ),
                    ),
                "construction" to
                    mapOf(
                        "tree_comm" to
                            mapOf(
                                "enable_fast_mode" to false,
                                "enable_summary" to true,
                                "merge_threshold" to 0.45,
                                "max_iterations" to 6,
                            ),
                    ),
            ),
        )

        assertEquals("noagent", config.triggers.mode)
        assertEquals(9, config.retrieval.topK)
        assertEquals(false, config.treeComm.enableFastMode)
        assertEquals(true, config.treeComm.enableSummary)
        assertEquals(0.45, config.treeComm.mergeThreshold)
        assertEquals(6, config.treeComm.maxIterations)
        assertEquals(false, config.retrieval.strategy.enableParallel)
        assertEquals(listOf("lexical_triple", "lexical_chunk"), config.retrieval.strategy.enabled)
        assertEquals(1.5, config.retrieval.strategy.weights["lexical_triple"])
        assertEquals(900, config.retrieval.strategy.timeoutMs)
        assertEquals(2, config.retrieval.strategy.maxConcurrency)
    }

    @Test
    fun `formats prompts and fails on missing variable`() {
        val config = ConfigManager("config/base_config.json")

        val rendered =
            config.getPromptFormatted(
                category = "retrieval",
                promptType = "general",
                variables =
                    mapOf(
                        "question" to "What is the capital of France?",
                        "context" to "Paris is the capital city of France.",
                    ),
            )

        assertTrue(rendered.contains("What is the capital of France?"))
        assertTrue(rendered.contains("Paris is the capital city of France."))

        assertFailsWith<IllegalArgumentException> {
            config.getPromptFormatted(
                category = "retrieval",
                promptType = "general",
                variables = mapOf("question" to "Missing context"),
            )
        }
    }

    @Test
    fun `creates output directories based on config`() {
        val tempRoot = createTempDirectory("youtu-graphrag-config-test")
        val baseDir = tempRoot.resolve("output")

        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
                "output" to
                    mapOf(
                        "base_dir" to baseDir.toString(),
                        "graphs_dir" to baseDir.resolve("graphs").toString(),
                        "chunks_dir" to baseDir.resolve("chunks").toString(),
                        "logs_dir" to baseDir.resolve("logs").toString(),
                    ),
            ),
        )

        config.createOutputDirectories()

        assertTrue(Files.isDirectory(baseDir))
        assertTrue(Files.isDirectory(baseDir.resolve("graphs")))
        assertTrue(Files.isDirectory(baseDir.resolve("chunks")))
        assertTrue(Files.isDirectory(baseDir.resolve("logs")))
    }

    @Test
    fun `saveConfig persists json only and round-trips config map`() {
        val tempRoot = createTempDirectory("youtu-graphrag-config-save-json-test")
        val jsonPath = tempRoot.resolve("saved_config.json")

        val config = ConfigManager("config/base_config.json")
        config.saveConfig(jsonPath.toString())
        val savedText = jsonPath.readText()
        assertTrue(savedText.contains("\"datasets\""))

        val reloaded = ConfigManager(jsonPath.toString())
        assertEquals(config.toMap(), reloaded.toMap())

        assertFailsWith<IllegalArgumentException> {
            config.saveConfig(tempRoot.resolve("saved_config.yaml").toString())
        }
    }

    @Test
    fun `rejects invalid retrieval strategy config values`() {
        val config = ConfigManager("config/base_config.json")

        assertFailsWith<IllegalArgumentException> {
            config.overrideConfig(
                mapOf(
                    "retrieval" to
                        mapOf(
                            "strategy" to
                                mapOf(
                                    "timeout_ms" to 0,
                                ),
                        ),
                ),
            )
        }
        assertFailsWith<IllegalArgumentException> {
            config.overrideConfig(
                mapOf(
                    "retrieval" to
                        mapOf(
                            "strategy" to
                                mapOf(
                                    "enabled" to emptyList<String>(),
                                ),
                        ),
                ),
            )
        }
        assertFailsWith<IllegalArgumentException> {
            config.overrideConfig(
                mapOf(
                    "retrieval" to
                        mapOf(
                            "strategy" to
                                mapOf(
                                    "enabled" to listOf("lexical-triple"),
                                ),
                        ),
                ),
            )
        }
        assertFailsWith<IllegalArgumentException> {
            config.overrideConfig(
                mapOf(
                    "retrieval" to
                        mapOf(
                            "strategy" to
                                mapOf(
                                    "weights" to mapOf("lexical-triple" to 1.0),
                                ),
                        ),
                ),
            )
        }
    }
}
