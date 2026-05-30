package com.youtu.graphrag.cli

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.youtu.graphrag.shared.llm.LlmClient
import picocli.CommandLine
import java.nio.file.Files
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.readText
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MainCommandTest {
    @Test
    fun `loadQaItems extracts common qa shapes`() {
        val tempDir = Files.createTempDirectory("main-command-qa")
        val qaFile = tempDir.resolve("qa.json")
        qaFile.writeText(
            """
            {
              "qa_pairs": [
                {"question": "Who leads the team?", "answer": "Alice"},
                {"query": "Where is the team based?", "gold_answer": "Tokyo"}
              ],
              "items": [
                {"q": "What does the team build?", "output": "Knowledge graph system"}
              ]
            }
            """.trimIndent(),
        )

        val items = loadQaItems(qaFile.toString())

        assertEquals(3, items.size)
        assertEquals("Who leads the team?", items[0].question)
        assertEquals("Alice", items[0].referenceAnswer)
        assertEquals("Where is the team based?", items[1].question)
        assertEquals("Tokyo", items[1].referenceAnswer)
        assertEquals("What does the team build?", items[2].question)
    }

    @Test
    fun `main command executes constructor and retrieval workflows`() {
        val tempDir = Files.createTempDirectory("main-command-run")

        val dataDir = tempDir.resolve("data/demo").also { it.createDirectories() }
        val schemasDir = tempDir.resolve("schemas").also { it.createDirectories() }
        val outputBase = tempDir.resolve("output").also { it.createDirectories() }
        val outputGraphs = outputBase.resolve("graphs").also { it.createDirectories() }
        val outputChunks = outputBase.resolve("chunks").also { it.createDirectories() }
        val outputLogs = outputBase.resolve("logs").also { it.createDirectories() }
        val cacheDir = tempDir.resolve("retriever/faiss_cache_new").also { it.createDirectories() }
        val configDir = tempDir.resolve("config").also { it.createDirectories() }

        val corpusPath = dataDir.resolve("demo_corpus.json")
        val qaPath = dataDir.resolve("demo.json")
        val schemaPath = schemasDir.resolve("demo.json")
        val graphOutputPath = outputGraphs.resolve("demo_new.json")
        val chunkOutputPath = outputChunks.resolve("demo.txt")
        val qaResultPath = outputLogs.resolve("demo_qa_results.json")
        val qaSummaryPath = outputLogs.resolve("demo_qa_summary.json")

        corpusPath.writeText(
            """
            [
              {
                "title": "Project Alpha",
                "text": "Alice leads Project Alpha in Tokyo."
              }
            ]
            """.trimIndent(),
        )
        qaPath.writeText(
            """
            [
              {
                "question": "Who leads Project Alpha?",
                "answer": "Alice"
              }
            ]
            """.trimIndent(),
        )
        schemaPath.writeText(
            """
            {
              "Nodes": ["person", "organization", "location"],
              "Relations": ["leads", "located_in"],
              "Attributes": ["name"]
            }
            """.trimIndent(),
        )

        val configPath = configDir.resolve("base_config.json")
        configPath.writeText(
            buildConfigJson(
                corpusPath = corpusPath.toAbsolutePath().toString(),
                qaPath = qaPath.toAbsolutePath().toString(),
                schemaPath = schemaPath.toAbsolutePath().toString(),
                graphOutputPath = graphOutputPath.toAbsolutePath().toString(),
                cacheDir = cacheDir.toAbsolutePath().toString(),
                outputBase = outputBase.toAbsolutePath().toString(),
                outputGraphs = outputGraphs.toAbsolutePath().toString(),
                outputChunks = outputChunks.toAbsolutePath().toString(),
                outputLogs = outputLogs.toAbsolutePath().toString(),
            ),
        )

        val mockLlmClient =
            object : LlmClient {
                override fun complete(prompt: String): String =
                    when {
                        prompt.contains("expert evaluator", ignoreCase = true) -> "1"
                        prompt.contains("decompose") -> "{\"sub_questions\": [{\"sub-question\": \"Who leads Project Alpha?\"}]}"
                        else -> "Alice is the leader."
                    }
            }

        val command = MainCommand(llmClient = mockLlmClient)
        command.configPath = configPath.toString()
        command.datasets = listOf("demo")

        command.run()

        assertTrue(graphOutputPath.exists(), "Graph output should be generated")
        assertTrue(chunkOutputPath.exists(), "Chunk output should be generated")
        assertTrue(qaResultPath.exists(), "QA result output should be generated")
        assertTrue(qaSummaryPath.exists(), "QA summary output should be generated")

        val mapper = ObjectMapper().registerKotlinModule()
        val resultNode: JsonNode = mapper.readTree(qaResultPath.toFile())
        assertTrue(resultNode.isArray, "QA results should be an array")
        val firstResult = resultNode[0]
        assertEquals("Who leads Project Alpha?", firstResult["question"].asText())
        assertEquals("Alice is the leader.", firstResult["answer"].asText())
        assertEquals("1", firstResult["eval_result"].asText())
        assertEquals("llm", firstResult["eval_method"].asText())
        assertTrue(firstResult.has("reasoning_steps"), "Should have reasoning steps")

        val summaryNode: JsonNode = mapper.readTree(qaSummaryPath.toFile())
        assertEquals(1, summaryNode["total_questions"].asInt())
        assertEquals(1.0, summaryNode["accuracy"].asDouble())
    }

    @Test
    fun `main command supports config datasets and override matrix`() {
        val tempDir = Files.createTempDirectory("main-command-options")
        val configPath = createCliConfig(tempDir)
        val mockLlmClient =
            object : LlmClient {
                override fun complete(prompt: String): String = "1"
            }
        val command = CommandLine(MainCommand(llmClient = mockLlmClient))

        val exitCode =
            command.execute(
                "--config",
                configPath.toString(),
                "--datasets",
                "demo",
                "demo_extra",
                "--override",
                """{"triggers":{"constructor_trigger":false,"retrieve_trigger":false}}""",
            )

        assertEquals(0, exitCode)
    }

    @Test
    fun `main command returns non-zero exit code for invalid override json`() {
        val tempDir = Files.createTempDirectory("main-command-invalid-override")
        val configPath = createCliConfig(tempDir)
        val mockLlmClient =
            object : LlmClient {
                override fun complete(prompt: String): String = "1"
            }
        val command = CommandLine(MainCommand(llmClient = mockLlmClient))

        val exitCode =
            command.execute(
                "--config",
                configPath.toString(),
                "--override",
                """{"triggers":""",
            )

        assertTrue(exitCode != 0)
    }

    private fun createCliConfig(tempDir: java.nio.file.Path): java.nio.file.Path {
        val dataDir = tempDir.resolve("data/demo").also { it.createDirectories() }
        val schemasDir = tempDir.resolve("schemas").also { it.createDirectories() }
        val outputBase = tempDir.resolve("output").also { it.createDirectories() }
        val outputGraphs = outputBase.resolve("graphs").also { it.createDirectories() }
        val outputChunks = outputBase.resolve("chunks").also { it.createDirectories() }
        val outputLogs = outputBase.resolve("logs").also { it.createDirectories() }
        val cacheDir = tempDir.resolve("retriever/faiss_cache_new").also { it.createDirectories() }
        val configDir = tempDir.resolve("config").also { it.createDirectories() }

        val corpusPath = dataDir.resolve("demo_corpus.json")
        val qaPath = dataDir.resolve("demo.json")
        val schemaPath = schemasDir.resolve("demo.json")
        val graphOutputPath = outputGraphs.resolve("demo_new.json")

        corpusPath.writeText("[{\"title\":\"x\",\"text\":\"Alice leads Project Alpha in Tokyo.\"}]")
        qaPath.writeText("[{\"question\":\"Who leads Project Alpha?\",\"answer\":\"Alice\"}]")
        schemaPath.writeText("""{"Nodes":["person"],"Relations":["leads"],"Attributes":["name"]}""")

        val configPath = configDir.resolve("base_config.json")
        configPath.writeText(
            buildConfigJson(
                corpusPath = corpusPath.toAbsolutePath().toString(),
                qaPath = qaPath.toAbsolutePath().toString(),
                schemaPath = schemaPath.toAbsolutePath().toString(),
                graphOutputPath = graphOutputPath.toAbsolutePath().toString(),
                cacheDir = cacheDir.toAbsolutePath().toString(),
                outputBase = outputBase.toAbsolutePath().toString(),
                outputGraphs = outputGraphs.toAbsolutePath().toString(),
                outputChunks = outputChunks.toAbsolutePath().toString(),
                outputLogs = outputLogs.toAbsolutePath().toString(),
            ),
        )
        return configPath
    }

    private fun buildConfigJson(
        corpusPath: String,
        qaPath: String,
        schemaPath: String,
        graphOutputPath: String,
        cacheDir: String,
        outputBase: String,
        outputGraphs: String,
        outputChunks: String,
        outputLogs: String,
    ): String =
        """
        {
          "datasets": {
            "demo": {
              "corpus_path": "$corpusPath",
              "qa_path": "$qaPath",
              "schema_path": "$schemaPath",
              "graph_output": "$graphOutputPath"
            }
          },
          "triggers": {
            "constructor_trigger": true,
            "retrieve_trigger": true,
            "mode": "noagent"
          },
          "construction": {
            "mode": "noagent"
          },
          "retrieval": {
            "top_k_filter": 5,
            "recall_paths": 2,
            "cache_dir": "$cacheDir"
          },
          "output": {
            "base_dir": "$outputBase",
            "graphs_dir": "$outputGraphs",
            "chunks_dir": "$outputChunks",
            "logs_dir": "$outputLogs"
          }
        }
        """.trimIndent()
}
