package com.youtu.graphrag.shared.decomposer

import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.llm.LlmClient
import java.nio.file.Files
import kotlin.io.path.createDirectories
import kotlin.io.path.pathString
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GraphQTest {
    @Test
    fun `decompose keeps single question for simple input`() {
        val config = ConfigManager("config/base_config.json")
        val graphQ = GraphQ(datasetName = "demo", config = config, llmClient = emptyLlmClient())

        val result = graphQ.decompose("Who leads Project Alpha?", "schemas/demo.json")
        val subQuestions =
            (result["sub_questions"] as? List<*>)
                ?: error("Expected 'sub_questions' list in decompose result")

        assertEquals(1, subQuestions.size)
        val first =
            (subQuestions.first() as? Map<*, *>)
                ?: error("Expected sub-question map in sub_questions list")
        assertEquals("Who leads Project Alpha?", first["sub-question"])
    }

    @Test
    fun `decompose splits conjunction question into multiple subquestions`() {
        val config = ConfigManager("config/base_config.json")
        val graphQ = GraphQ(datasetName = "demo", config = config, llmClient = emptyLlmClient())

        val result =
            graphQ.decompose(
                "Who leads Project Alpha and where is Project Alpha based?",
                "schemas/demo.json",
            )
        val subQuestions =
            (result["sub_questions"] as? List<*>)
                ?: error("Expected 'sub_questions' list in decompose result")

        assertEquals(2, subQuestions.size)
        val first =
            (subQuestions[0] as? Map<*, *>)
                ?: error("Expected first sub-question map")
        val second =
            (subQuestions[1] as? Map<*, *>)
                ?: error("Expected second sub-question map")
        assertTrue((first["sub-question"] as String).endsWith("?"))
        assertTrue((second["sub-question"] as String).endsWith("?"))
    }

    @Test
    fun `decompose uses llm json output when available`() {
        val config = ConfigManager("config/base_config.json")
        val graphQ =
            GraphQ(
                datasetName = "demo",
                config = config,
                llmClient =
                    object : LlmClient {
                        override fun complete(prompt: String): String =
                            """
                            {
                              "sub_questions": [
                                {"sub-question": "Who is the director of Ethnic Notions?"},
                                {"sub-question": "When did the director die?"}
                              ],
                              "involved_types": {
                                "nodes": ["creative_work", "person"],
                                "relations": ["directed_by"],
                                "attributes": ["date"]
                              }
                            }
                            """.trimIndent()
                    },
            )

        val result = graphQ.decompose("Complex question?", "schemas/demo.json")
        val subQuestions =
            (result["sub_questions"] as? List<*>)
                ?: error("Expected 'sub_questions' list")
        val involvedTypes =
            (result["involved_types"] as? Map<*, *>)
                ?: error("Expected 'involved_types' map")

        assertEquals(2, subQuestions.size)
        val nodes =
            (involvedTypes["nodes"] as? List<*>)
                ?: error("Expected 'nodes' list in involved_types")
        assertEquals(2, nodes.size)
    }

    @Test
    fun `decompose supports legacy llm list format`() {
        val config = ConfigManager("config/base_config.json")
        val graphQ =
            GraphQ(
                datasetName = "demo",
                config = config,
                llmClient =
                    object : LlmClient {
                        override fun complete(prompt: String): String =
                            """
                            [
                              {"sub-question": "Subquestion A?"},
                              {"sub-question": "Subquestion B?"}
                            ]
                            """.trimIndent()
                    },
            )

        val result = graphQ.decompose("Question", "schemas/demo.json")
        val subQuestions =
            (result["sub_questions"] as? List<*>)
                ?: error("Expected 'sub_questions' list")
        val involvedTypes =
            (result["involved_types"] as? Map<*, *>)
                ?: error("Expected 'involved_types' map")

        assertEquals(2, subQuestions.size)
        val nodes = (involvedTypes["nodes"] as? List<*>) ?: error("Expected 'nodes' list")
        val relations = (involvedTypes["relations"] as? List<*>) ?: error("Expected 'relations' list")
        val attributes = (involvedTypes["attributes"] as? List<*>) ?: error("Expected 'attributes' list")
        assertTrue(nodes.isEmpty())
        assertTrue(relations.isEmpty())
        assertTrue(attributes.isEmpty())
    }

    @Test
    fun `decompose uses inline fallback template when prompts are missing from config`() {
        val root = Files.createTempDirectory("graphrag-graphq-minimal-config")
        val dataDir = root.resolve("data/demo").also { it.createDirectories() }
        val schemaDir = root.resolve("schemas").also { it.createDirectories() }
        val outputDir = root.resolve("output").also { it.createDirectories() }
        val configDir = root.resolve("config").also { it.createDirectories() }

        val corpusPath = dataDir.resolve("demo_corpus.json").also { it.writeText("[]") }
        val qaPath = dataDir.resolve("demo.json").also { it.writeText("[]") }
        val schemaPath = schemaDir.resolve("demo.json").also { it.writeText("{\"Nodes\":[]}") }
        val configPath = configDir.resolve("base_config.json")
        configPath.writeText(
            """
            {
              "datasets": {
                "demo": {
                  "corpus_path": "${corpusPath.pathString}",
                  "qa_path": "${qaPath.pathString}",
                  "schema_path": "${schemaPath.pathString}",
                  "graph_output": "${outputDir.resolve("graphs/demo_new.json").pathString}"
                }
              },
              "triggers": {
                "mode": "noagent"
              },
              "construction": {
                "mode": "noagent"
              },
              "retrieval": {
                "top_k_filter": 5
              },
              "output": {
                "base_dir": "${outputDir.pathString}"
              }
            }
            """.trimIndent(),
        )

        val config = ConfigManager(configPath.pathString)
        var observedPrompt = ""
        val graphQ =
            GraphQ(
                datasetName = "demo",
                config = config,
                llmClient =
                    object : LlmClient {
                        override fun complete(prompt: String): String {
                            observedPrompt = prompt
                            return ""
                        }
                    },
            )

        graphQ.decompose("Who leads Project Alpha?", schemaPath.pathString)

        assertTrue(observedPrompt.contains("decompose", ignoreCase = true))
    }

    @Test
    fun `decompose supports python prompt alias decomposition anony_chs`() {
        val root = Files.createTempDirectory("graphrag-graphq-prompt-alias")
        val schemaPath = root.resolve("demo_schema.json")
        schemaPath.writeText("""{"Nodes":["person"],"Relations":["related_to"],"Attributes":["name"]}""")

        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
                "prompts" to
                    mapOf(
                        "decomposition" to
                            mapOf(
                                "anony_chs" to "ALIAS_CHS::{question}::{ontology}",
                                "novel" to "NOVEL::{question}::{ontology}",
                                "general" to "GENERAL::{question}::{ontology}",
                            ),
                    ),
            ),
        )

        var observedPrompt = ""
        val graphQ =
            GraphQ(
                datasetName = "anony_chs",
                config = config,
                llmClient =
                    object : LlmClient {
                        override fun complete(prompt: String): String {
                            observedPrompt = prompt
                            return ""
                        }
                    },
            )

        graphQ.decompose("PERSON#1是谁？", schemaPath.pathString)

        assertTrue(observedPrompt.startsWith("ALIAS_CHS::"))
        assertTrue(observedPrompt.contains("PERSON#1是谁？"))
        assertTrue(observedPrompt.contains("\"Nodes\""))
    }

    private fun emptyLlmClient(): LlmClient =
        object : LlmClient {
            override fun complete(prompt: String): String = ""
        }
}
