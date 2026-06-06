package com.youtu.graphrag.shared.constructor

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.llm.LlmClient
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.pathString
import kotlin.io.path.readText
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class KTBuilderTest {
    private val mapper = ObjectMapper().registerKotlinModule()

    @Test
    fun `buildKnowledgeGraph maps llm extraction payload into graph relationships`() {
        val root = Files.createTempDirectory("youtu-graphrag-ktbuilder-extraction-test")
        val schemaPath = root.resolve("schemas/demo.json").also { it.parent.createDirectories() }
        val corpusPath = root.resolve("data/sample_corpus.json").also { it.parent.createDirectories() }

        schemaPath.writeText(
            """
            {
              "Nodes": ["person", "organization"],
              "Relations": ["works_at"],
              "Attributes": ["profession"]
            }
            """.trimIndent(),
        )
        mapper.writerWithDefaultPrettyPrinter().writeValue(
            corpusPath.toFile(),
            listOf(
                mapOf(
                    "title" to "Doc Alpha",
                    "text" to "Alice works at Acme.\nAlice leads platform\tengineering.",
                ),
            ),
        )

        val config = createTestConfig(root = root, mode = "noagent")
        var observedPrompt = ""
        val llmClient =
            object : LlmClient {
                override fun complete(prompt: String): String {
                    observedPrompt = prompt
                    return """
                        {
                          "attributes": {
                            "Alice": ["profession: engineer"]
                          },
                          "triples": [
                            ["Alice", "works_at", "Acme"]
                          ],
                          "entity_types": {
                            "Alice": "person",
                            "Acme": "organization"
                          }
                        }
                        """.trimIndent()
                }
            }

        val builder =
            KTBuilder(
                datasetName = "sample_ds",
                schemaPath = schemaPath.pathString,
                mode = "noagent",
                config = config,
                rootDir = root,
                llmClient = llmClient,
            )

        val relationships = builder.buildKnowledgeGraph(corpusPath.pathString)

        assertTrue(
            relationships.any { relationship ->
                relationship.relation == "has_attribute" &&
                    relationship.startNode.properties["name"] == "Alice" &&
                    relationship.endNode.properties["name"] == "profession: engineer"
            },
        )
        assertTrue(
            relationships.any { relationship ->
                relationship.relation == "works_at" &&
                    relationship.startNode.properties["name"] == "Alice" &&
                    relationship.endNode.properties["name"] == "Acme"
            },
        )
        assertTrue(
            relationships.any { relationship ->
                relationship.startNode.properties["name"] == "Alice" &&
                    relationship.startNode.properties["schema_type"] == "person"
            },
        )

        val chunksPath = root.resolve("output/chunks/sample_ds.txt")
        assertTrue(chunksPath.exists())
        val chunksText = chunksPath.readText()
        assertTrue(chunksText.contains("\\n"))
        assertTrue(chunksText.contains("\\t"))

        assertTrue(observedPrompt.contains("\"Nodes\""))
        assertTrue(observedPrompt.contains("Alice works at Acme"))
    }

    @Test
    fun `agent mode applies new_schema_types to schema file`() {
        val root = Files.createTempDirectory("youtu-graphrag-ktbuilder-schema-evolution-test")
        val schemaPath = root.resolve("schemas/demo.json").also { it.parent.createDirectories() }
        val corpusPath = root.resolve("data/agent_corpus.json").also { it.parent.createDirectories() }

        schemaPath.writeText(
            """
            {
              "Nodes": ["person"],
              "Relations": ["related_to"],
              "Attributes": ["name"]
            }
            """.trimIndent(),
        )
        mapper.writerWithDefaultPrettyPrinter().writeValue(
            corpusPath.toFile(),
            listOf(
                mapOf(
                    "title" to "Doc Beta",
                    "text" to "Alice discovered Relic X in Site Y.",
                ),
            ),
        )

        val config = createTestConfig(root = root, mode = "agent")
        val llmClient =
            object : LlmClient {
                override fun complete(prompt: String): String =
                    """
                    {
                      "attributes": {
                        "Relic X": ["material: obsidian"]
                      },
                      "triples": [
                        ["Alice", "discovered_in", "Site Y"]
                      ],
                      "entity_types": {
                        "Alice": "person",
                        "Relic X": "artifact",
                        "Site Y": "location"
                      },
                      "new_schema_types": {
                        "nodes": ["artifact", "location"],
                        "relations": ["discovered_in"],
                        "attributes": ["material"]
                      }
                    }
                    """.trimIndent()
            }

        val builder =
            KTBuilder(
                datasetName = "agent_ds",
                schemaPath = schemaPath.pathString,
                mode = "agent",
                config = config,
                rootDir = root,
                llmClient = llmClient,
            )

        val relationships = builder.buildKnowledgeGraph(corpusPath.pathString)
        assertTrue(relationships.any { relationship -> relationship.relation == "discovered_in" })

        val evolvedSchema = mapper.readTree(schemaPath.toFile())
        val nodes = evolvedSchema.path("Nodes").map { value -> value.asText() }
        val relations = evolvedSchema.path("Relations").map { value -> value.asText() }
        val attributes = evolvedSchema.path("Attributes").map { value -> value.asText() }

        assertTrue("artifact" in nodes)
        assertTrue("location" in nodes)
        assertTrue("discovered_in" in relations)
        assertTrue("material" in attributes)
    }

    @Test
    fun `construction prompt uses novel alias for anony_chs dataset`() {
        val root = Files.createTempDirectory("youtu-graphrag-ktbuilder-prompt-alias-test")
        val schemaPath = root.resolve("schemas/demo.json").also { it.parent.createDirectories() }
        val corpusPath = root.resolve("data/novel_corpus.json").also { it.parent.createDirectories() }

        schemaPath.writeText("""{"Nodes":["person"],"Relations":[],"Attributes":[]}""")
        mapper.writerWithDefaultPrettyPrinter().writeValue(
            corpusPath.toFile(),
            listOf(
                mapOf(
                    "title" to "Novel",
                    "text" to "PERSON#1 met PERSON#2.",
                ),
            ),
        )

        val config =
            createTestConfig(
                root = root,
                mode = "noagent",
                extraOverrides =
                    mapOf(
                        "prompts" to
                            mapOf(
                                "construction" to
                                    mapOf(
                                        "general" to "GENERAL::{chunk}",
                                        "novel" to "NOVEL::{chunk}",
                                    ),
                            ),
                    ),
            )

        var observedPrompt = ""
        val llmClient =
            object : LlmClient {
                override fun complete(prompt: String): String {
                    observedPrompt = prompt
                    return "{}"
                }
            }

        val builder =
            KTBuilder(
                datasetName = "anony_chs",
                schemaPath = schemaPath.pathString,
                mode = "noagent",
                config = config,
                rootDir = root,
                llmClient = llmClient,
            )

        val relationships = builder.buildKnowledgeGraph(corpusPath.pathString)
        assertTrue(relationships.isEmpty())
        assertTrue(observedPrompt.startsWith("NOVEL::"))
    }

    @Test
    fun `construction prompt uses novel_eng_agent alias for anony_eng dataset in agent mode`() {
        val root = Files.createTempDirectory("youtu-graphrag-ktbuilder-prompt-alias-agent-test")
        val schemaPath = root.resolve("schemas/demo.json").also { it.parent.createDirectories() }
        val corpusPath = root.resolve("data/novel_eng_corpus.json").also { it.parent.createDirectories() }

        schemaPath.writeText("""{"Nodes":["person"],"Relations":[],"Attributes":[]}""")
        mapper.writerWithDefaultPrettyPrinter().writeValue(
            corpusPath.toFile(),
            listOf(
                mapOf(
                    "title" to "NovelEng",
                    "text" to "PERSON#1 met PERSON#2.",
                ),
            ),
        )

        val config =
            createTestConfig(
                root = root,
                mode = "agent",
                extraOverrides =
                    mapOf(
                        "prompts" to
                            mapOf(
                                "construction" to
                                    mapOf(
                                        "general_agent" to "GENERAL_AGENT::{chunk}",
                                        "novel_eng_agent" to "NOVEL_ENG_AGENT::{chunk}",
                                    ),
                            ),
                    ),
            )

        var observedPrompt = ""
        val llmClient =
            object : LlmClient {
                override fun complete(prompt: String): String {
                    observedPrompt = prompt
                    return "{}"
                }
            }

        val builder =
            KTBuilder(
                datasetName = "anony_eng",
                schemaPath = schemaPath.pathString,
                mode = "agent",
                config = config,
                rootDir = root,
                llmClient = llmClient,
            )

        builder.buildKnowledgeGraph(corpusPath.pathString)
        assertTrue(observedPrompt.startsWith("NOVEL_ENG_AGENT::"))
    }

    @Test
    fun `buildKnowledgeGraph emits treecomm community and keyword artifacts`() {
        val root = Files.createTempDirectory("youtu-graphrag-ktbuilder-treecomm-test")
        val schemaPath = root.resolve("schemas/demo.json").also { it.parent.createDirectories() }
        val corpusPath = root.resolve("data/treecomm_corpus.json").also { it.parent.createDirectories() }

        schemaPath.writeText(
            """{"Nodes":["person","organization","location"],"Relations":["works_at","located_in"],"Attributes":["name"]}""",
        )
        mapper.writerWithDefaultPrettyPrinter().writeValue(
            corpusPath.toFile(),
            listOf(
                mapOf(
                    "title" to "Doc Tree",
                    "text" to "Alice works at Acme in Tokyo.",
                ),
            ),
        )

        val config =
            createTestConfig(
                root = root,
                mode = "noagent",
                extraOverrides =
                    mapOf(
                        "construction" to
                            mapOf(
                                "mode" to "noagent",
                                "chunk_size" to 4096,
                                "overlap" to 128,
                                "min_tail_tokens" to 32,
                                "datasets_no_chunk" to emptyList<String>(),
                                "tree_comm" to
                                    mapOf(
                                        "enable_fast_mode" to false,
                                        "max_total_communities" to 2,
                                        "enable_summary" to true,
                                        "merge_threshold" to 0.4,
                                        "max_iterations" to 6,
                                        "summary_max_words" to 20,
                                    ),
                            ),
                    ),
            )
        assertEquals(false, config.treeComm.enableFastMode)
        assertEquals(2, config.treeComm.maxTotalCommunities)
        assertEquals(true, config.treeComm.enableSummary)
        assertEquals(0.4, config.treeComm.mergeThreshold)
        assertEquals(6, config.treeComm.maxIterations)
        assertEquals(20, config.treeComm.summaryMaxWords)
        val llmClient =
            object : LlmClient {
                override fun complete(prompt: String): String {
                    if (prompt.startsWith("Summarize this graph community")) {
                        return ""
                    }
                    return """
                        {
                          "attributes": {},
                          "triples": [
                            ["Alice", "works_at", "Acme"],
                            ["Acme", "located_in", "Tokyo"]
                          ],
                          "entity_types": {
                            "Alice": "person",
                            "Acme": "organization",
                            "Tokyo": "location"
                          }
                        }
                        """.trimIndent()
                }
            }

        val builder =
            KTBuilder(
                datasetName = "treecomm_ds",
                schemaPath = schemaPath.pathString,
                mode = "noagent",
                config = config,
                rootDir = root,
                llmClient = llmClient,
            )

        val relationships = builder.buildKnowledgeGraph(corpusPath.pathString)

        assertTrue(relationships.any { relationship -> relationship.relation == "member_of" })
        assertTrue(relationships.any { relationship -> relationship.relation == "represented_by" })
        assertTrue(relationships.any { relationship -> relationship.relation == "kw_filter_by" })
        assertTrue(relationships.any { relationship -> relationship.relation == "keyword_of" })

        val communityNode =
            relationships
                .firstOrNull { relationship -> relationship.relation == "member_of" }
                ?.endNode
        assertEquals("community", communityNode?.label)
        val description =
            communityNode
                ?.properties
                ?.get("description")
                ?.toString()
                .orEmpty()
        assertTrue(description.startsWith("Community centered on"))
        val members = communityNode?.properties?.get("members") as? List<*>
        assertTrue(members != null)
        assertTrue(members.contains("Alice"))
        assertTrue(members.contains("Acme"))

        val graphPath = root.resolve("output/graphs/treecomm_ds_new.json")
        assertTrue(graphPath.exists())
        val graphText = graphPath.readText()
        assertTrue("\"member_of\"" in graphText)
        assertTrue("\"keyword_of\"" in graphText)
    }

    private fun createTestConfig(
        root: Path,
        mode: String,
        extraOverrides: Map<String, Any?> = emptyMap(),
    ): ConfigManager {
        val config = ConfigManager("config/base_config.json")

        val overrides =
            mutableMapOf<String, Any?>(
                "construction" to
                    mapOf(
                        "mode" to mode,
                        "chunk_size" to 4096,
                        "overlap" to 128,
                        "min_tail_tokens" to 32,
                        "datasets_no_chunk" to emptyList<String>(),
                    ),
                "output" to
                    mapOf(
                        "base_dir" to root.resolve("output").toString(),
                        "graphs_dir" to root.resolve("output/graphs").toString(),
                        "chunks_dir" to root.resolve("output/chunks").toString(),
                        "logs_dir" to root.resolve("output/logs").toString(),
                    ),
                "retrieval" to
                    mapOf(
                        "cache_dir" to root.resolve("retriever/faiss_cache_new").toString(),
                    ),
            )
        overrides.putAll(extraOverrides)

        config.overrideConfig(overrides)
        assertEquals(mode, config.construction.mode)
        return config
    }
}
