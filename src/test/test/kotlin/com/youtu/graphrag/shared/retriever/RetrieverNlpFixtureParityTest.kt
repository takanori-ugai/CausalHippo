package com.youtu.graphrag.shared.retriever

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.PropertyNamingStrategies
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.graph.GraphNode
import com.youtu.graphrag.shared.graph.GraphRelationship
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.createDirectories
import kotlin.io.path.writeText
import kotlin.math.roundToInt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

private data class RetrieverNlpParityFixture(
    val name: String,
    val provider: String,
    val question: String,
    val stopwords: List<String>,
    val pythonEntities: List<String>,
    val pythonKeywords: List<String>,
    val requiredKeywords: List<String> = emptyList(),
    val forbiddenKeywords: List<String> = emptyList(),
    val minEntityOverlap: Double = 1.0,
    val minKeywordOverlap: Double = 1.0,
)

class RetrieverNlpFixtureParityTest {
    private val mapper =
        ObjectMapper()
            .registerKotlinModule()
            .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)

    @Test
    fun `nlp fixtures match python baseline tolerances`() {
        val fixtures = loadFixtures()
        assertTrue(fixtures.isNotEmpty())

        fixtures.forEach { fixture ->
            val root = Files.createTempDirectory("youtu-graphrag-nlp-fixture-${fixture.name}")
            val config = createTestConfig(root, fixture)
            val datasetName = "nlp_fixture_${fixture.name}"
            val graphPath = root.resolve("output/graphs/${datasetName}_new.json")
            val chunkPath = root.resolve("output/chunks/$datasetName.txt")

            graphPath.parent?.createDirectories()
            mapper.writerWithDefaultPrettyPrinter().writeValue(
                graphPath.toFile(),
                listOf(
                    GraphRelationship(
                        startNode = GraphNode(label = "entity", properties = mapOf("name" to "OpenAI", "chunk id" to "c1")),
                        relation = "related_to",
                        endNode = GraphNode(label = "entity", properties = mapOf("name" to "San Francisco", "chunk id" to "c1")),
                    ),
                ),
            )
            writeChunks(
                chunkPath,
                listOf("c1" to "OpenAI was founded in 2015 and is based in San Francisco."),
            )

            val retriever =
                KTRetriever.createAndBuild(
                    datasetName = datasetName,
                    graphPath = graphPath.toString(),
                    recallPaths = 1,
                    schemaPath = "unused_schema.json",
                    topK = 2,
                    mode = "agent",
                    config = config,
                    rootDir = root,
                )

            val entities =
                retriever
                    .extractQueryEntities(fixture.question)
                    .map { value -> value.toString().trim().lowercase() }
                    .filter { value -> value.isNotBlank() }
                    .toSet()
            val keywords =
                retriever
                    .extractQueryKeywords(fixture.question)
                    .map { value -> value.trim().lowercase() }
                    .filter { value -> value.isNotBlank() }
                    .toSet()

            val pythonEntityBaseline = fixture.pythonEntities.map { it.trim().lowercase() }.toSet()
            val pythonKeywordBaseline = fixture.pythonKeywords.map { it.trim().lowercase() }.toSet()
            val entityOverlap = overlapRatio(expected = pythonEntityBaseline, actual = entities)
            val keywordOverlap = overlapRatio(expected = pythonKeywordBaseline, actual = keywords)

            assertTrue(
                entityOverlap >= fixture.minEntityOverlap,
                "Entity overlap for fixture '${fixture.name}' was ${percent(
                    entityOverlap,
                )} but required ${percent(fixture.minEntityOverlap)}",
            )
            assertTrue(
                keywordOverlap >= fixture.minKeywordOverlap,
                "Keyword overlap for fixture '${fixture.name}' was ${percent(
                    keywordOverlap,
                )} but required ${percent(fixture.minKeywordOverlap)}",
            )

            fixture.requiredKeywords.map { it.lowercase() }.forEach { required ->
                assertTrue(required in keywords, "Required keyword '$required' missing for fixture '${fixture.name}'")
            }
            fixture.forbiddenKeywords.map { it.lowercase() }.forEach { forbidden ->
                assertTrue(forbidden !in keywords, "Forbidden keyword '$forbidden' present for fixture '${fixture.name}'")
            }

            val enhanced = retriever.enhanceQuery(fixture.question)
            assertTrue(enhanced.startsWith(fixture.question))
            assertTrue("Key terms:" in enhanced || "Entities:" in enhanced)
            assertEquals(
                fixture.provider.lowercase(),
                config.nlp.provider.lowercase(),
                "fixture provider wiring mismatch for '${fixture.name}'",
            )
        }
    }

    private fun loadFixtures(): List<RetrieverNlpParityFixture> {
        val resourcePath = "fixtures/nlp/retrieval_nlp_parity_fixtures.json"
        val stream =
            requireNotNull(javaClass.classLoader.getResourceAsStream(resourcePath)) {
                "Fixture resource not found: $resourcePath"
            }
        return stream.use { input ->
            mapper.readValue(input, object : TypeReference<List<RetrieverNlpParityFixture>>() {})
        }
    }

    private fun createTestConfig(
        root: Path,
        fixture: RetrieverNlpParityFixture,
    ): ConfigManager {
        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
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
                        "enable_caching" to false,
                    ),
                "nlp" to
                    mapOf(
                        "provider" to fixture.provider,
                        "stopwords" to fixture.stopwords,
                        "opennlp" to
                            mapOf(
                                "tokenizer_model_path" to "",
                                "pos_model_path" to "",
                                "person_model_path" to "",
                                "organization_model_path" to "",
                                "location_model_path" to "",
                            ),
                    ),
            ),
        )
        return config
    }

    private fun writeChunks(
        chunkPath: Path,
        entries: List<Pair<String, String>>,
    ) {
        chunkPath.parent?.createDirectories()
        val content =
            buildString {
                entries.forEach { (chunkId, text) ->
                    append("id: ")
                    append(chunkId)
                    append('\t')
                    append("Chunk: ")
                    append(text)
                    append('\n')
                }
            }
        chunkPath.writeText(content)
    }

    private fun overlapRatio(
        expected: Set<String>,
        actual: Set<String>,
    ): Double {
        if (expected.isEmpty()) {
            return 1.0
        }
        val matched = expected.count { term -> term in actual }
        return matched.toDouble() / expected.size.toDouble()
    }

    private fun percent(value: Double): String = "${(value * 100.0).roundToInt()}%"
}
