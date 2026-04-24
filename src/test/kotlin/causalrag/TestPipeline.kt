package causalrag

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Tests pipeline initialization, indexing, querying, and persistence.
 */
class TestPipeline {
    private lateinit var tempDir: Path
    private lateinit var configPath: Path
    private lateinit var commonConfigPath: Path

    private val testDocs =
        listOf(
            "Climate change causes rising sea levels.",
            "Rising sea levels leads to coastal flooding.",
            "Coastal flooding causes population displacement.",
        )

    /**
     * Creates a temporary configuration for each test run.
     */
    @BeforeTest
    fun setUp() {
        tempDir = Files.createTempDirectory("causalrag-test")
        configPath = tempDir.resolve("causalrag-test-config.json")
        commonConfigPath = tempDir.resolve("common-rag-config.json")
        writeTestConfig(configPath)
        writeCommonConfig(commonConfigPath)
    }

    /**
     * Removes temporary test files and directories.
     */
    @AfterTest
    fun tearDown() {
        tempDir.toFile().deleteRecursively()
    }

    /**
     * Verifies that the pipeline initializes its core components.
     */
    @Test
    fun testPipelineInit() {
        val pipeline = CausalRAGPipeline(configPath = configPath.toString())
        assertNotNull(pipeline)
        assertNotNull(pipeline.graphBuilder)
        assertNotNull(pipeline.vectorRetriever)
    }

    @Test
    fun testPipelineInitWithCommonConfig() {
        val pipeline = CausalRAGPipeline(configPath = commonConfigPath.toString())
        assertNotNull(pipeline)
        assertNotNull(pipeline.graphBuilder)
        assertNotNull(pipeline.vectorRetriever)
    }

    /**
     * Verifies that indexing populates the causal graph.
     */
    @Test
    fun testDocumentIndexing() {
        val pipeline = CausalRAGPipeline(configPath = configPath.toString())
        pipeline.index(testDocs)
        val graph = pipeline.graphBuilder.getGraph()
        assertTrue(graph.numberOfNodes() > 0)
    }

    /**
     * Verifies that retrieval APIs return context and causal paths.
     */
    @Test
    fun testQueryExecution() {
        val pipeline = CausalRAGPipeline(configPath = configPath.toString())
        pipeline.index(testDocs)
        val context = pipeline.retrieveContext("What causes coastal flooding?", topK = 3)
        assertTrue(context.isNotEmpty())
        val paths = pipeline.retrieveCausalPaths("What causes coastal flooding?", maxPaths = 3)
        assertTrue(paths.isNotEmpty())
    }

    /**
     * Verifies that saved pipeline artifacts can be loaded into a fresh instance.
     */
    @Test
    fun testSaveAndLoad() {
        val pipeline1 = CausalRAGPipeline(configPath = configPath.toString())
        pipeline1.index(testDocs)

        val saveDir = tempDir.resolve("causalrag_index")
        Files.createDirectories(saveDir)
        val saved = pipeline1.save(saveDir.toString())
        assertTrue(saved)

        val pipeline2 = CausalRAGPipeline(configPath = configPath.toString())
        val loaded = pipeline2.load(saveDir.toString())
        assertTrue(loaded)

        val context = pipeline2.retrieveContext("What is climate change?", topK = 3)
        assertTrue(context.isNotEmpty())
        val paths = pipeline2.retrieveCausalPaths("What is climate change?", maxPaths = 3)
        assertTrue(paths.isNotEmpty())
    }

    private fun writeTestConfig(path: Path) {
        val json =
            JsonObject(
                mapOf(
                    "modelName" to JsonPrimitive("gpt-4o-mini"),
                    "embeddingModel" to JsonPrimitive("text-embedding-3-small"),
                    "llmProvider" to JsonPrimitive("mock"),
                    "llmApiKey" to JsonPrimitive(""),
                    "llmBaseUrl" to JsonNull,
                    "embeddingApiKey" to JsonPrimitive(""),
                    "graphPath" to JsonNull,
                    "indexPath" to JsonNull,
                    "templateStyle" to JsonPrimitive("detailed"),
                ),
            )
        val content =
            Json { prettyPrint = true }.encodeToString(
                JsonElement.serializer(),
                json,
            )
        Files.writeString(path, content)
    }

    private fun writeCommonConfig(path: Path) {
        val json =
            JsonObject(
                mapOf(
                    "shared" to
                        JsonObject(
                            mapOf(
                                "llmProvider" to JsonPrimitive("mock"),
                                "modelName" to JsonPrimitive("gpt-4o-mini"),
                                "embeddingModel" to JsonPrimitive("text-embedding-3-small"),
                                "llmApiKey" to JsonPrimitive(""),
                                "embeddingApiKey" to JsonPrimitive(""),
                            ),
                        ),
                    "causalrag" to
                        JsonObject(
                            mapOf(
                                "templateStyle" to JsonPrimitive("detailed"),
                                "graphPath" to JsonNull,
                                "indexPath" to JsonNull,
                            ),
                        ),
                ),
            )
        val content =
            Json { prettyPrint = true }.encodeToString(
                JsonElement.serializer(),
                json,
            )
        Files.writeString(path, content)
    }
}
