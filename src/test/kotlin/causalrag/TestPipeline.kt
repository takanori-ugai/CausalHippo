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
 * Tests CausalRAG initialization, indexing, querying, and persistence.
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
     * Verifies that CausalRAG initializes.
     */
    @Test
    fun testPipelineInit() {
        val rag = CausalRAG(configPath = configPath.toString())
        assertNotNull(rag)
    }

    @Test
    fun testPipelineInitWithCommonConfig() {
        val rag = CausalRAG(configPath = commonConfigPath.toString())
        assertNotNull(rag)
    }

    /**
     * Verifies that indexing populates the causal graph.
     */
    @Test
    fun testDocumentIndexing() {
        val rag = CausalRAG(configPath = configPath.toString())
        rag.upsert(testDocs)
        val snapshot = rag.inspectGraph()
        val metadata = snapshot["metadata"] as? Map<*, *> ?: emptyMap<Any?, Any?>()
        val nodeCount = metadata["nodeCount"] as? Int ?: 0
        assertTrue(nodeCount > 0)
    }

    /**
     * Verifies that retrieval APIs return context and causal paths.
     */
    @Test
    fun testQueryExecution() {
        val rag = CausalRAG(configPath = configPath.toString())
        rag.upsert(testDocs)
        val contextResult = rag.query("What causes coastal flooding?", QueryParam(topK = 3, onlyNeedContext = true))
        assertTrue(contextResult.context.isNotEmpty())
        val pathResult = rag.query("What causes coastal flooding?", QueryParam(maxPaths = 3, onlyNeedCausalPaths = true))
        assertTrue(pathResult.causalPaths.isNotEmpty())
    }

    /**
     * Verifies that saved pipeline artifacts can be loaded into a fresh instance.
     */
    @Test
    fun testSaveAndLoad() {
        val rag1 = CausalRAG(configPath = configPath.toString())
        rag1.upsert(testDocs)

        val saveDir = tempDir.resolve("causalrag_index")
        Files.createDirectories(saveDir)
        rag1.saveGraph(saveDir.toString())

        val rag2 = CausalRAG(configPath = configPath.toString())
        rag2.loadGraph(saveDir.toString())

        val contextResult = rag2.query("What is climate change?", QueryParam(topK = 3, onlyNeedContext = true))
        assertTrue(contextResult.context.isNotEmpty())
        val pathResult = rag2.query("What is climate change?", QueryParam(maxPaths = 3, onlyNeedCausalPaths = true))
        assertTrue(pathResult.causalPaths.isNotEmpty())
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
