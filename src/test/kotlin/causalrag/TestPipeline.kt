package causalrag

import causalrag.generator.llm.LLMInterface
import io.mockk.every
import io.mockk.mockkConstructor
import io.mockk.unmockkConstructor
import kotlinx.coroutines.runBlocking
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
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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

    @Test
    fun testRepeatedUpsertPreservesPreviouslyIndexedPassages() {
        val rag = CausalRAG(configPath = configPath.toString())
        testDocs.forEach(rag::upsert)

        assertEquals(testDocs.toSet(), rag.retrieveContext("coastal flooding", topK = 10).toSet())
    }

    @Test
    fun vectorOnlyUpsertDoesNotSuppressSubsequentGraphExtraction() =
        runBlocking {
            val rag = CausalRAG(configPath = configPath.toString())
            val passage = testDocs.first()
            rag.upsert(mapOf("first" to mapOf("content" to passage)))
            assertEquals(0, (rag.inspectGraph()["metadata"] as Map<*, *>)["edgeCount"])

            rag.aupsert(passage)
            val indexed = rag.inspectGraph()
            assertTrue(((indexed["metadata"] as Map<*, *>)["edgeCount"] as Int) > 0)
            assertEquals(listOf(passage), rag.retrieveContext("climate change"))
            rag.aupsert(passage)
            assertEquals(indexed, rag.inspectGraph())
        }

    @Test
    fun testCorruptSnapshotPreservesLiveIndex() {
        val rag = CausalRAG(configPath = configPath.toString())
        rag.upsert(testDocs)
        val graph = rag.inspectGraph()
        val query = "coastal flooding"
        val context = rag.retrieveContext(query, topK = 5)
        val replacement = CausalRAG(configPath = configPath.toString())
        replacement.upsert("Heavy rainfall causes river flooding.")
        val snapshot = tempDir.resolve("corrupt-snapshot")
        replacement.saveGraph(snapshot.toString())
        Files.writeString(snapshot.resolve("vector_cache/vectors.json"), "[[1.0], [1.0, 0.0]]")

        assertFailsWith<IllegalStateException> { rag.loadGraph(snapshot.toString()) }
        assertEquals(graph, rag.inspectGraph())
        assertEquals(context, rag.retrieveContext(query, topK = 5))
    }

    @Test
    fun testAnswerAndContextOnlyUseTheSameEvidence() {
        mockkConstructor(LLMInterface::class)
        try {
            val prompts = mutableListOf<String>()
            every { anyConstructed<LLMInterface>().generate(any(), any(), any(), any(), any(), any()) } answers {
                prompts.add(firstArg())
                "[]"
            }
            val rag = CausalRAG(configPath = configPath.toString())
            rag.upsert(testDocs)
            val query = "What causes coastal flooding?"
            val params = QueryParam(topK = 3, maxPaths = 1)
            val allPaths = rag.retrieveCausalPaths(query, maxPaths = 3)
            assertTrue(allPaths.size > 1)
            val answer = rag.query(query, params)
            val evidence = rag.query(query, params.copy(onlyNeedContext = true, onlyNeedCausalPaths = true))

            assertEquals(answer.context, evidence.context)
            assertEquals(answer.causalPaths, evidence.causalPaths)
            assertEquals(1, answer.causalPaths.size)
            assertTrue(answer.causalPaths.single().joinToString(" → ") in prompts.last())
            for (path in allPaths - answer.causalPaths.toSet()) {
                assertTrue(path.joinToString(" → ") !in prompts.last())
            }
        } finally {
            unmockkConstructor(LLMInterface::class)
        }
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
