package hipporag

import hipporag.config.BaseConfig
import hipporag.utils.QuerySolution
import io.mockk.mockk
import org.junit.After
import org.junit.Before
import org.junit.Ignore
import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class HippoRagTest {
    private lateinit var tempDir: File

    @Before
    fun setup() {
        tempDir = createTempDirectory("hipporag_test").toFile()
    }

    @After
    fun cleanup() {
        tempDir.deleteRecursively()
    }

    @Test
    fun testHippoRagInitializationWithDefaultConfig() {
        val config = BaseConfig(saveDir = tempDir.path)
        assertNotNull(config)
        assertEquals(tempDir.path, config.saveDir)
    }

    @Test
    fun testHippoRagInitializationWithCustomConfig() {
        val config =
            BaseConfig(
                saveDir = tempDir.path,
                llmName = "test-model",
                embeddingModelName = "test-embedding",
            )
        assertNotNull(config)
        assertEquals("test-model", config.llmName)
        assertEquals("test-embedding", config.embeddingModelName)
    }

    @Test
    fun testConfigOverrides() {
        val initialConfig = BaseConfig(saveDir = "original")
        val config = initialConfig.copy(saveDir = tempDir.path)

        assertEquals(tempDir.path, config.saveDir)
        assertEquals("original", initialConfig.saveDir)
    }

    @Test
    fun testWorkingDirectoryCreation() {
        val config = BaseConfig(saveDir = tempDir.path)
        val llmLabel = config.llmName.replace("/", "_")
        val embeddingLabel = config.embeddingModelName.replace("/", "_")
        val workingDir = File(tempDir, "${llmLabel}_$embeddingLabel")

        assertFalse(workingDir.exists())
        workingDir.mkdirs()
        assertTrue(workingDir.exists())
    }

    @Test
    fun testQuerySolutionCreation() {
        val solution =
            QuerySolution(
                question = "What is the capital of France?",
                docs = listOf("Paris is the capital of France"),
                docScores = doubleArrayOf(0.95),
            )

        assertEquals("What is the capital of France?", solution.question)
        assertEquals(1, solution.docs.size)
        assertEquals(0.95, requireNotNull(solution.docScores)[0])
    }

    @Test
    fun testQuerySolutionWithAnswer() {
        val solution =
            QuerySolution(
                question = "What is 2+2?",
                docs = listOf("2+2=4"),
                docScores = doubleArrayOf(1.0),
                answer = "4",
            )

        assertEquals("4", solution.answer)
    }

    @Test
    fun testConfigToMapSerialization() {
        val config =
            BaseConfig(
                saveDir = tempDir.path,
                llmName = "test-model",
                temperature = 0.7,
            )

        val map = config.toMap()
        assertEquals(tempDir.path, map["saveDir"])
        assertEquals("test-model", map["llmName"])
        assertEquals(0.7, map["temperature"])
    }

    @Test
    fun testEmbeddingModelConfiguration() {
        val config =
            BaseConfig(
                embeddingModelName = "test-embedding",
                embeddingBatchSize = 32,
                embeddingMaxSeqLen = 512,
            )

        assertEquals("test-embedding", config.embeddingModelName)
        assertEquals(32, config.embeddingBatchSize)
        assertEquals(512, config.embeddingMaxSeqLen)
    }

    @Test
    fun testRetrievalConfiguration() {
        val config =
            BaseConfig(
                retrievalTopK = 100,
                linkingTopK = 10,
                qaTopK = 3,
            )

        assertEquals(100, config.retrievalTopK)
        assertEquals(10, config.linkingTopK)
        assertEquals(3, config.qaTopK)
    }

    @Test
    fun testGraphConfiguration() {
        val config =
            BaseConfig(
                isDirectedGraph = true,
                skipGraph = false,
                damping = 0.7,
            )

        assertTrue(config.isDirectedGraph)
        assertFalse(config.skipGraph)
        assertEquals(0.7, config.damping)
    }

    @Test
    fun testOpenieConfiguration() {
        val config =
            BaseConfig(
                openieMode = "offline",
                forceOpenieFromScratch = true,
                saveOpenie = true,
            )

        assertEquals("offline", config.openieMode)
        assertTrue(config.forceOpenieFromScratch)
        assertTrue(config.saveOpenie)
    }

    @Test
    @Ignore("Requires full HippoRag initialization with mocked dependencies")
    fun testIndexDocuments() {
        // This would require extensive mocking of LLM, embedding model, and graph components
        // Left as placeholder for integration testing
    }

    @Test
    @Ignore("Requires full HippoRag initialization with mocked dependencies")
    fun testRetrieve() {
        // This would require extensive mocking of LLM, embedding model, and graph components
        // Left as placeholder for integration testing
    }

    @Test
    @Ignore("Requires full HippoRag initialization with mocked dependencies")
    fun testRagQa() {
        // This would require extensive mocking of LLM, embedding model, and graph components
        // Left as placeholder for integration testing
    }

    @Test
    fun testMultipleQuerySolutions() {
        val solutions =
            listOf(
                QuerySolution(
                    question = "Q1",
                    docs = listOf("D1"),
                    docScores = doubleArrayOf(0.9),
                ),
                QuerySolution(
                    question = "Q2",
                    docs = listOf("D2", "D3"),
                    docScores = doubleArrayOf(0.8, 0.7),
                ),
            )

        assertEquals(2, solutions.size)
        assertEquals("Q1", solutions[0].question)
        assertEquals("Q2", solutions[1].question)
        assertEquals(1, solutions[0].docs.size)
        assertEquals(2, solutions[1].docs.size)
    }

    @Test
    fun testConfigLlmParameters() {
        val config =
            BaseConfig(
                llmName = "gpt-4",
                llmBaseUrl = "http://localhost:8080",
                maxNewTokens = 4096,
                temperature = 0.8,
                numGenChoices = 3,
            )

        assertEquals("gpt-4", config.llmName)
        assertEquals("http://localhost:8080", config.llmBaseUrl)
        assertEquals(4096, config.maxNewTokens)
        assertEquals(0.8, config.temperature)
        assertEquals(3, config.numGenChoices)
    }

    @Test
    fun testConfigSynonymyParameters() {
        val config =
            BaseConfig(
                synonymyEdgeTopK = 100,
                synonymyEdgeSimThreshold = 0.9,
                synonymyEdgeQueryBatchSize = 500,
                synonymyEdgeKeyBatchSize = 5000,
            )

        assertEquals(100, config.synonymyEdgeTopK)
        assertEquals(0.9, config.synonymyEdgeSimThreshold)
        assertEquals(500, config.synonymyEdgeQueryBatchSize)
        assertEquals(5000, config.synonymyEdgeKeyBatchSize)
    }

    @Test
    fun testConfigPreprocessingParameters() {
        val config =
            BaseConfig(
                textPreprocessorClassName = "CustomPreprocessor",
                preprocessEncoderName = "gpt-4",
                preprocessChunkOverlapTokenSize = 256,
                preprocessChunkMaxTokenSize = 2048,
                preprocessChunkFunc = "by_sentence",
            )

        assertEquals("CustomPreprocessor", config.textPreprocessorClassName)
        assertEquals("gpt-4", config.preprocessEncoderName)
        assertEquals(256, config.preprocessChunkOverlapTokenSize)
        assertEquals(2048, config.preprocessChunkMaxTokenSize)
        assertEquals("by_sentence", config.preprocessChunkFunc)
    }

    @Test
    fun testConfigForceFlags() {
        val config =
            BaseConfig(
                forceIndexFromScratch = true,
                forceOpenieFromScratch = true,
            )

        assertTrue(config.forceIndexFromScratch)
        assertTrue(config.forceOpenieFromScratch)
    }

    @Test
    fun testQuerySolutionWithGoldData() {
        val solution =
            QuerySolution(
                question = "Test question",
                docs = listOf("Doc1", "Doc2"),
                docScores = doubleArrayOf(0.9, 0.8),
                goldDocs = mutableListOf("GoldDoc1", "GoldDoc2"),
                goldAnswers = mutableListOf("Answer1", "Answer2"),
            )

        assertEquals(2, solution.goldDocs?.size)
        assertEquals(2, solution.goldAnswers?.size)
        assertEquals("GoldDoc1", solution.goldDocs?.get(0))
        assertEquals("Answer1", solution.goldAnswers?.get(0))
    }

    @Test
    fun testConfigDatasetParameter() {
        val config = BaseConfig(dataset = "musique")
        assertEquals("musique", config.dataset)
    }

    @Test
    fun testConfigMaxQaSteps() {
        val config = BaseConfig(maxQaSteps = 3)
        assertEquals(3, config.maxQaSteps)
    }

    @Test
    fun testConfigPassageNodeWeight() {
        val config = BaseConfig(passageNodeWeight = 0.1)
        assertEquals(0.1, config.passageNodeWeight)
    }

    @Test
    fun testConfigResponseFormat() {
        val config = BaseConfig()
        val responseFormat = config.responseFormat
        assertNotNull(responseFormat)
        assertEquals("json_object", responseFormat.get("type"))
    }

    @Test
    fun testConfigMaxRetryAttempts() {
        val config = BaseConfig(maxRetryAttempts = 10)
        assertEquals(10, config.maxRetryAttempts)
    }
}
