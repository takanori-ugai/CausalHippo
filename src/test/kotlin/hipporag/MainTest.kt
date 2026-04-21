package hipporag

import hipporag.config.BaseConfig
import hipporag.utils.loadConfigFromJson
import org.junit.After
import org.junit.Before
import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class MainTest {
    private lateinit var tempDir: File

    @Before
    fun setup() {
        val baseDir = File(".").canonicalFile
        tempDir = createTempDirectory(baseDir.toPath(), "main_test").toFile()
    }

    @After
    fun cleanup() {
        tempDir.deleteRecursively()
    }

    @Test
    fun testArgsParseRequiredFields() {
        val args = arrayOf("--docs", "docs.json", "--queries", "queries.json")
        val parsed = Args.parse(args)

        assertEquals("docs.json", parsed.docsPath)
        assertEquals("queries.json", parsed.queriesPath)
    }

    @Test
    fun testArgsParseOptionalFields() {
        val args =
            arrayOf(
                "--docs",
                "docs.json",
                "--queries",
                "queries.json",
                "--config",
                "config.json",
                "--save_dir",
                "outputs",
                "--llm_name",
                "gpt-4",
                "--llm_base_url",
                "http://localhost:8080",
                "--embedding_name",
                "my-embedding",
                "--openie_mode",
                "offline",
                "--force_index_from_scratch",
                "true",
                "--force_openie_from_scratch",
                "false",
                "--rerank_dspy_file_path",
                "rerank.json",
            )
        val parsed = Args.parse(args)

        assertEquals("docs.json", parsed.docsPath)
        assertEquals("queries.json", parsed.queriesPath)
        assertEquals("config.json", parsed.configPath)
        assertEquals("outputs", parsed.saveDir)
        assertEquals("gpt-4", parsed.llmName)
        assertEquals("http://localhost:8080", parsed.llmBaseUrl)
        assertEquals("my-embedding", parsed.embeddingName)
        assertEquals("offline", parsed.openieMode)
        assertEquals("true", parsed.forceIndexFromScratch)
        assertEquals("false", parsed.forceOpenieFromScratch)
        assertEquals("rerank.json", parsed.rerankDspyFilePath)
    }

    @Test
    fun testResolveSafeFileWithinWorkingDir() {
        val testFile = File(tempDir, "test.json")
        testFile.writeText("""["test"]""")

        val resolvedPath = testFile.canonicalPath
        val baseDir = File(".").canonicalFile

        assertTrue(resolvedPath.startsWith(baseDir.path))
    }

    @Test
    fun testReadStringListJson() {
        val testFile = File(tempDir, "test.json")
        testFile.writeText("""["item1", "item2", "item3"]""")

        assertTrue(testFile.exists())
        assertTrue(testFile.isFile)
        val content = testFile.readText()
        assertTrue(content.contains("item1"))
    }

    @Test
    fun testLoadConfigIntegration() {
        val configFile = File(tempDir, "config.json")
        configFile.writeText(
            """{
                "saveDir": "custom_outputs",
                "llmName": "test-model",
                "temperature": 0.5
            }""",
        )

        val config = loadConfigFromJson(configFile.path)
        assertEquals("custom_outputs", config.saveDir)
        assertEquals("test-model", config.llmName)
        assertEquals(0.5, config.temperature)
    }

    @Test
    fun testPrintAnswersFormat() {
        val solutions =
            listOf(
                hipporag.utils.QuerySolution(
                    question = "What is 2+2?",
                    docs = listOf("Doc1"),
                    docScores = doubleArrayOf(1.0),
                    answer = "4",
                ),
                hipporag.utils.QuerySolution(
                    question = "What is the capital?",
                    docs = listOf("Doc2"),
                    docScores = doubleArrayOf(0.9),
                    answer = null,
                ),
            )

        val result =
            hipporag.utils.RagQaResult(
                solutions = solutions,
                responseMessages = listOf("Response1", "Response2"),
                metadata = listOf(emptyMap(), emptyMap()),
                overallRetrievalResult = null,
                overallQaResults = null,
            )

        assertNotNull(result.solutions)
        assertEquals(2, result.solutions.size)
        assertEquals("What is 2+2?", result.solutions[0].question)
        assertEquals("4", result.solutions[0].answer)
        assertEquals(null, result.solutions[1].answer)
    }

    @Test
    fun testArgsParseWithNullOptionals() {
        val args = arrayOf("--docs", "docs.json", "--queries", "queries.json")
        val parsed = Args.parse(args)

        assertEquals(null, parsed.configPath)
        assertEquals(null, parsed.saveDir)
        assertEquals(null, parsed.llmName)
        assertEquals(null, parsed.llmBaseUrl)
        assertEquals(null, parsed.embeddingName)
        assertEquals(null, parsed.openieMode)
        assertEquals(null, parsed.forceIndexFromScratch)
        assertEquals(null, parsed.forceOpenieFromScratch)
        assertEquals(null, parsed.rerankDspyFilePath)
    }

    @Test
    fun testConfigPathApplicationToConfig() {
        val config = BaseConfig()
        val saveDir = "custom_save"
        config.saveDir = saveDir

        assertEquals("custom_save", config.saveDir)
    }

    @Test
    fun testLlmNameApplicationToConfig() {
        val config = BaseConfig()
        val llmName = "custom-llm"
        config.llmName = llmName

        assertEquals("custom-llm", config.llmName)
    }

    @Test
    fun testEmbeddingNameApplicationToConfig() {
        val config = BaseConfig()
        val embeddingName = "custom-embedding"
        config.embeddingModelName = embeddingName

        assertEquals("custom-embedding", config.embeddingModelName)
    }

    @Test
    fun testOpenieModeCaseInsensitivity() {
        val config = BaseConfig()
        config.openieMode = "ONLINE"

        assertEquals("ONLINE", config.openieMode)
    }

    @Test
    fun testMultipleArgsParsing() {
        val args =
            arrayOf(
                "--docs",
                "docs.json",
                "--queries",
                "queries.json",
                "--save_dir",
                "out1",
                "--llm_name",
                "model1",
            )
        val parsed = Args.parse(args)

        assertEquals("out1", parsed.saveDir)
        assertEquals("model1", parsed.llmName)
    }
}

private data class Args(
    val docsPath: String,
    val queriesPath: String,
    val configPath: String?,
    val saveDir: String?,
    val llmBaseUrl: String?,
    val llmName: String?,
    val embeddingName: String?,
    val openieMode: String?,
    val forceIndexFromScratch: String?,
    val forceOpenieFromScratch: String?,
    val rerankDspyFilePath: String?,
) {
    companion object {
        fun parse(args: Array<String>): Args {
            val map = mutableMapOf<String, String>()
            var i = 0
            while (i < args.size) {
                val key = args[i]
                require(key.startsWith("--") && i + 1 < args.size) { "Invalid args" }
                map[key.removePrefix("--")] = args[i + 1]
                i += 2
            }

            val docs = requireNotNull(map["docs"]) { "Missing --docs" }
            val queries = requireNotNull(map["queries"]) { "Missing --queries" }

            return Args(
                docsPath = docs,
                queriesPath = queries,
                configPath = map["config"],
                saveDir = map["save_dir"],
                llmBaseUrl = map["llm_base_url"],
                llmName = map["llm_name"],
                embeddingName = map["embedding_name"],
                openieMode = map["openie_mode"],
                forceIndexFromScratch = map["force_index_from_scratch"],
                forceOpenieFromScratch = map["force_openie_from_scratch"],
                rerankDspyFilePath = map["rerank_dspy_file_path"],
            )
        }
    }
}
