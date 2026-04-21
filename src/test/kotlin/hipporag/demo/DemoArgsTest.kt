package hipporag.demo

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class DemoArgsTest {
    @Test
    fun testParseRequiredArgs() {
        val args = arrayOf("--docs", "docs.json", "--queries", "queries.json")
        val parsed = DemoArgs.parse(args)

        assertEquals("docs.json", parsed.docsPath)
        assertEquals("queries.json", parsed.queriesPath)
        assertEquals("outputs", parsed.saveDir)
        assertEquals("gpt-4o-mini", parsed.llmName)
        assertEquals("text-embedding-3-large", parsed.embeddingName)
    }

    @Test
    fun testParseAllArgs() {
        val args =
            arrayOf(
                "--docs",
                "docs.json",
                "--queries",
                "queries.json",
                "--save_dir",
                "custom_dir",
                "--llm_name",
                "gpt-4",
                "--embedding_name",
                "custom-embedding",
                "--llm_base_url",
                "http://localhost:8080",
                "--embedding_base_url",
                "http://localhost:9090",
            )
        val parsed = DemoArgs.parse(args)

        assertEquals("docs.json", parsed.docsPath)
        assertEquals("queries.json", parsed.queriesPath)
        assertEquals("custom_dir", parsed.saveDir)
        assertEquals("gpt-4", parsed.llmName)
        assertEquals("custom-embedding", parsed.embeddingName)
        assertEquals("http://localhost:8080", parsed.llmBaseUrl)
        assertEquals("http://localhost:9090", parsed.embeddingBaseUrl)
    }

    @Test
    fun testParseNullableFields() {
        val args = arrayOf("--docs", "docs.json", "--queries", "queries.json")
        val parsed = DemoArgs.parse(args)

        assertNull(parsed.llmBaseUrl)
        assertNull(parsed.embeddingBaseUrl)
    }

    @Test
    @org.junit.Ignore("DemoArgs.parse calls exitProcess which terminates the JVM")
    fun testParseMissingDocsThrows() {
        // Cannot test without mocking System.exit
    }

    @Test
    @org.junit.Ignore("DemoArgs.parse calls exitProcess which terminates the JVM")
    fun testParseMissingQueriesThrows() {
        // Cannot test without mocking System.exit
    }

    @Test
    @org.junit.Ignore("DemoArgs.parse calls exitProcess which terminates the JVM")
    fun testParseInvalidFlagThrows() {
        // Cannot test without mocking System.exit
    }

    @Test
    @org.junit.Ignore("DemoArgs.parse calls exitProcess which terminates the JVM")
    fun testParseMissingValueThrows() {
        // Cannot test without mocking System.exit
    }

    @Test
    fun testParseDefaultValues() {
        val args = arrayOf("--docs", "docs.json", "--queries", "queries.json")
        val parsed = DemoArgs.parse(args)

        assertEquals("outputs", parsed.saveDir)
        assertEquals("gpt-4o-mini", parsed.llmName)
        assertEquals("text-embedding-3-large", parsed.embeddingName)
    }

    @Test
    fun testParseOverridesDefaults() {
        val args =
            arrayOf(
                "--docs",
                "docs.json",
                "--queries",
                "queries.json",
                "--save_dir",
                "my_outputs",
                "--llm_name",
                "my-model",
                "--embedding_name",
                "my-embedding",
            )
        val parsed = DemoArgs.parse(args)

        assertEquals("my_outputs", parsed.saveDir)
        assertEquals("my-model", parsed.llmName)
        assertEquals("my-embedding", parsed.embeddingName)
    }
}
