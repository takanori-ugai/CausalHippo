package shared.rag.unified

import com.youtu.graphrag.shared.config.ConfigManager
import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class YoutuRagUnifiedAdapterTest {
    @Test
    fun `adapter metadata-only query returns unified metadata without invoking qa`() {
        val root = Files.createTempDirectory("unified_youtu_adapter_query_test_")
        val configPath = writeMinimalYoutuConfig(root)
        try {
            val adapter =
                YoutuRagUnifiedAdapter(
                    config = ConfigManager(configPath.toString()),
                    datasetName = "demo",
                    rootDir = root,
                )

            val response =
                runBlocking {
                    adapter.aquery(
                        "What is the relation?",
                        UnifiedQuery(
                            mode = UnifiedMode.NAIVE,
                            includeAnswer = false,
                            includeContext = false,
                            includeReferences = false,
                            includeFollowUps = false,
                        ),
                    )
                }

            assertEquals(null, response.answer)
            assertTrue(response.context.isEmpty())
            assertTrue(response.references.isEmpty())
            assertTrue(response.followUpQueries.isEmpty())
            assertEquals(RagId.YOUTU_RAG.name, response.metadata["ragId"])
            assertEquals("noagent", response.metadata["modeUsed"])
            assertEquals("demo", response.metadata["datasetName"])
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `adapter inspect graph returns normalized empty graph shape when graph file is absent`() {
        val root = Files.createTempDirectory("unified_youtu_adapter_inspect_test_")
        val configPath = writeMinimalYoutuConfig(root)
        try {
            val adapter =
                YoutuRagUnifiedAdapter(
                    config = ConfigManager(configPath.toString()),
                    datasetName = "demo",
                    rootDir = root,
                )

            val inspection = adapter.inspectGraph()

            assertTrue(inspection["nodes"] is List<*>)
            assertTrue(inspection["edges"] is List<*>)
            val metadata = inspection["metadata"] as Map<*, *>
            assertEquals("demo", metadata["datasetName"])
            assertEquals(0, metadata["nodeCount"])
            assertEquals(0, metadata["edgeCount"])
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `factory creates youtu handle with unified persistence session`() {
        val root = Files.createTempDirectory("unified_youtu_factory_test_")
        val configPath = writeMinimalYoutuConfig(root)
        try {
            val handle =
                UnifiedRagFactory.create(
                    ragId = RagId.YOUTU_RAG,
                    configPath = configPath.toString(),
                    overrides =
                        mapOf(
                            "rootDir" to root.toString(),
                            "datasetName" to "demo",
                            "useUnifiedPersistence" to true,
                            "persistenceBackend" to "in_memory",
                        ),
                )

            assertEquals(RagId.YOUTU_RAG, handle.id)
            assertNotNull(handle.persistence)
            assertEquals("in_memory", handle.persistence.backendId)

            val inspection = handle.rag.inspectGraph()
            val metadata = inspection["metadata"] as Map<*, *>
            assertEquals("in_memory", metadata["persistenceBackend"])
            assertEquals(true, metadata["unifiedSpiEnabled"])
            handle.close()
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    private fun writeMinimalYoutuConfig(root: java.nio.file.Path): java.nio.file.Path {
        val configPath = root.resolve("base_config.yaml")
        Files.writeString(configPath, "{}")
        return configPath
    }
}
