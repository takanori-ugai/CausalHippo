package shared.rag.unified

import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.llm.LlmClientFactory
import com.youtu.graphrag.shared.llm.NoopLlmClient
import io.mockk.every
import io.mockk.mockkObject
import io.mockk.unmockkObject
import kotlinx.coroutines.runBlocking
import shared.rag.spi.persistence.InMemoryPersistenceBackend
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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

    @Test
    fun `SPI artifacts hydrate a fresh adapter and survive a checkpoint round trip`() {
        val root = Files.createTempDirectory("youtu-spi-round-trip")
        val config = ConfigManager(writeMinimalYoutuConfig(root).toString())
        val fixture = Files.createDirectory(root.resolve("fixture"))
        val graph =
            """
            [{"start_node":{"label":"Person","properties":{"name":"Alice"}},"relation":"leads",
              "end_node":{"label":"Project","properties":{"name":"Apollo"}}}]
            """.trimIndent()
        val chunks = "id: 1\tChunk: Alice leads Apollo.\n"
        Files.writeString(fixture.resolve("sample_new.json"), graph)
        Files.writeString(fixture.resolve("sample.txt"), chunks)
        Files.writeString(fixture.resolve("corpus.json"), """[{"text":"Alice leads Apollo."}]""")

        mockkObject(LlmClientFactory)
        try {
            every { LlmClientFactory.fromEnvironment(any()) } returns NoopLlmClient()
            InMemoryPersistenceBackend().open(emptyMap()).use { session ->
                fun adapter(directory: Path) =
                    YoutuRagUnifiedAdapter(
                        config = config,
                        datasetName = "sample",
                        rootDir = directory,
                        persistenceSession = session,
                        useUnifiedSpiForRetrievalAndIndex = true,
                    )

                val writer = adapter(root.resolve("writer"))
                writer.loadGraph(fixture.toString())
                val files = session.kv("youturag_sample_files")
                assertEquals(graph, files.get("graph_json"))
                assertEquals(chunks, files.get("chunks_text"))
                assertEquals(
                    1,
                    session
                        .graph("youturag_sample_graph")
                        .snapshot()
                        .edges.size,
                )
                assertEquals(2, (session.kv("youturag_sample_metadata").get("nodeCount") as Number).toInt())

                val readerRoot = root.resolve("reader")
                val reader = adapter(readerRoot)
                val response = reader.query("Who leads Apollo?", UnifiedQuery(mode = UnifiedMode.NAIVE, topK = 2))
                assertEquals("unavailable", response.metadata["answerStatus"])
                assertEquals(null, response.answer)
                assertEquals(chunks, Files.readString(readerRoot.resolve("output/chunks/sample.txt")))
                assertTrue(Files.isRegularFile(readerRoot.resolve("data/uploaded/sample/corpus.json")))

                val checkpoint = root.resolve("checkpoint")
                session.kv("checkpoint-probe").put("marker", "before checkpoint")
                reader.saveGraph(checkpoint.toString())
                session.kv("checkpoint-probe").put("marker", "after checkpoint")
                assertEquals(graph, Files.readString(checkpoint.resolve("sample_new.json")))
                assertEquals(chunks, Files.readString(checkpoint.resolve("sample.txt")))
                assertTrue(Files.exists(root.resolve("checkpoint.unified_spi/manifest.json")))
                reader.drop()
                assertTrue(files.snapshot().entries.isEmpty())
                reader.loadGraph(checkpoint.toString())
                assertEquals("before checkpoint", session.kv("checkpoint-probe").get("marker"))
                assertEquals(graph, session.kv("youturag_sample_files").get("graph_json"))
                assertEquals(
                    1,
                    session
                        .graph("youturag_sample_graph")
                        .snapshot()
                        .edges.size,
                )
            }
        } finally {
            unmockkObject(LlmClientFactory)
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `loading a graph-only snapshot clears earlier chunks`() {
        val root = Files.createTempDirectory("youtu-graph-only-checkpoint")
        try {
            val config = ConfigManager(writeMinimalYoutuConfig(root).toString())
            val adapter = YoutuRagUnifiedAdapter(config, datasetName = "sample", rootDir = root)
            val snapshot = Files.createDirectory(root.resolve("snapshot"))
            Files.writeString(snapshot.resolve("sample_new.json"), "[]")
            val localChunks = root.resolve("output/chunks/sample.txt")
            Files.createDirectories(localChunks.parent)
            Files.writeString(localChunks, "previous generation")

            adapter.loadGraph(snapshot.toString())

            assertEquals("", Files.readString(localChunks))
            assertEquals("[]", Files.readString(root.resolve("output/graphs/sample_new.json")))
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `snapshot publication preserves earlier chunks when graph replacement fails`() {
        val root = Files.createTempDirectory("youtu-snapshot-publish")
        try {
            val config = ConfigManager(writeMinimalYoutuConfig(root).toString())
            val adapter = YoutuRagUnifiedAdapter(config, datasetName = "sample", rootDir = root)
            val source = Files.createDirectory(root.resolve("source"))
            Files.writeString(source.resolve("sample_new.json"), "[]")
            adapter.loadGraph(source.toString())

            val destination = Files.createDirectory(root.resolve("destination"))
            val previous = Files.createDirectory(destination.resolve("sample_new.json"))
            Files.writeString(previous.resolve("marker"), "previous graph")
            val destinationChunks = destination.resolve("sample.txt")
            Files.writeString(destinationChunks, "previous chunks")
            assertFailsWith<IOException> { adapter.saveGraph(destination.toString()) }
            assertEquals("previous chunks", Files.readString(destinationChunks))

            previous.toFile().deleteRecursively()
            Files.delete(root.resolve("output/chunks/sample.txt"))
            adapter.saveGraph(destination.toString())
            assertEquals("[]", Files.readString(destination.resolve("sample_new.json")))
            assertEquals("", Files.readString(destinationChunks))
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `SPI hydration keeps complete local artifacts and fills only incomplete pairs`() {
        val root = Files.createTempDirectory("youtu-spi-hydration")
        try {
            val config = ConfigManager(writeMinimalYoutuConfig(root).toString())
            InMemoryPersistenceBackend().open(emptyMap()).use { session ->
                fun adapter(dataset: String) =
                    YoutuRagUnifiedAdapter(
                        config,
                        datasetName = dataset,
                        rootDir = root,
                        persistenceSession = session,
                        useUnifiedSpiForRetrievalAndIndex = true,
                    )

                val graphs = Files.createDirectories(root.resolve("output/graphs"))
                val chunks = Files.createDirectories(root.resolve("output/chunks"))
                val localGraph = graphs.resolve("local_new.json")
                val localChunks = chunks.resolve("local.txt")
                Files.writeString(localGraph, "[]")
                Files.writeString(localChunks, "local chunks")
                session.kv("youturag_local_files").putAll(mapOf("graph_json" to "stale graph", "chunks_text" to "stale chunks"))

                adapter("local").inspectGraph()
                assertEquals("[]", Files.readString(localGraph))
                assertEquals("local chunks", Files.readString(localChunks))

                Files.writeString(graphs.resolve("incomplete_new.json"), "stale graph")
                session.kv("youturag_incomplete_files").put("graph_json", "[]")
                adapter("incomplete").inspectGraph()
                assertEquals("[]", Files.readString(graphs.resolve("incomplete_new.json")))
                assertEquals("", Files.readString(chunks.resolve("incomplete.txt")))

                val corpus = """[{"text":"corpus without published evidence"}]"""
                session.kv("youturag_empty_files").put("corpus_json", corpus)
                val empty = adapter("empty")
                empty.inspectGraph()
                assertEquals(corpus, Files.readString(root.resolve("data/uploaded/empty/corpus.json")))
                val snapshot = root.resolve("empty-snapshot")
                empty.saveGraph(snapshot.toString())
                assertTrue(Files.notExists(snapshot.resolve("empty_new.json")))
                empty.loadGraph(snapshot.toString())
                assertTrue(Files.notExists(graphs.resolve("empty_new.json")))
            }
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
