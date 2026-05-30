package com.youtu.graphrag.server.api

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.youtu.graphrag.shared.config.ConfigManager
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.readText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class GraphConstructionServiceTest {
    private val mapper = ObjectMapper().registerKotlinModule()

    @Test
    fun `construct graph writes graph and chunk outputs`() {
        val root = createRootDir()
        val config = createTestConfig(root)
        writeCorpus(
            root = root,
            datasetName = "sample_ds",
            documents =
                listOf(
                    mapOf(
                        "title" to "Alpha Document",
                        "text" to "Alpha text contains relationships and entities for graph construction.",
                    ),
                ),
        )

        val service = GraphConstructionService(config = config, rootDir = root)
        val result = service.constructGraph("sample_ds")

        assertTrue(result.success)
        assertEquals("Graph construction completed!", result.message)

        val graphPath = root.resolve("output/graphs/sample_ds_new.json")
        val chunksPath = root.resolve("output/chunks/sample_ds.txt")

        assertTrue(graphPath.exists())
        assertTrue(chunksPath.exists())
        assertTrue(chunksPath.readText().contains("id:"))

        val graphJson = mapper.readTree(graphPath.toFile())
        assertTrue(graphJson.isArray)
        graphJson.forEach { relation ->
            assertTrue(relation.has("startNode"), "Missing startNode")
            assertTrue(relation.has("relation"), "Missing relation")
            assertTrue(relation.has("endNode"), "Missing endNode")
        }
        assertNotNull(result.graphData["nodes"])
        assertNotNull(result.graphData["links"])
    }

    @Test
    fun `construct graph falls back to demo corpus when dataset corpus missing`() {
        val root = createRootDir()
        val config = createTestConfig(root)

        root.resolve("data/demo").createDirectories()
        mapper.writerWithDefaultPrettyPrinter().writeValue(
            root.resolve("data/demo/demo_corpus.json").toFile(),
            listOf(
                mapOf(
                    "title" to "Demo",
                    "text" to "This is fallback demo corpus content.",
                ),
            ),
        )

        val service = GraphConstructionService(config = config, rootDir = root)
        val result = service.constructGraph("missing_dataset")

        assertTrue(result.success)
        assertTrue(root.resolve("output/graphs/missing_dataset_new.json").exists())
        assertTrue(root.resolve("output/chunks/missing_dataset.txt").exists())
    }

    @Test
    fun `construct graph clears stale cache files with dataset prefix`() {
        val root = createRootDir()
        val config = createTestConfig(root)

        writeCorpus(
            root = root,
            datasetName = "stale_ds",
            documents = listOf(mapOf("title" to "Doc", "text" to "Some test text for construction.")),
        )

        root.resolve("output/graphs").createDirectories()
        root.resolve("output/chunks").createDirectories()
        root.resolve("output/logs").createDirectories()
        root.resolve("retriever/faiss_cache_new/stale_ds").createDirectories()

        Files.writeString(root.resolve("output/graphs/stale_ds_old.json"), "[]")
        Files.writeString(root.resolve("output/chunks/stale_ds_old.txt"), "old")
        Files.writeString(root.resolve("output/logs/stale_ds_old.log"), "old")
        Files.writeString(root.resolve("retriever/faiss_cache_new/stale_ds/cache.bin"), "old")

        val service = GraphConstructionService(config = config, rootDir = root)
        service.constructGraph("stale_ds")

        assertTrue(!root.resolve("output/graphs/stale_ds_old.json").exists())
        assertTrue(!root.resolve("output/chunks/stale_ds_old.txt").exists())
        assertTrue(!root.resolve("output/logs/stale_ds_old.log").exists())
        assertTrue(!root.resolve("retriever/faiss_cache_new/stale_ds").exists())

        assertTrue(root.resolve("output/graphs/stale_ds_new.json").exists())
        assertTrue(root.resolve("output/chunks/stale_ds.txt").exists())
    }

    @Test
    fun `construct graph throws when no dataset corpus and no demo corpus`() {
        val root = createRootDir()
        val config = createTestConfig(root)
        val service = GraphConstructionService(config = config, rootDir = root)

        assertFailsWith<DatasetNotFoundException> {
            service.constructGraph("no_data")
        }
    }

    @Test
    fun `reconstruct graph requires existing non-demo dataset`() {
        val root = createRootDir()
        val config = createTestConfig(root)
        val service = GraphConstructionService(config = config, rootDir = root)

        root.resolve("data/demo").createDirectories()
        mapper.writerWithDefaultPrettyPrinter().writeValue(
            root.resolve("data/demo/demo_corpus.json").toFile(),
            listOf(mapOf("title" to "Demo", "text" to "Demo content")),
        )

        assertFailsWith<DatasetNotFoundException> {
            service.reconstructGraph("missing_dataset")
        }
    }

    @Test
    fun `reconstruct graph succeeds for existing dataset`() {
        val root = createRootDir()
        val config = createTestConfig(root)
        writeCorpus(
            root = root,
            datasetName = "existing_ds",
            documents = listOf(mapOf("title" to "Doc", "text" to "Existing dataset corpus text.")),
        )

        val service = GraphConstructionService(config = config, rootDir = root)
        val result = service.reconstructGraph("existing_ds")

        assertTrue(result.success)
        assertEquals("Dataset reconstructed successfully", result.message)
        assertTrue(root.resolve("output/graphs/existing_ds_new.json").exists())
        assertTrue(root.resolve("output/chunks/existing_ds.txt").exists())
    }

    private fun createTestConfig(root: Path): ConfigManager {
        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
                "construction" to
                    mapOf(
                        "chunk_size" to 64,
                        "overlap" to 16,
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
            ),
        )
        return config
    }

    private fun writeCorpus(
        root: Path,
        datasetName: String,
        documents: List<Map<String, String>>,
    ) {
        val corpusDir = root.resolve("data/uploaded/$datasetName")
        corpusDir.createDirectories()
        mapper.writerWithDefaultPrettyPrinter().writeValue(corpusDir.resolve("corpus.json").toFile(), documents)
    }

    private fun createRootDir(): Path = Files.createTempDirectory("youtu-graphrag-graph-construction-test")
}
