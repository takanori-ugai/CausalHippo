package causalrag

import causalrag.retriever.VectorStoreRetriever
import causalrag.utils.EmbeddingModel
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests vector index persistence and cache integrity checks.
 */
class TestVectorStoreRetriever {
    private lateinit var tempDir: Path

    /**
     * Creates a temporary directory for each test.
     */
    @BeforeTest
    fun setUp() {
        tempDir = Files.createTempDirectory("causalrag-vectors")
    }

    /**
     * Removes temporary test files and directories.
     */
    @AfterTest
    fun tearDown() {
        tempDir.toFile().deleteRecursively()
    }

    /**
     * Verifies that reindexing replaces passages even when storeOriginal is disabled.
     */
    @Test
    fun testReindexKeepsPassagesAligned() {
        val retriever = VectorStoreRetriever(dimension = 64)
        retriever.indexCorpus(listOf("alpha signal"), storeOriginal = false)
        retriever.indexCorpus(listOf("beta signal"), storeOriginal = false)

        assertEquals(listOf("beta signal"), retriever.getPassages())
        assertEquals(listOf("beta signal"), retriever.search("beta", topK = 1))
    }

    /**
     * Verifies that saveIndex reports failure when the target cannot be created as a directory.
     */
    @Test
    fun testSaveIndexFailsWhenCacheWriteFails() {
        val retriever = VectorStoreRetriever(dimension = 64)
        retriever.indexCorpus(listOf("cached document"))
        val filePath = Files.createTempFile(tempDir, "vector-cache", ".tmp")

        val saved = retriever.saveIndex(filePath.toString())

        assertFalse(saved)
    }

    /**
     * Verifies that malformed caches with mismatched passage and vector counts are rejected.
     */
    @Test
    fun testLoadIndexRejectsMismatchedPassageCount() {
        val cacheDir = tempDir.resolve("bad-count-cache")
        Files.createDirectories(cacheDir)
        Files.writeString(cacheDir.resolve("vectors.json"), "[[1.0, 0.0, 0.0]]")
        Files.writeString(
            cacheDir.resolve("metadata.json"),
            """
            {
              "passages": [],
              "metadata": [],
              "vector_dimension": 3
            }
            """.trimIndent(),
        )

        val retriever = VectorStoreRetriever(dimension = 3)
        val loaded = retriever.loadIndex(cacheDir.toString())

        assertFalse(loaded)
        assertTrue(retriever.getPassages().isEmpty())
    }

    /**
     * Verifies that caches built with a different vector dimension are rejected.
     */
    @Test
    fun testLoadIndexRejectsDimensionMismatch() {
        val cacheDir = tempDir.resolve("dimension-cache")
        val writer = VectorStoreRetriever(dimension = 64)
        writer.indexCorpus(listOf("dimensioned document"))
        assertTrue(writer.saveIndex(cacheDir.toString()))

        val reader = VectorStoreRetriever(dimension = 32)
        val loaded = reader.loadIndex(cacheDir.toString())

        assertFalse(loaded)
        assertTrue(reader.getPassages().isEmpty())
    }

    @Test
    fun testRejectsDifferentEmbeddingModelWithSameDimension() {
        val cache = tempDir.resolve("different-model")
        val writer = VectorStoreRetriever(embeddingModel = "first-model", dimension = 64)
        writer.indexCorpus(listOf("foreign evidence"))
        assertTrue(writer.saveIndex(cache.toString()))

        val reader = VectorStoreRetriever(embeddingModel = "second-model", dimension = 64)
        reader.indexCorpus(listOf("local evidence"))
        assertFalse(reader.loadIndex(cache.toString()))
        assertEquals(listOf("local evidence"), reader.getPassages())
    }

    @Test
    fun testFailedEmbeddingBatchKeepsPreviousIndex() {
        val encoder =
            object : EmbeddingModel {
                override fun encode(text: String): DoubleArray {
                    check(text != "failure") { "Injected embedding error" }
                    return doubleArrayOf(1.0)
                }
            }
        val retriever = VectorStoreRetriever(dimension = 1, batchSize = 1, embeddingModelOverride = encoder)
        retriever.indexCorpus(listOf("existing"))

        assertFailsWith<IllegalStateException> { retriever.indexCorpus(listOf("new", "failure")) }
        assertEquals(listOf("existing"), retriever.getPassages())
        assertEquals(listOf("existing"), retriever.search("query"))
    }
}
