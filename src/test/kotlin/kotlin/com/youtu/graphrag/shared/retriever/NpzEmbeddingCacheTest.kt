package com.youtu.graphrag.shared.retriever

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NpzEmbeddingCacheTest {
    @Test
    fun `npz embedding cache round-trips float32 vectors`() {
        val root = Files.createTempDirectory("youtu-graphrag-npz-roundtrip-test")
        val npzPath = root.resolve("chunk_embedding_cache.npz")
        val source =
            linkedMapOf(
                "c1" to floatArrayOf(0.1f, 0.2f, 0.3f),
                "c2" to floatArrayOf(-0.4f, 0.5f, 0.6f),
            )

        NpzEmbeddingCache.write(
            path = npzPath,
            vectorsByKey = source,
            expectedDimensions = 3,
        )
        val loaded =
            NpzEmbeddingCache.read(
                path = npzPath,
                expectedDimensions = 3,
            )

        assertEquals(2, loaded.size)
        assertTrue(loaded.getValue("c1").contentEquals(source.getValue("c1")))
        assertTrue(loaded.getValue("c2").contentEquals(source.getValue("c2")))
    }

    @Test
    fun `npz embedding cache reader filters vectors with unexpected dimensions`() {
        val root = Files.createTempDirectory("youtu-graphrag-npz-filter-test")
        val npzPath = root.resolve("chunk_embedding_cache.npz")

        NpzEmbeddingCache.write(
            path = npzPath,
            vectorsByKey =
                mapOf(
                    "c1" to floatArrayOf(0.1f, 0.2f),
                    "c2" to floatArrayOf(0.3f, 0.4f),
                ),
            expectedDimensions = 2,
        )

        val loaded =
            NpzEmbeddingCache.read(
                path = npzPath,
                expectedDimensions = 3,
            )

        assertTrue(loaded.isEmpty())
    }

    @Test
    fun `npz embedding cache round-trips non ascii keys`() {
        val root = Files.createTempDirectory("youtu-graphrag-npz-unicode-key-test")
        val npzPath = root.resolve("triple_embedding_cache.npz")
        val key = "[\"Messi's goals\", \"compared_to\", \"ディエゴ・マラドーナ⚽\"]"
        val source = mapOf(key to floatArrayOf(1.0f, 2.0f, 3.0f))

        NpzEmbeddingCache.write(
            path = npzPath,
            vectorsByKey = source,
            expectedDimensions = 3,
        )
        val loaded =
            NpzEmbeddingCache.read(
                path = npzPath,
                expectedDimensions = 3,
            )

        assertEquals(1, loaded.size)
        assertTrue(loaded.containsKey(key))
        assertTrue(loaded.getValue(key).contentEquals(source.getValue(key)))
    }
}
