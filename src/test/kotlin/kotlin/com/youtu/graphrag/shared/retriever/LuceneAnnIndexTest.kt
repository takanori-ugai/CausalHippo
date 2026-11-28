package com.youtu.graphrag.shared.retriever

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LuceneAnnIndexTest {
    @Test
    fun `lucene ann index returns nearest item for lexical query`() {
        val items =
            listOf(
                "alpha relates beta",
                "guitar melody rhythm",
                "tokyo city japan",
            )

        val index =
            LuceneAnnIndex.build(
                items = items,
                embedder = HashTextEmbedder(dimensions = 256),
                textSelector = { value -> value },
            )

        val hits = index.search(query = "capital city tokyo", limit = 2)

        assertTrue(hits.isNotEmpty())
        assertEquals("tokyo city japan", hits.first().item)
        assertEquals("hybrid_lexical_lucene_ann", index.strategyTag)
    }

    @Test
    fun `both indexes rank non-unit vectors by direction within the eligible set`() {
        val embedder =
            object : TextEmbedder {
                override val dimensions = 2
                override val modelTag = "test"

                override fun embed(text: String) = floatArrayOf(2f, 0f)
            }
        val items = listOf("aligned", "angled", "excluded")
        val vectors = listOf(floatArrayOf(100f, 0f), floatArrayOf(2f, 1f), floatArrayOf(2f, 0f))
        val vectorFor: (String) -> FloatArray = { vectors[items.indexOf(it)] }
        val indexes =
            listOf(
                LocalVectorIndex.build(items, embedder, { it }, vectorFor),
                LuceneAnnIndex.build(items, embedder, { it }, vectorFor),
            )
        for (index in indexes) {
            assertEquals(listOf("aligned", "angled"), index.search("query", 2, setOf(0, 1)).map { it.item })
            assertEquals("angled", index.search("query", 1, setOf(1)).single().item)
        }
    }

    @Test
    fun `lucene ann index honors result limit`() {
        val items =
            listOf(
                "node one",
                "node two",
                "node three",
            )

        val index =
            LuceneAnnIndex.build(
                items = items,
                textSelector = { value -> value },
            )

        val hits = index.search(query = "node", limit = 1)
        assertEquals(1, hits.size)
    }
}
