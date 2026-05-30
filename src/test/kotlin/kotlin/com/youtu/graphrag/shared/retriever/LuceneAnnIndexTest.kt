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
