package hipporag.embeddingstore

import hipporag.embeddingmodel.BaseEmbeddingModel
import hipporag.utils.EmbeddingRow
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.After
import org.junit.Before
import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class EmbeddingStoreTest {
    private lateinit var tempDir: File
    private lateinit var embeddingModel: BaseEmbeddingModel

    @Before
    fun setup() {
        tempDir = createTempDirectory("embedding_store_test").toFile()
        embeddingModel = mockk<BaseEmbeddingModel>()
    }

    @After
    fun cleanup() {
        tempDir.deleteRecursively()
    }

    @Test
    fun testInitialization() {
        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        val ids = store.getAllIds()
        assertTrue(ids.isEmpty())
    }

    @Test
    fun testInsertStrings() {
        every { embeddingModel.batchEncode(any()) } returns
            arrayOf(
                doubleArrayOf(0.1, 0.2, 0.3),
                doubleArrayOf(0.4, 0.5, 0.6),
            )

        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        val texts = listOf("text1", "text2")
        store.insertStrings(texts)

        val ids = store.getAllIds()
        assertEquals(2, ids.size)

        val allTexts = store.getAllTexts()
        assertTrue(allTexts.contains("text1"))
        assertTrue(allTexts.contains("text2"))

        verify { embeddingModel.batchEncode(any()) }
    }

    @Test
    fun testInsertStringsDuplicates() {
        every { embeddingModel.batchEncode(any()) } returns
            arrayOf(
                doubleArrayOf(0.1, 0.2, 0.3),
            )

        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        val texts = listOf("text1", "text1", "text1")
        store.insertStrings(texts)

        val ids = store.getAllIds()
        assertEquals(1, ids.size)
    }

    @Test
    fun testGetMissingStringHashIds() {
        every { embeddingModel.batchEncode(any()) } returns
            arrayOf(
                doubleArrayOf(0.1, 0.2, 0.3),
            )

        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        store.insertStrings(listOf("text1"))

        val missing = store.getMissingStringHashIds(listOf("text1", "text2", "text3"))
        assertEquals(2, missing.size)
        assertFalse(missing.any { it.value.content == "text1" })
    }

    @Test
    fun testGetRow() {
        every { embeddingModel.batchEncode(any()) } returns
            arrayOf(
                doubleArrayOf(0.1, 0.2, 0.3),
            )

        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        store.insertStrings(listOf("text1"))

        val ids = store.getAllIds()
        val row = store.getRow(ids[0])
        assertEquals("text1", row.content)
        assertEquals(ids[0], row.hashId)
    }

    @Test
    fun testGetEmbedding() {
        every { embeddingModel.batchEncode(any()) } returns
            arrayOf(
                doubleArrayOf(0.1, 0.2, 0.3),
            )

        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        store.insertStrings(listOf("text1"))

        val ids = store.getAllIds()
        val embedding = store.getEmbedding(ids[0])
        assertEquals(3, embedding.size)
        assertEquals(0.1, embedding[0])
        assertEquals(0.2, embedding[1])
        assertEquals(0.3, embedding[2])
    }

    @Test
    fun testGetEmbeddings() {
        every { embeddingModel.batchEncode(any()) } returns
            arrayOf(
                doubleArrayOf(0.1, 0.2, 0.3),
                doubleArrayOf(0.4, 0.5, 0.6),
            )

        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        store.insertStrings(listOf("text1", "text2"))

        val ids = store.getAllIds()
        val embeddings = store.getEmbeddings(ids)
        assertEquals(2, embeddings.size)
        assertEquals(3, embeddings[0].size)
    }

    @Test
    fun testDelete() {
        every { embeddingModel.batchEncode(any()) } returns
            arrayOf(
                doubleArrayOf(0.1, 0.2, 0.3),
                doubleArrayOf(0.4, 0.5, 0.6),
                doubleArrayOf(0.7, 0.8, 0.9),
            )

        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        store.insertStrings(listOf("text1", "text2", "text3"))

        val ids = store.getAllIds()
        assertEquals(3, ids.size)

        store.delete(listOf(ids[1]))
        val remainingIds = store.getAllIds()
        assertEquals(2, remainingIds.size)
        assertFalse(remainingIds.contains(ids[1]))
    }

    @Test
    fun testGetAllIdToRows() {
        every { embeddingModel.batchEncode(any()) } returns
            arrayOf(
                doubleArrayOf(0.1, 0.2, 0.3),
                doubleArrayOf(0.4, 0.5, 0.6),
            )

        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        store.insertStrings(listOf("text1", "text2"))

        val idToRows = store.getAllIdToRows()
        assertEquals(2, idToRows.size)
        assertTrue(idToRows.values.any { it.content == "text1" })
        assertTrue(idToRows.values.any { it.content == "text2" })
    }

    @Test
    fun testGetRows() {
        every { embeddingModel.batchEncode(any()) } returns
            arrayOf(
                doubleArrayOf(0.1, 0.2, 0.3),
                doubleArrayOf(0.4, 0.5, 0.6),
            )

        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        store.insertStrings(listOf("text1", "text2"))

        val ids = store.getAllIds()
        val rows = store.getRows(ids)
        assertEquals(2, rows.size)
        assertEquals(ids[0], rows[ids[0]]?.hashId)
        assertEquals(ids[1], rows[ids[1]]?.hashId)
    }

    @Test
    fun testPersistenceAcrossInstances() {
        every { embeddingModel.batchEncode(any()) } returns
            arrayOf(
                doubleArrayOf(0.1, 0.2, 0.3),
            )

        val store1 = EmbeddingStore(embeddingModel, tempDir.path, "test")
        store1.insertStrings(listOf("text1"))

        val store2 = EmbeddingStore(embeddingModel, tempDir.path, "test")
        val ids = store2.getAllIds()
        assertEquals(1, ids.size)

        val row = store2.getRow(ids[0])
        assertEquals("text1", row.content)
    }

    @Test
    fun testTextToHashIdMapping() {
        every { embeddingModel.batchEncode(any()) } returns
            arrayOf(
                doubleArrayOf(0.1, 0.2, 0.3),
            )

        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        store.insertStrings(listOf("text1"))

        val mapping = store.textToHashId
        assertTrue(mapping.containsKey("text1"))

        val hashId = store.getHashId("text1")
        assertEquals(mapping["text1"], hashId)
    }

    @Test
    fun testGetHashIdNotFoundThrows() {
        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        assertFailsWith<IllegalStateException> {
            store.getHashId("nonexistent")
        }
    }

    @Test
    fun testInsertEmptyList() {
        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        store.insertStrings(emptyList())

        val ids = store.getAllIds()
        assertTrue(ids.isEmpty())
    }

    @Test
    fun testGetRowsEmptyList() {
        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        val rows = store.getRows(emptyList())
        assertTrue(rows.isEmpty())
    }

    @Test
    fun testGetEmbeddingsEmptyList() {
        val store = EmbeddingStore(embeddingModel, tempDir.path, "test")
        val embeddings = store.getEmbeddings(emptyList())
        assertTrue(embeddings.isEmpty())
    }
}
