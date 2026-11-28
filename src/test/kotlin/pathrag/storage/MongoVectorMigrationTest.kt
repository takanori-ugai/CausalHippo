package pathrag.storage

import com.mongodb.MongoClientSettings
import com.mongodb.bulk.BulkWriteResult
import com.mongodb.client.model.DeleteManyModel
import com.mongodb.client.model.DeleteOptions
import com.mongodb.client.model.Filters
import com.mongodb.client.model.ReplaceOneModel
import com.mongodb.client.model.WriteModel
import com.mongodb.client.result.DeleteResult
import com.mongodb.kotlin.client.coroutine.MongoClient
import com.mongodb.kotlin.client.coroutine.MongoCollection
import com.mongodb.kotlin.client.coroutine.MongoDatabase
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.slot
import io.mockk.unmockkObject
import kotlinx.coroutines.runBlocking
import org.bson.Document
import org.bson.conversions.Bson
import pathrag.utils.EmbeddingFunc
import pathrag.utils.computeMdHashId
import pathrag.utils.relationshipVectorId
import kotlin.test.Test
import kotlin.test.assertEquals

class MongoVectorMigrationTest {
    @Test
    fun replacesLegacyVectorsOnlyForMatchingEndpoints() =
        runBlocking {
            val uri = "mongodb://localhost:27017"
            mockkObject(MongoClient.Factory)
            try {
                val client = mockk<MongoClient>(relaxed = true)
                val database = mockk<MongoDatabase>()
                val collection = mockk<MongoCollection<Document>>()
                every { MongoClient.create(uri) } returns client
                every { client.getDatabase("test_db") } returns database
                every { database.getCollection("test_vector", Document::class.java) } returns collection
                val writes = slot<List<WriteModel<Document>>>()
                coEvery { collection.bulkWrite(capture(writes), any()) } returns mockk<BulkWriteResult>()
                coEvery { collection.deleteOne(any<Bson>(), any<DeleteOptions>()) } returns DeleteResult.acknowledged(1)
                coEvery { collection.deleteMany(any<Bson>(), any<DeleteOptions>()) } returns DeleteResult.acknowledged(1)
                val embedding = EmbeddingFunc(2, 10, { inputs -> inputs.map { doubleArrayOf(1.0, 0.0) } })
                MongoVectorStorage("test", mapOf("mongo_uri" to uri, "mongo_database" to "test_db"), embedding, emptySet()).use { store ->
                    val newId = relationshipVectorId("A", "BC")
                    store.upsert(mapOf(newId to mapOf("src_id" to "A", "tgt_id" to "BC", "content" to "evidence")))
                    val replacement = writes.captured[0] as ReplaceOneModel<Document>
                    assertEquals(newId, replacement.replacement.getString("_id"))
                    assertEquals("A", replacement.replacement.getString("src_id"))
                    assertEquals("BC", replacement.replacement.getString("tgt_id"))
                    val legacyRemoval = writes.captured[1] as DeleteManyModel<Document>
                    val endpoints = Filters.and(Filters.eq("src_id", "A"), Filters.eq("tgt_id", "BC"))
                    val legacyIds =
                        listOf(
                            computeMdHashId("ABC", prefix = "rel-"),
                            computeMdHashId("1:ABC", prefix = "rel-"),
                        )
                    val expected = Filters.and(Filters.`in`("_id", legacyIds), Filters.eq("src_id", "A"), Filters.eq("tgt_id", "BC"))
                    assertEquals(expected.asDocument(), legacyRemoval.filter.asDocument())

                    store.deleteRelationBetween("A", "BC")
                    coVerify { collection.deleteOne(match<Bson> { it.asDocument() == Document("_id", newId) }, any<DeleteOptions>()) }
                    coVerify {
                        collection.deleteMany(match<Bson> { it.asDocument() == endpoints.asDocument() }, any<DeleteOptions>())
                    }
                }
            } finally {
                unmockkObject(MongoClient.Factory)
            }
        }

    private fun Bson.asDocument(): Document =
        Document.parse(toBsonDocument(Document::class.java, MongoClientSettings.getDefaultCodecRegistry()).toJson())
}
