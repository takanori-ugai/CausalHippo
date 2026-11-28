package pathrag.storage

import com.mongodb.kotlin.client.coroutine.MongoClient
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.bson.Document
import org.junit.Assume.assumeTrue
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.GraphDatabase
import org.testcontainers.DockerClientFactory
import org.testcontainers.containers.MongoDBContainer
import org.testcontainers.containers.Neo4jContainer
import org.testcontainers.utility.DockerImageName
import pathrag.base.BaseVectorStorage
import pathrag.utils.EmbeddingFunc
import pathrag.utils.computeMdHashId
import pathrag.utils.relationshipVectorId
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class RelationshipVectorContainerTest {
    private val namespace = "relationships"
    private val embedding = EmbeddingFunc(3, 100, { texts -> texts.map { doubleArrayOf(1.0, 0.0, 0.0) } })

    @BeforeTest
    fun requireDocker() {
        assumeTrue("Testcontainers requires a reachable Docker daemon", DockerClientFactory.instance().isDockerAvailable)
    }

    @Test
    fun mongoReplacesAndDeletesOnlyTheRequestedRelationship() =
        runBlocking {
            MongoDBContainer(DockerImageName.parse("mongo:7.0")).use { mongo ->
                mongo.start()
                val database = "pathrag"
                val config = mapOf("mongo_uri" to mongo.replicaSetUrl, "mongo_database" to database)
                MongoClient.create(mongo.replicaSetUrl).use { client ->
                    val collection = client.getDatabase(database).getCollection<Document>("${namespace}_vector")
                    MongoVectorStorage(namespace, config, embedding).use { legacy ->
                        MongoVectorStorage(namespace, config, embedding, emptySet()).use { current ->
                            verifyMigration(legacy, current) {
                                collection.find().toList().associate { row ->
                                    row.getString("_id") to (row.getString("src_id") to row.getString("tgt_id"))
                                }
                            }
                        }
                    }
                }
            }
        }

    @Test
    fun neo4jReplacesAndDeletesOnlyTheRequestedRelationship() =
        runBlocking {
            val password = "testpassword"
            Neo4jContainer(DockerImageName.parse("neo4j:5.26")).withAdminPassword(password).use { neo4j ->
                neo4j.start()
                val config = mapOf("neo4j_uri" to neo4j.boltUrl, "neo4j_user" to "neo4j", "neo4j_password" to password)
                GraphDatabase.driver(neo4j.boltUrl, AuthTokens.basic("neo4j", password)).use { driver ->
                    driver.verifyConnectivity()
                    Neo4jVectorStorage(namespace, config, embedding).use { legacy ->
                        Neo4jVectorStorage(namespace, config, embedding, emptySet()).use { current ->
                            verifyMigration(legacy, current) {
                                driver
                                    .executableQuery("MATCH (v:RELATIONSHIPS_VECTOR) RETURN v.id AS id, v.src_id AS src, v.tgt_id AS tgt")
                                    .execute()
                                    .records()
                                    .associate { row ->
                                        row["id"].asString() to (row["src"].asString(null) to row["tgt"].asString(null))
                                    }
                            }
                        }
                    }
                }
            }
        }

    private suspend fun verifyMigration(
        legacy: BaseVectorStorage,
        current: BaseVectorStorage,
        read: suspend () -> Map<String, Pair<String?, String?>>,
    ) {
        val sharedLegacyId = computeMdHashId("ABC", prefix = "rel-")
        val secondLegacyId = computeMdHashId("1:ABC", prefix = "rel-")
        val opaqueId = computeMdHashId("XYZ", prefix = "rel-")
        val relationId = relationshipVectorId("A", "BC")
        val otherId = relationshipVectorId("AB", "C")

        legacy.upsert(
            mapOf(
                sharedLegacyId to relationship("A", "BC"),
                secondLegacyId to relationship("1:A", "BC"),
            ),
        )
        current.upsert(mapOf(opaqueId to relationship("X", "YZ"), otherId to relationship("AB", "C")))
        assertEquals(
            mapOf(sharedLegacyId to ("A" to "BC"), secondLegacyId to ("1:A" to "BC"), opaqueId to (null to null), otherId to ("AB" to "C")),
            read(),
        )

        current.upsert(mapOf(relationId to relationship("A", "BC")))
        assertEquals(
            mapOf(relationId to ("A" to "BC"), secondLegacyId to ("1:A" to "BC"), opaqueId to (null to null), otherId to ("AB" to "C")),
            read(),
        )

        current.deleteRelationBetween("A", "BC")
        current.deleteRelationBetween("X", "YZ")
        assertEquals(mapOf(secondLegacyId to ("1:A" to "BC"), opaqueId to (null to null), otherId to ("AB" to "C")), read())
    }

    private fun relationship(
        source: String,
        target: String,
    ) = mapOf("src_id" to source, "tgt_id" to target, "content" to "$source connects to $target")
}
