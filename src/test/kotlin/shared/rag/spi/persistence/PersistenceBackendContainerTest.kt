package shared.rag.spi.persistence

import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.testcontainers.DockerClientFactory
import org.testcontainers.containers.MongoDBContainer
import org.testcontainers.containers.Neo4jContainer
import org.testcontainers.utility.DockerImageName
import java.nio.file.Files
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class PersistenceBackendContainerTest {
    @BeforeTest
    fun requireDocker() {
        val available = DockerClientFactory.instance().isDockerAvailable
        val message = "Testcontainers requires a reachable Docker daemon"
        if (System.getenv("GITHUB_ACTIONS") == "true") {
            assertTrue(available, message)
        } else {
            assumeTrue(message, available)
        }
    }

    @Test
    fun mongoPersistsAndRestoresAllStoreTypes() =
        runBlocking {
            MongoDBContainer(DockerImageName.parse("mongo:7.0")).use { mongo ->
                mongo.start()
                verifyPersistence(
                    backend = MongoDbPersistenceBackend(),
                    backendId = "mongodb",
                    config = mapOf("connectionString" to mongo.replicaSetUrl, "database" to "persistence_it"),
                )
            }
        }

    @Test
    fun neo4jPersistsAndRestoresAllStoreTypes() =
        runBlocking {
            val password = "testpassword"
            Neo4jContainer(DockerImageName.parse("neo4j:5.26")).withAdminPassword(password).use { neo4j ->
                neo4j.start()
                verifyPersistence(
                    backend = Neo4jPersistenceBackend(),
                    backendId = "neo4j",
                    config = mapOf("uri" to neo4j.boltUrl, "username" to "neo4j", "password" to password, "database" to "neo4j"),
                )
            }
        }

    private suspend fun verifyPersistence(
        backend: PersistenceBackendFactory,
        backendId: String,
        config: Map<String, Any?>,
    ) {
        val root = Files.createTempDirectory("spi-$backendId")
        try {
            val checkpoint = root.resolve("checkpoint").toString()
            val alpha = mapOf("label" to "Alpha")
            val forward = GraphEdgeRecord("a", "b", mapOf("relation" to "causes"))
            val reverse = GraphEdgeRecord("b", "a", mapOf("relation" to "describes"))
            val firstVector = VectorRecord("alpha", listOf(1.0, 0.0), mapOf("text" to "Alpha"))
            val secondVector = VectorRecord("beta", listOf(0.0, 1.0))
            val bytes = byteArrayOf(0, 1, -1)
            val metadata = mapOf("labels" to listOf("Alpha", "Beta"), "ready" to true)

            backend.open(config).use { session ->
                val graph = session.graph("knowledge")
                graph.upsertNode("a", alpha)
                graph.upsertNode("b", mapOf("label" to "Beta"))
                graph.upsertEdge(forward.source, forward.target, forward.data)
                graph.upsertEdge(reverse.source, reverse.target, reverse.data)
                val vectors = session.vector("passages", dimensions = 2)
                vectors.upsert(listOf(firstVector, secondVector))
                assertFailsWith<IllegalArgumentException> { vectors.upsert(VectorRecord("bad", listOf(1.0))) }
                session.kv("documents").put("doc", metadata)
                session.artifacts("files").writeBytes("nested/evidence.bin", bytes)
            }

            backend.open(config).use { session ->
                val graph = session.graph("knowledge")
                assertEquals(alpha, graph.nodes().single { it.id == "a" }.data)
                assertEquals(setOf(forward, reverse), graph.edges().toSet())
                val vectors = session.vector("passages", dimensions = 2)
                assertEquals(listOf(firstVector), vectors.query(listOf(1.0, 0.0), topK = 1))
                assertEquals(listOf(secondVector), vectors.query(listOf(0.0, 1.0), topK = 1))
                val kv = session.kv("documents")
                assertEquals(metadata, kv.get("doc"))
                val artifacts = session.artifacts("files")
                assertEquals(listOf("nested/evidence.bin"), artifacts.listPaths())
                assertContentEquals(bytes, artifacts.readBytes("nested/evidence.bin"))

                val manifest = session.checkpoint(checkpoint)
                assertEquals(backendId, manifest.backendId)
                assertEquals("knowledge", manifest.graph.single().namespace)
                assertEquals("passages", manifest.vector.single().namespace)
                assertEquals("documents", manifest.kv.single().namespace)
                assertEquals("files", manifest.artifacts.single().namespace)

                graph.deleteEdge(forward.source, forward.target)
                assertEquals(listOf(reverse), graph.edges())
                graph.deleteNode("b")
                assertTrue(graph.edges().isEmpty())
                vectors.delete("beta")
                kv.delete("doc")
                artifacts.delete("nested/evidence.bin")
                val discarded = session.graph("discarded")
                discarded.upsertEdge("extra", "target", emptyMap())
                assertEquals(setOf("extra", "target"), discarded.nodes().map { it.id }.toSet())
                assertEquals(listOf("a"), graph.nodes().map { it.id })
                assertEquals(listOf(firstVector), vectors.snapshot().records)
                assertNull(kv.get("doc"))
                assertNull(artifacts.readBytes("nested/evidence.bin"))

                session.restore(checkpoint)
                assertTrue(session.graph("discarded").nodes().isEmpty())
                val restored = session.graph("knowledge")
                assertEquals(mapOf("a" to alpha, "b" to mapOf("label" to "Beta")), restored.nodes().associate { it.id to it.data })
                assertEquals(setOf(forward, reverse), restored.edges().toSet())
                assertEquals(listOf(secondVector), session.vector("passages", dimensions = 2).query(listOf(0.0, 1.0), topK = 1))
                assertEquals(metadata, session.kv("documents").get("doc"))
                assertContentEquals(bytes, session.artifacts("files").readBytes("nested/evidence.bin"))
            }
        } finally {
            root.toFile().deleteRecursively()
        }
    }
}
