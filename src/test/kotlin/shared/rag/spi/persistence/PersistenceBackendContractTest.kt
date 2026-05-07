package shared.rag.spi.persistence

import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class PersistenceBackendContractTest {
    @Test
    fun `neo4j backend session initializes capability metadata`() {
        val backend = Neo4jPersistenceBackend()
        backend.open(
            mapOf(
                "uri" to "bolt://localhost:7687",
                "username" to "neo4j",
                "password" to "test",
                "database" to "neo4j",
            ),
        ).use { session ->
            assertEquals("neo4j", session.backendId)
            assertTrue(session.capabilities.supportsGraphPersistence)
            assertTrue(session.capabilities.supportsVectorPersistence)
            assertTrue(session.capabilities.supportsKvPersistence)
        }
    }

    @Test
    fun `mongodb backend session initializes capability metadata`() {
        val backend = MongoDbPersistenceBackend()
        backend.open(
            mapOf(
                "connectionString" to "mongodb://localhost:27017",
                "database" to "unified_rag_test",
            ),
        ).use { session ->
            assertEquals("mongodb", session.backendId)
            assertTrue(session.capabilities.supportsGraphPersistence)
            assertTrue(session.capabilities.supportsVectorPersistence)
            assertTrue(session.capabilities.supportsKvPersistence)
        }
    }

    @Test
    fun `in-memory backend supports checkpoint and restore round-trip`() =
        runBlocking {
            val tempDir = Files.createTempDirectory("persistence_in_memory_roundtrip")
            val checkpoint = tempDir.resolve("snapshot")

            val backend = InMemoryPersistenceBackend()
            backend.open().use { session ->
                val graph = session.graph("knowledge")
                graph.upsertNode("A", mapOf("label" to "Alpha"))
                graph.upsertNode("B", mapOf("label" to "Beta"))
                graph.upsertEdge("A", "B", mapOf("weight" to 0.9))

                val vectors = session.vector("entities", dimensions = 2)
                vectors.upsert(VectorRecord(id = "v1", vector = listOf(1.0, 0.0), metadata = mapOf("text" to "alpha")))
                vectors.upsert(VectorRecord(id = "v2", vector = listOf(0.0, 1.0), metadata = mapOf("text" to "beta")))

                val kv = session.kv("docs")
                kv.put("doc-1", mapOf("title" to "Doc1", "lang" to "en"))

                val artifacts = session.artifacts("blobs")
                artifacts.writeText("notes/readme.txt", "hello")

                val manifest = session.checkpoint(checkpoint.toString())
                assertEquals("in_memory", manifest.backendId)
                assertEquals(1, manifest.graph.size)
                assertEquals(1, manifest.vector.size)
                assertEquals(1, manifest.kv.size)
                assertEquals(1, manifest.artifacts.size)
            }

            backend.open().use { restored ->
                restored.restore(checkpoint.toString())

                val graph = restored.graph("knowledge")
                assertEquals(2, graph.nodes().size)
                assertEquals(1, graph.edges().size)

                val nearest = restored.vector("entities", dimensions = 2).query(listOf(1.0, 0.0), topK = 1)
                assertEquals(1, nearest.size)
                assertEquals("v1", nearest.first().id)

                val doc = restored.kv("docs").get("doc-1")
                assertNotNull(doc)
                val docMap = doc as Map<*, *>
                assertEquals("Doc1", docMap["title"])

                val text = restored.artifacts("blobs").readText("notes/readme.txt")
                assertEquals("hello", text)
            }
        }

    @Test
    fun `filesystem backend resolves relative checkpoint path against rootDir`() =
        runBlocking {
            val root = Files.createTempDirectory("persistence_fs_root")
            val backend = FilesystemSnapshotBackend()

            backend.open(mapOf("rootDir" to root.toString())).use { session ->
                session.kv("meta").put("k", "v")
                val manifest = session.checkpoint("snap_a")
                assertEquals("filesystem_snapshot", manifest.backendId)
            }

            val manifestPath = root.resolve("snap_a").resolve("manifest.json")
            assertTrue(Files.exists(manifestPath), "Expected manifest at $manifestPath")

            backend.open(mapOf("rootDir" to root.toString())).use { session ->
                session.restore("snap_a")
                assertEquals("v", session.kv("meta").get("k"))
            }
        }

    @Test
    fun `snapshot created by filesystem backend can be restored by in-memory backend`() =
        runBlocking {
            val tempDir = Files.createTempDirectory("persistence_cross_backend")
            val snapshotDir = tempDir.resolve("snapshot_cross")

            val fs = FilesystemSnapshotBackend()
            fs.open().use { session ->
                session.graph("g").upsertNode("X", mapOf("label" to "NodeX"))
                session.vector("v", dimensions = 2).upsert(VectorRecord("vx", listOf(0.5, 0.5)))
                session.kv("k").put("ready", true)
                session.artifacts("a").writeBytes("binary.bin", byteArrayOf(1, 2, 3))
                session.checkpoint(snapshotDir.toString())
            }

            val mem = InMemoryPersistenceBackend()
            mem.open().use { session ->
                session.restore(snapshotDir.toString())
                assertEquals(1, session.graph("g").nodes().size)
                assertEquals(1, session.vector("v", dimensions = 2).query(listOf(0.5, 0.5), topK = 1).size)
                assertEquals(true, session.kv("k").get("ready"))
                val blob = session.artifacts("a").readBytes("binary.bin")
                assertNotNull(blob)
                assertEquals(3, blob.size)
            }
        }
}
