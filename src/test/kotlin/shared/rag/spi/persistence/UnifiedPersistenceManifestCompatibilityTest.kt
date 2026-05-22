package shared.rag.spi.persistence

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class UnifiedPersistenceManifestCompatibilityTest {
    private val objectMapper = jacksonObjectMapper().findAndRegisterModules()

    @Test
    fun `filesystem checkpoint writes versioned canonical manifest shape`() =
        runBlocking {
            val root = Files.createTempDirectory("persistence_manifest_shape_")
            val checkpointPath = root.resolve("snapshot")
            try {
                val backend = FilesystemSnapshotBackend()
                val manifest =
                    backend.open(mapOf("metadata" to mapOf("suite" to "p4"))).use { session ->
                        session.graph("graph-ns").apply {
                            upsertNode("n1", mapOf("kind" to "entity"))
                            upsertEdge("n1", "n2", mapOf("type" to "linked"))
                        }
                        session.vector("vec-ns", dimensions = 2, metric = "cosine").upsert(
                            VectorRecord("v1", listOf(1.0, 0.0), mapOf("source" to "seed")),
                        )
                        session.kv("kv-ns").put("k1", "v1")
                        session.artifacts("art-ns").writeText("readme.txt", "ok")
                        session.checkpoint(checkpointPath.toString())
                    }

                assertEquals(1, manifest.manifestVersion)
                assertEquals("filesystem_snapshot", manifest.backendId)
                assertEquals("p4", manifest.metadata["suite"])

                val manifestPath = checkpointPath.resolve("manifest.json")
                assertTrue(Files.exists(manifestPath))

                val tree = objectMapper.readTree(manifestPath.toFile())
                assertEquals(1, tree["manifestVersion"].asInt())
                assertEquals("filesystem_snapshot", tree["backendId"].asText())
                assertTrue(tree.has("createdAt"))
                assertTrue(tree.has("graph"))
                assertTrue(tree.has("vector"))
                assertTrue(tree.has("kv"))
                assertTrue(tree.has("artifacts"))
                assertTrue(tree.has("metadata"))

                assertTrue(
                    manifest.graph
                        .single()
                        .payloadPath
                        .startsWith("graph/"),
                )
                assertTrue(
                    manifest.vector
                        .single()
                        .payloadPath
                        .startsWith("vector/"),
                )
                assertTrue(
                    manifest.kv
                        .single()
                        .payloadPath
                        .startsWith("kv/"),
                )
                assertTrue(
                    manifest.artifacts
                        .single()
                        .payloadPath
                        .startsWith("artifacts/"),
                )
            } finally {
                root.toFile().deleteRecursively()
            }
        }

    @Test
    fun `manifest produced by filesystem backend restores into in-memory backend`() =
        runBlocking {
            val root = Files.createTempDirectory("persistence_manifest_cross_restore_")
            val checkpointPath = root.resolve("snapshot")
            try {
                FilesystemSnapshotBackend().open().use { session ->
                    session.graph("g").upsertNode("A", mapOf("name" to "Alpha"))
                    session.vector("v", dimensions = 2, metric = "cosine").upsert(
                        VectorRecord("vec-A", listOf(0.3, 0.7)),
                    )
                    session.kv("k").put("flag", true)
                    session.artifacts("a").writeText("data.txt", "payload")
                    session.checkpoint(checkpointPath.toString())
                }

                InMemoryPersistenceBackend().open().use { session ->
                    session.restore(checkpointPath.toString())

                    assertEquals(1, session.graph("g").nodes().size)
                    assertEquals(
                        1,
                        session
                            .vector("v", dimensions = 2, metric = "cosine")
                            .snapshot()
                            .records.size,
                    )
                    assertEquals(true, session.kv("k").get("flag"))
                    val artifact = session.artifacts("a").readText("data.txt")
                    assertNotNull(artifact)
                    assertEquals("payload", artifact)
                }
            } finally {
                root.toFile().deleteRecursively()
            }
        }
}
