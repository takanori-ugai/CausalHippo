package shared.rag.spi.persistence

import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class UnifiedPersistenceBackendMatrixTest {
    @Test
    fun `checkpoint produced by each offline backend restores across backend matrix`() =
        runBlocking {
            val backends =
                listOf(
                    BackendCase("filesystem_snapshot", FilesystemSnapshotBackend()) { root ->
                        mapOf("rootDir" to root.resolve("fs_root").toString())
                    },
                    BackendCase("in_memory", InMemoryPersistenceBackend()),
                )

            backends.forEach { producer ->
                backends.forEach { consumer ->
                    val root = Files.createTempDirectory("persistence_matrix_${producer.id}_${consumer.id}_")
                    val checkpointPath = root.resolve("checkpoint")
                    try {
                        producer.factory.open(producer.config(root)).use { session ->
                            seedState(session)
                            val manifest = session.checkpoint(checkpointPath.toString())
                            assertEquals(producer.id, manifest.backendId)
                        }

                        consumer.factory.open(consumer.config(root)).use { session ->
                            session.restore(checkpointPath.toString())
                            assertCanonicalState(session)
                        }
                    } finally {
                        root.toFile().deleteRecursively()
                    }
                }
            }
        }

    private fun seedState(session: PersistenceSession) {
        session.graph("knowledge").apply {
            upsertNode("alpha", mapOf("label" to "Alpha"))
            upsertNode("beta", mapOf("label" to "Beta"))
            upsertEdge("alpha", "beta", mapOf("relation" to "influences", "weight" to 0.9))
        }

        session.vector("entities", dimensions = 3, metric = "cosine").upsert(
            listOf(
                VectorRecord("v-alpha", listOf(1.0, 0.0, 0.0), mapOf("entityId" to "alpha")),
                VectorRecord("v-beta", listOf(0.0, 1.0, 0.0), mapOf("entityId" to "beta")),
            ),
        )

        session.kv("docs").putAll(
            mapOf(
                "doc-alpha" to mapOf("title" to "Alpha"),
                "doc-beta" to mapOf("title" to "Beta"),
            ),
        )

        session.artifacts("files").writeText("notes/context.txt", "alpha->beta")
    }

    private fun assertCanonicalState(session: PersistenceSession) {
        val graphSnapshot = session.graph("knowledge").snapshot()
        assertEquals(2, graphSnapshot.nodes.size)
        assertEquals(1, graphSnapshot.edges.size)
        assertEquals("alpha", graphSnapshot.edges.single().source)
        assertEquals("beta", graphSnapshot.edges.single().target)

        val vectorSnapshot = session.vector("entities", dimensions = 3, metric = "cosine").snapshot()
        assertEquals(3, vectorSnapshot.dimensions)
        assertEquals("cosine", vectorSnapshot.metric)
        assertEquals(setOf("v-alpha", "v-beta"), vectorSnapshot.records.map { it.id }.toSet())

        val docs = session.kv("docs").entries()
        assertEquals(setOf("doc-alpha", "doc-beta"), docs.keys)
        val alphaDoc = docs["doc-alpha"] as? Map<*, *>
        assertNotNull(alphaDoc)
        assertEquals("Alpha", alphaDoc["title"])

        val artifact = session.artifacts("files")
        assertEquals(listOf("notes/context.txt"), artifact.listPaths())
        assertEquals("alpha->beta", artifact.readText("notes/context.txt"))
    }

    private data class BackendCase(
        val id: String,
        val factory: PersistenceBackendFactory,
        val config: (Path) -> Map<String, Any?> = { emptyMap() },
    )
}
