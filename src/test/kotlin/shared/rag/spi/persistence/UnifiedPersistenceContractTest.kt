package shared.rag.spi.persistence

import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class UnifiedPersistenceContractTest {
    @Test
    fun `offline backends expose required capabilities and preserve metadata`() =
        runBlocking {
            val root = Files.createTempDirectory("persistence_contract_capabilities_")
            val backends =
                listOf(
                    BackendCase("filesystem_snapshot", FilesystemSnapshotBackend(), mapOf("rootDir" to root.resolve("fs").toString())),
                    BackendCase("in_memory", InMemoryPersistenceBackend(), emptyMap()),
                )

            try {
                backends.forEach { backend ->
                    backend.factory
                        .open(
                            backend.config +
                                mapOf(
                                    "metadata" to
                                        mapOf(
                                            "suite" to "p4",
                                            "backend" to backend.id,
                                        ),
                                ),
                        ).use { session ->
                            assertEquals(backend.id, session.backendId)
                            assertTrue(session.capabilities.supportsGraphPersistence)
                            assertTrue(session.capabilities.supportsVectorPersistence)
                            assertTrue(session.capabilities.supportsKvPersistence)
                            assertTrue(session.capabilities.supportsAtomicCheckpoint)
                            assertFalse(session.capabilities.supportsIncrementalCheckpoint)
                            assertTrue(session.capabilities.supportsCrossBackendImport)

                            session.graph("g").upsertNode("n1", mapOf("k" to "v"))
                            val manifest = session.checkpoint(root.resolve("checkpoint-${backend.id}").toString())
                            assertEquals("p4", manifest.metadata["suite"])
                            assertEquals(backend.id, manifest.metadata["backend"])
                        }
                }
            } finally {
                root.toFile().deleteRecursively()
            }
        }

    private data class BackendCase(
        val id: String,
        val factory: PersistenceBackendFactory,
        val config: Map<String, Any?>,
    )
}
