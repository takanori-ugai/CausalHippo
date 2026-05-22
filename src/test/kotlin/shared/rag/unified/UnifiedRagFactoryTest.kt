package shared.rag.unified

import shared.rag.spi.persistence.VectorRecord
import shared.rag.spi.persistence.VectorSnapshot
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class UnifiedRagFactoryTest {
    @Test
    fun `factory uses unified persistence defaults from common config`() {
        val root = Files.createTempDirectory("unified_defaults_common_config_test_")
        val configPath = root.resolve("common_rag.json")
        try {
            Files.writeString(
                configPath,
                """
                {
                  "shared": {
                    "modelName": "gpt-4o-mini",
                    "embeddingModel": "all-MiniLM-L6-v2",
                    "embeddingApiKey": ""
                  },
                  "unified": {
                    "useUnifiedPersistence": true,
                    "useUnifiedSpiForRetrievalAndIndex": true,
                    "persistenceBackend": "in_memory"
                  }
                }
                """.trimIndent(),
            )

            val handle =
                UnifiedRagFactory.create(
                    ragId = RagId.CAUSAL_RAG,
                    configPath = configPath.toString(),
                )

            assertNotNull(handle.persistence)
            assertEquals("in_memory", handle.persistence.backendId)
            handle.close()
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `factory override disables unified persistence even when common config enables it`() {
        val root = Files.createTempDirectory("unified_defaults_override_test_")
        val configPath = root.resolve("common_rag.json")
        try {
            Files.writeString(
                configPath,
                """
                {
                  "unified": {
                    "useUnifiedPersistence": true,
                    "persistenceBackend": "in_memory"
                  }
                }
                """.trimIndent(),
            )

            val handle =
                UnifiedRagFactory.create(
                    ragId = RagId.GRAPH_RAG,
                    configPath = configPath.toString(),
                    overrides = mapOf("useUnifiedPersistence" to false),
                )

            assertEquals(null, handle.persistence)
            handle.close()
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `factory creates graph rag handle`() {
        val root = Files.createTempDirectory("unified_graphrag_test_")
        try {
            val handle =
                UnifiedRagFactory.create(
                    ragId = RagId.GRAPH_RAG,
                    overrides = mapOf("rootDir" to root.toString()),
                )

            assertEquals(RagId.GRAPH_RAG, handle.id)
            assertTrue(UnifiedMode.GLOBAL in handle.capabilities.supportedModes)
            assertTrue(handle.capabilities.supportsStreaming)
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `factory creates path rag handle without common config`() {
        val workingDir = Files.createTempDirectory("unified_pathrag_test_")
        try {
            val handle =
                UnifiedRagFactory.create(
                    ragId = RagId.PATH_RAG,
                    overrides =
                        mapOf(
                            "workingDir" to workingDir.toString(),
                            "chunkTokenSize" to 16,
                            "chunkOverlapTokenSize" to 4,
                        ),
                )

            assertEquals(RagId.PATH_RAG, handle.id)
            assertTrue(UnifiedMode.HYBRID in handle.capabilities.supportedModes)
            val snapshot = handle.rag.inspectGraph()
            assertTrue(snapshot.containsKey("metadata"))
            handle.rag.drop()
        } finally {
            workingDir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `factory attaches unified persistence session when enabled`() {
        val workingDir = Files.createTempDirectory("unified_pathrag_persistence_test_")
        val snapshotPath = workingDir.resolve("graph_snapshot.json")
        try {
            val handle =
                UnifiedRagFactory.create(
                    ragId = RagId.PATH_RAG,
                    overrides =
                        mapOf(
                            "workingDir" to workingDir.toString(),
                            "chunkTokenSize" to 16,
                            "chunkOverlapTokenSize" to 4,
                            "useUnifiedPersistence" to true,
                            "persistenceBackend" to "in_memory",
                        ),
                )

            assertNotNull(handle.persistence)
            handle.rag.drop()
            handle.rag.upsert(listOf("Alpha influences Beta.", "Beta affects Gamma."))
            handle.rag.saveGraph(snapshotPath.toString())

            val inspect = handle.rag.inspectGraph()
            val metadata = inspect["metadata"] as Map<*, *>
            assertEquals("in_memory", metadata["persistenceBackend"])
            assertTrue(
                Files.exists(Path.of("$snapshotPath.unified_spi").resolve("manifest.json")),
            )
            handle.close()
        } finally {
            workingDir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `factory resolves neo4j persistence backend for unified path rag`() {
        val workingDir = Files.createTempDirectory("unified_pathrag_neo4j_persistence_test_")
        try {
            val handle =
                UnifiedRagFactory.create(
                    ragId = RagId.PATH_RAG,
                    overrides =
                        mapOf(
                            "workingDir" to workingDir.toString(),
                            "chunkTokenSize" to 16,
                            "chunkOverlapTokenSize" to 4,
                            "useUnifiedPersistence" to true,
                            "persistenceBackend" to "neo4j",
                            "persistenceConfig" to
                                mapOf(
                                    "uri" to "bolt://localhost:7687",
                                    "username" to "neo4j",
                                    "password" to "test",
                                    "database" to "neo4j",
                                ),
                        ),
                )

            assertNotNull(handle.persistence)
            assertEquals("neo4j", handle.persistence.backendId)
            handle.close()
        } finally {
            workingDir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `factory resolves mongodb persistence backend for unified graph rag`() {
        val root = Files.createTempDirectory("unified_graphrag_mongo_persistence_test_")
        try {
            val handle =
                UnifiedRagFactory.create(
                    ragId = RagId.GRAPH_RAG,
                    overrides =
                        mapOf(
                            "rootDir" to root.toString(),
                            "useUnifiedPersistence" to true,
                            "persistenceBackend" to "mongodb",
                            "persistenceConfig" to
                                mapOf(
                                    "connectionString" to "mongodb://localhost:27017",
                                    "database" to "unified_rag_test",
                                ),
                        ),
                )

            assertNotNull(handle.persistence)
            assertEquals("mongodb", handle.persistence.backendId)
            handle.close()
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `factory enables lightrag unified spi retrieval and index for context mode`() {
        val root = Files.createTempDirectory("unified_lightrag_spi_test_")
        try {
            val handle =
                UnifiedRagFactory.create(
                    ragId = RagId.LIGHT_RAG,
                    overrides =
                        mapOf(
                            "workingDir" to root.toString(),
                            "useUnifiedPersistence" to true,
                            "persistenceBackend" to "in_memory",
                            "useUnifiedSpiForRetrievalAndIndex" to true,
                        ),
                )

            assertNotNull(handle.persistence)
            assertEquals("in_memory", handle.persistence.backendId)

            handle.persistence.vector("lightrag_vector", dimensions = 2, metric = "cosine").restore(
                VectorSnapshot(
                    dimensions = 2,
                    metric = "cosine",
                    records =
                        listOf(
                            VectorRecord(
                                id = "chunk:chunk-1",
                                vector = listOf(1.0, 0.0),
                                metadata =
                                    mapOf(
                                        "kind" to "chunk",
                                        "chunk_id" to "chunk-1",
                                        "content" to "Alpha influences Beta and Beta affects Gamma.",
                                    ),
                            ),
                        ),
                ),
            )
            handle.persistence.kv("lightrag_chunks").put(
                "chunk-1",
                mapOf(
                    "content" to "Alpha influences Beta and Beta affects Gamma.",
                    "file_path" to "memory://chunk-1",
                ),
            )

            val contextOnly =
                handle.rag.query(
                    "How does Alpha relate to Gamma?",
                    UnifiedQuery(
                        mode = UnifiedMode.HYBRID,
                        includeAnswer = false,
                        includeContext = true,
                        topK = 3,
                    ),
                )

            assertTrue(contextOnly.context.isNotEmpty())
            assertTrue(contextOnly.context.any { it.text.contains("alpha", ignoreCase = true) })
            handle.close()
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `factory enables pathrag unified spi retrieval and index for context mode`() {
        val root = Files.createTempDirectory("unified_pathrag_spi_test_")
        try {
            val handle =
                UnifiedRagFactory.create(
                    ragId = RagId.PATH_RAG,
                    overrides =
                        mapOf(
                            "workingDir" to root.toString(),
                            "chunkTokenSize" to 16,
                            "chunkOverlapTokenSize" to 4,
                            "useUnifiedPersistence" to true,
                            "persistenceBackend" to "in_memory",
                            "useUnifiedSpiForRetrievalAndIndex" to true,
                        ),
                )

            assertNotNull(handle.persistence)
            assertEquals("in_memory", handle.persistence.backendId)

            handle.persistence.kv("pathrag_chunks").put(
                "chunk-1",
                mapOf(
                    "content" to "Alpha influences Beta and Beta affects Gamma.",
                    "source_id" to "chunk-1",
                ),
            )

            val contextOnly =
                handle.rag.query(
                    "How does Alpha relate to Gamma?",
                    UnifiedQuery(
                        mode = UnifiedMode.HYBRID,
                        includeAnswer = false,
                        includeContext = true,
                        topK = 3,
                    ),
                )

            assertTrue(contextOnly.context.isNotEmpty())
            assertTrue(contextOnly.context.any { it.text.contains("alpha", ignoreCase = true) })

            val inspect = handle.rag.inspectGraph()
            val metadata = inspect["metadata"] as Map<*, *>
            assertEquals("in_memory", metadata["persistenceBackend"])

            handle.close()
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `factory enables causalrag unified spi retrieval and index`() {
        val root = Files.createTempDirectory("unified_causalrag_spi_test_")
        val checkpoint = root.resolve("causal_spi_checkpoint")
        try {
            val handle =
                UnifiedRagFactory.create(
                    ragId = RagId.CAUSAL_RAG,
                    overrides =
                        mapOf(
                            "useUnifiedPersistence" to true,
                            "persistenceBackend" to "in_memory",
                            "useUnifiedSpiForRetrievalAndIndex" to true,
                            "modelName" to "gpt-4o-mini",
                            "embeddingModel" to "all-MiniLM-L6-v2",
                            "embeddingApiKey" to "",
                        ),
                )

            assertNotNull(handle.persistence)
            assertEquals("in_memory", handle.persistence.backendId)

            handle.rag.upsert(listOf("Alpha causes Beta.", "Beta causes Gamma."))

            val contextOnly =
                handle.rag.query(
                    "How does Alpha relate to Gamma?",
                    UnifiedQuery(
                        mode = UnifiedMode.CAUSAL,
                        includeAnswer = false,
                        includeContext = true,
                        includeGraphPaths = true,
                        topK = 3,
                    ),
                )
            assertTrue(contextOnly.context.isNotEmpty())
            assertTrue(contextOnly.graphPaths.isNotEmpty())

            handle.rag.saveGraph(checkpoint.toString())
            handle.rag.drop()
            handle.rag.loadGraph(checkpoint.toString())

            val restored =
                handle.rag.query(
                    "How does Alpha relate to Gamma?",
                    UnifiedQuery(
                        mode = UnifiedMode.CAUSAL,
                        includeAnswer = false,
                        includeContext = true,
                        includeGraphPaths = true,
                        topK = 3,
                    ),
                )
            assertTrue(restored.context.isNotEmpty())
            assertTrue(restored.graphPaths.isNotEmpty())

            handle.close()
        } finally {
            root.toFile().deleteRecursively()
        }
    }
}
