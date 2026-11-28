package causalrag

import causalrag.generator.llm.LLMInterface
import causalrag.retriever.VectorStoreRetriever
import io.mockk.every
import io.mockk.mockkConstructor
import io.mockk.unmockkConstructor
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import shared.rag.spi.persistence.InMemoryPersistenceBackend
import shared.rag.spi.persistence.PersistenceSession
import shared.rag.spi.persistence.VectorIndexStore
import shared.rag.spi.persistence.VectorRecord
import shared.rag.spi.persistence.VectorSnapshot
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Tests CausalRAG initialization, indexing, querying, and persistence.
 */
class TestPipeline {
    private lateinit var tempDir: Path
    private lateinit var configPath: Path
    private lateinit var commonConfigPath: Path

    private val testDocs =
        listOf(
            "Climate change causes rising sea levels.",
            "Rising sea levels leads to coastal flooding.",
            "Coastal flooding causes population displacement.",
        )

    /**
     * Creates a temporary configuration for each test run.
     */
    @BeforeTest
    fun setUp() {
        tempDir = Files.createTempDirectory("causalrag-test")
        configPath = tempDir.resolve("causalrag-test-config.json")
        commonConfigPath = tempDir.resolve("common-rag-config.json")
        writeTestConfig(configPath)
        writeCommonConfig(commonConfigPath)
    }

    /**
     * Removes temporary test files and directories.
     */
    @AfterTest
    fun tearDown() {
        tempDir.toFile().deleteRecursively()
    }

    /**
     * Verifies that CausalRAG initializes.
     */
    @Test
    fun testPipelineInit() {
        val rag = CausalRAG(configPath = configPath.toString())
        assertNotNull(rag)
    }

    @Test
    fun testPipelineInitWithCommonConfig() {
        val rag = CausalRAG(configPath = commonConfigPath.toString())
        assertNotNull(rag)
    }

    /**
     * Verifies that indexing populates the causal graph.
     */
    @Test
    fun testDocumentIndexing() {
        val rag = CausalRAG(configPath = configPath.toString())
        rag.upsert(testDocs)
        val snapshot = rag.inspectGraph()
        val metadata = snapshot["metadata"] as? Map<*, *> ?: emptyMap<Any?, Any?>()
        val nodeCount = metadata["nodeCount"] as? Int ?: 0
        assertTrue(nodeCount > 0)
    }

    /**
     * Verifies that retrieval APIs return context and causal paths.
     */
    @Test
    fun testQueryExecution() {
        val rag = CausalRAG(configPath = configPath.toString())
        rag.upsert(testDocs)
        val contextResult = rag.query("What causes coastal flooding?", QueryParam(topK = 3, onlyNeedContext = true))
        assertTrue(contextResult.context.isNotEmpty())
        val pathResult = rag.query("What causes coastal flooding?", QueryParam(maxPaths = 3, onlyNeedCausalPaths = true))
        assertTrue(pathResult.causalPaths.isNotEmpty())
    }

    /**
     * Verifies that saved pipeline artifacts can be loaded into a fresh instance.
     */
    @Test
    fun testSaveAndLoad() {
        val rag1 = CausalRAG(configPath = configPath.toString())
        rag1.upsert(testDocs)

        val saveDir = tempDir.resolve("causalrag_index")
        Files.createDirectories(saveDir)
        rag1.saveGraph(saveDir.toString())

        val rag2 = CausalRAG(configPath = configPath.toString())
        rag2.loadGraph(saveDir.toString())

        val contextResult = rag2.query("What is climate change?", QueryParam(topK = 3, onlyNeedContext = true))
        assertTrue(contextResult.context.isNotEmpty())
        val pathResult = rag2.query("What is climate change?", QueryParam(maxPaths = 3, onlyNeedCausalPaths = true))
        assertTrue(pathResult.causalPaths.isNotEmpty())
    }

    @Test
    fun failedSnapshotSaveKeepsTheIndexedEvidence() {
        val rag = CausalRAG(configPath = configPath.toString())
        rag.upsert(testDocs.first())
        val occupied = Files.writeString(tempDir.resolve("occupied"), "original")
        val failure = assertFailsWith<IllegalStateException> { rag.saveGraph(occupied.toString()) }
        assertTrue(failure.message.orEmpty().contains("Failed to save"))
        assertEquals("original", Files.readString(occupied))
        assertEquals(listOf(testDocs.first()), rag.retrieveContext("climate change"))
    }

    @Test
    fun testRepeatedUpsertPreservesPreviouslyIndexedPassages() {
        val rag = CausalRAG(configPath = configPath.toString())
        testDocs.forEach(rag::upsert)
        val before = rag.inspectGraph()
        rag.upsert(testDocs)

        assertEquals(before, rag.inspectGraph())
        assertEquals(testDocs.toSet(), rag.retrieveContext("coastal flooding", topK = 10).toSet())
    }

    @Test
    fun vectorOnlyUpsertDoesNotSuppressSubsequentGraphExtraction() =
        runBlocking {
            val rag = CausalRAG(configPath = configPath.toString())
            val passage = testDocs.first()
            rag.upsert(mapOf("first" to mapOf("content" to passage)))
            assertEquals(0, (rag.inspectGraph()["metadata"] as Map<*, *>)["edgeCount"])

            rag.aupsert(passage)
            val indexed = rag.inspectGraph()
            assertTrue(((indexed["metadata"] as Map<*, *>)["edgeCount"] as Int) > 0)
            assertEquals(listOf(passage), rag.retrieveContext("climate change"))
            rag.aupsert(passage)
            assertEquals(indexed, rag.inspectGraph())
        }

    @Test
    fun failedSpiPublicationRestoresGraphVectorsAndMetadataBeforeRetry() =
        runBlocking {
            InMemoryPersistenceBackend().open(emptyMap()).use { backing ->
                val restoreFailures = ArrayDeque<IllegalStateException>()
                val session =
                    object : PersistenceSession by backing {
                        override fun vector(
                            namespace: String,
                            dimensions: Int?,
                            metric: String,
                        ): VectorIndexStore {
                            val store = backing.vector(namespace, dimensions, metric)
                            return object : VectorIndexStore by store {
                                override fun restore(snapshot: VectorSnapshot) {
                                    restoreFailures.removeFirstOrNull()?.let { throw it }
                                    store.restore(snapshot)
                                }
                            }
                        }
                    }
                val rag =
                    CausalRAG(configPath = configPath.toString(), persistenceSession = session, useUnifiedSpiForRetrievalAndIndex = true)
                rag.aupsert(testDocs.first())
                val graph = backing.graph("causalrag_graph").snapshot()
                val vectors = backing.vector("causalrag_vector").snapshot()
                val metadata = backing.kv("causalrag_metadata").snapshot()
                val writeFailure = IllegalStateException("injected SPI vector publication failure")
                val rollbackFailure = IllegalStateException("injected SPI rollback failure")
                for (rollback in listOf(null, rollbackFailure, writeFailure)) {
                    restoreFailures.add(writeFailure)
                    rollback?.let(restoreFailures::add)
                    val error = assertFailsWith<IllegalStateException> { rag.aupsert(testDocs[1]) }
                    assertTrue(error === writeFailure)
                    if (rollback === rollbackFailure) assertTrue(rollbackFailure in error.suppressed)
                    assertEquals(graph, backing.graph("causalrag_graph").snapshot())
                    assertEquals(vectors, backing.vector("causalrag_vector").snapshot())
                    assertEquals(metadata, backing.kv("causalrag_metadata").snapshot())
                    assertEquals(listOf(testDocs.first()), rag.retrieveContext("climate change", 10))
                }

                rag.aupsert(testDocs[1])
                assertEquals(testDocs.take(2).toSet(), rag.retrieveContext("coastal flooding", 10).toSet())
                val restarted =
                    CausalRAG(configPath = configPath.toString(), persistenceSession = backing, useUnifiedSpiForRetrievalAndIndex = true)
                assertEquals(testDocs.take(2).toSet(), restarted.retrieveContext("coastal flooding", 10).toSet())
            }
        }

    @Test
    fun unifiedReplacementPublishesNewEvidenceAndDropClearsTheBackend() =
        runBlocking {
            InMemoryPersistenceBackend().open(emptyMap()).use { session ->
                val rag =
                    CausalRAG(configPath = configPath.toString(), persistenceSession = session, useUnifiedSpiForRetrievalAndIndex = true)
                rag.aupsert(testDocs)
                rag.areindex(listOf(testDocs[1]))
                assertEquals(listOf(testDocs[1]), rag.retrieveContext("coastal flooding", 10))
                assertTrue(((rag.ainspectGraph()["metadata"] as Map<*, *>)["edgeCount"] as Int) > 0)
                rag.areindex(emptyList())
                assertTrue(rag.retrieveContext("coastal flooding", 10).isEmpty())
                rag.aupsert(testDocs.first())
                rag.adrop()
                assertTrue(
                    session
                        .vector("causalrag_vector")
                        .snapshot()
                        .records
                        .isEmpty(),
                )
                assertTrue(
                    session
                        .graph("causalrag_graph")
                        .snapshot()
                        .edges
                        .isEmpty(),
                )
                assertTrue(
                    session
                        .kv("causalrag_metadata")
                        .snapshot()
                        .entries
                        .isEmpty(),
                )
            }
        }

    @Test
    fun shortPassageDoesNotRequireCausalExtraction() =
        runBlocking {
            val rag = CausalRAG(configPath = configPath.toString())
            rag.aupsert("rainfall")
            assertTrue(rag.lastIndexingReport.failures.isEmpty())
            assertEquals(listOf("rainfall"), rag.retrieveContext("rainfall"))
            assertEquals(0, (rag.inspectGraph()["metadata"] as Map<*, *>)["edgeCount"])
        }

    @Test
    fun bootstrapReembedsLegacySpiRecordsAndRejectsAnIncompatibleIdentity() =
        runBlocking {
            InMemoryPersistenceBackend().open(emptyMap()).use { session ->
                val metadata = session.kv("causalrag_metadata")
                val encoder = VectorStoreRetriever("text-embedding-3-small", embeddingApiKey = "")
                val dimension = assertNotNull(encoder.encodeForQuery(testDocs.first())).size
                val legacy = List(dimension) { 0.0 }
                session.vector("causalrag_vector", dimensions = dimension).upsert(
                    VectorRecord("legacy", legacy, mapOf("content" to testDocs.first())),
                )
                val rag =
                    CausalRAG(configPath = configPath.toString(), persistenceSession = session, useUnifiedSpiForRetrievalAndIndex = true)
                val migrated =
                    session
                        .vector("causalrag_vector")
                        .snapshot()
                        .records
                        .single()
                assertEquals("legacy", migrated.id)
                assertEquals(dimension, migrated.vector.size)
                assertTrue(migrated.vector != legacy)
                assertTrue(metadata.get("embeddingIdentity") is String)
                assertEquals(listOf(testDocs.first()), rag.retrieveContext("climate change", 10))

                metadata.put("embeddingIdentity", "different-provider-and-model")
                assertFailsWith<IllegalStateException> {
                    CausalRAG(configPath = configPath.toString(), persistenceSession = session, useUnifiedSpiForRetrievalAndIndex = true)
                }
                assertEquals(
                    migrated,
                    session
                        .vector("causalrag_vector")
                        .snapshot()
                        .records
                        .single(),
                )
            }
        }

    @Test
    fun vectorMutationsPublishOnlyWhenUnifiedPersistenceIsEnabled() =
        runBlocking {
            InMemoryPersistenceBackend().open(emptyMap()).use { session ->
                val local = CausalRAG(configPath = configPath.toString(), persistenceSession = session)
                val unified =
                    CausalRAG(configPath = configPath.toString(), persistenceSession = session, useUnifiedSpiForRetrievalAndIndex = true)
                local.upsert(mapOf("local" to mapOf("content" to testDocs[0])))
                val store = session.vector("causalrag_vector")
                assertTrue(store.snapshot().records.isEmpty())
                unified.upsert(mapOf("shared" to mapOf("content" to testDocs[1], "entity_name" to "coast")))
                assertEquals(listOf("shared"), store.snapshot().records.map { it.id })
                unified.deleteEntity("coast")
                assertTrue(store.snapshot().records.isEmpty())
                assertEquals(listOf(testDocs[0]), local.retrieveContext("climate change", 5))
            }
        }

    @Test
    fun retrievesUnifiedEvidenceFromLegacyContentAndRecordIds() =
        runBlocking {
            InMemoryPersistenceBackend().open(emptyMap()).use { session ->
                val rag =
                    CausalRAG(configPath = configPath.toString(), persistenceSession = session, useUnifiedSpiForRetrievalAndIndex = true)
                val query = "legacy climate evidence"
                val encoder = VectorStoreRetriever("text-embedding-3-small", embeddingApiKey = "")
                val vector = assertNotNull(encoder.encodeForQuery(query)).toList()
                val store = session.vector("causalrag_vector", dimensions = vector.size, metric = "cosine")
                store.upsert(VectorRecord("content-only", vector, mapOf("content" to "Historical climate report.")))
                store.upsert(VectorRecord("Climate bulletin.", vector))
                assertEquals(
                    setOf("Historical climate report.", "Climate bulletin."),
                    rag.retrieveContext(query, topK = 5).toSet(),
                )
            }
        }

    @Test
    fun usesBm25EvidenceWhenUnifiedVectorSearchFails() =
        runBlocking {
            InMemoryPersistenceBackend().open(emptyMap()).use { backing ->
                val session =
                    object : PersistenceSession by backing {
                        override fun vector(
                            namespace: String,
                            dimensions: Int?,
                            metric: String,
                        ): VectorIndexStore {
                            val store = backing.vector(namespace, dimensions, metric)
                            return object : VectorIndexStore by store {
                                override fun query(
                                    vector: List<Double>,
                                    topK: Int,
                                ): List<VectorRecord> = error("injected vector query failure")
                            }
                        }
                    }
                val rag =
                    CausalRAG(configPath = configPath.toString(), persistenceSession = session, useUnifiedSpiForRetrievalAndIndex = true)
                rag.aupsert(testDocs.first())
                assertEquals(listOf(testDocs.first()), rag.retrieveContext("climate change", 10))
            }
        }

    @Test
    fun directVectorStoragePreservesMetadataAndDeletesMatchingRows() =
        runBlocking {
            val retriever = VectorStoreRetriever("text-embedding-3-small", embeddingApiKey = "")
            val store: shared.rag.CommonVectorStorage<Any?> = CausalRAGVectorStorage(retriever)
            store.upsert(
                mapOf(
                    "entity" to mapOf("content" to testDocs[0], "entity_id" to "climate"),
                    "edge" to mapOf("content" to testDocs[1], "target_id" to "coast"),
                    "other" to mapOf("content" to testDocs[2]),
                ),
            )
            store.upsert(mapOf("other" to mapOf("source" to "report")))
            assertEquals("report", store.query("coastal flooding", 3).single { it["id"] == "other" }["source"])
            store.deleteEntity("climate")
            store.deleteEntityRelation("coast")
            assertEquals(listOf("other"), store.query("coastal flooding", 3).map { it["id"] })
        }

    @Test
    fun vectorDeletionsRefreshRetrievalAndReplacementClearsPriorPassages() =
        runBlocking {
            val rag = CausalRAG(configPath = configPath.toString())
            rag.upsert(
                mapOf(
                    "entity" to mapOf("content" to testDocs[0], "entity_name" to "climate"),
                    "relation" to mapOf("content" to testDocs[1], "src_id" to "climate", "tgt_id" to "coast"),
                    "keep" to mapOf("content" to testDocs[2]),
                ),
            )
            assertEquals(3, rag.query("coastal flooding", 5).size)
            rag.deleteEntity("climate")
            rag.deleteEntityRelation("climate")
            assertEquals(listOf(testDocs[2]), rag.retrieveContext("coastal flooding", 5))

            rag.areindex(listOf(testDocs.first()))
            assertEquals(listOf(testDocs.first()), rag.retrieveContext("coastal flooding", 5))
            rag.adrop()
            assertTrue(rag.retrieveContext("coastal flooding", 5).isEmpty())
        }

    @Test
    fun testCorruptSnapshotPreservesLiveIndex() {
        val rag = CausalRAG(configPath = configPath.toString())
        rag.upsert(testDocs)
        val graph = rag.inspectGraph()
        val query = "coastal flooding"
        val context = rag.retrieveContext(query, topK = 5)
        val replacement = CausalRAG(configPath = configPath.toString())
        replacement.upsert("Heavy rainfall causes river flooding.")
        val snapshot = tempDir.resolve("corrupt-snapshot")
        replacement.saveGraph(snapshot.toString())
        Files.writeString(snapshot.resolve("vector_cache/vectors.json"), "[[1.0], [1.0, 0.0]]")

        assertFailsWith<IllegalStateException> { rag.loadGraph(snapshot.toString()) }
        assertEquals(graph, rag.inspectGraph())
        assertEquals(context, rag.retrieveContext(query, topK = 5))
    }

    @Test
    fun testAnswerAndContextOnlyUseTheSameEvidence() {
        mockkConstructor(LLMInterface::class)
        try {
            val prompts = mutableListOf<String>()
            every { anyConstructed<LLMInterface>().generate(any(), any(), any(), any(), any(), any()) } answers {
                prompts.add(firstArg())
                "[]"
            }
            val rag = CausalRAG(configPath = configPath.toString())
            rag.upsert(testDocs)
            val query = "What causes coastal flooding?"
            val params = QueryParam(topK = 3, maxPaths = 1)
            val allPaths = rag.retrieveCausalPaths(query, maxPaths = 3)
            assertTrue(allPaths.size > 1)
            val answer = rag.query(query, params)
            val evidence = rag.query(query, params.copy(onlyNeedContext = true, onlyNeedCausalPaths = true))

            assertEquals(answer.context, evidence.context)
            assertEquals(answer.causalPaths, evidence.causalPaths)
            assertEquals(1, answer.causalPaths.size)
            assertTrue(answer.causalPaths.single().joinToString(" → ") in prompts.last())
            for (path in allPaths - answer.causalPaths.toSet()) {
                assertTrue(path.joinToString(" → ") !in prompts.last())
            }
            assertEquals("[]", rag.run(query, topK = 3))
        } finally {
            unmockkConstructor(LLMInterface::class)
        }
    }

    private fun writeTestConfig(path: Path) {
        val json =
            JsonObject(
                mapOf(
                    "modelName" to JsonPrimitive("gpt-4o-mini"),
                    "embeddingModel" to JsonPrimitive("text-embedding-3-small"),
                    "llmProvider" to JsonPrimitive("mock"),
                    "llmApiKey" to JsonPrimitive(""),
                    "llmBaseUrl" to JsonNull,
                    "embeddingApiKey" to JsonPrimitive(""),
                    "graphPath" to JsonNull,
                    "indexPath" to JsonNull,
                    "templateStyle" to JsonPrimitive("detailed"),
                ),
            )
        val content =
            Json { prettyPrint = true }.encodeToString(
                JsonElement.serializer(),
                json,
            )
        Files.writeString(path, content)
    }

    private fun writeCommonConfig(path: Path) {
        val json =
            JsonObject(
                mapOf(
                    "shared" to
                        JsonObject(
                            mapOf(
                                "llmProvider" to JsonPrimitive("mock"),
                                "modelName" to JsonPrimitive("gpt-4o-mini"),
                                "embeddingModel" to JsonPrimitive("text-embedding-3-small"),
                                "llmApiKey" to JsonPrimitive(""),
                                "embeddingApiKey" to JsonPrimitive(""),
                            ),
                        ),
                    "causalrag" to
                        JsonObject(
                            mapOf(
                                "templateStyle" to JsonPrimitive("detailed"),
                                "graphPath" to JsonNull,
                                "indexPath" to JsonNull,
                            ),
                        ),
                ),
            )
        val content =
            Json { prettyPrint = true }.encodeToString(
                JsonElement.serializer(),
                json,
            )
        Files.writeString(path, content)
    }
}
