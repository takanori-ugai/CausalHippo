package pathrag

import io.mockk.coEvery
import io.mockk.mockkStatic
import io.mockk.unmockkStatic
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import pathrag.base.QueryParam
import pathrag.storage.NanoVectorDBStorage
import pathrag.utils.EmbeddingFunc
import pathrag.utils.computeMdHashId
import pathrag.utils.relationshipVectorId
import shared.rag.spi.persistence.GraphSnapshot
import shared.rag.spi.persistence.GraphStore
import shared.rag.spi.persistence.InMemoryPersistenceBackend
import shared.rag.spi.persistence.PersistenceSession
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class IndexPublicationTest {
    private val document = "The evidence passage about Alpha and Beta."
    private val extraction =
        """{"entities":[],"relationships":[{"src_id":"ALPHA","tgt_id":"BETA","description":"related","keywords":"link","weight":1}]}"""

    @Test
    fun unifiedContextTraversesPublishedGraphAndHonorsRetrievalMode() =
        exercise { directory ->
            InMemoryPersistenceBackend().open(emptyMap()).use { session ->
                PathRAG(
                    workingDir = directory,
                    persistenceSession = session,
                    useUnifiedSpiForRetrievalAndIndex = true,
                    lowLevelKeywords = listOf("ALPHA", "BETA"),
                    highLevelKeywords = listOf("link"),
                ).use { rag ->
                    rag.ainsert(document)
                    val local = rag.aquery("query", QueryParam(mode = "local", onlyNeedContext = true))
                    assertTrue("through edge(" in local && document in local)
                    val global = rag.aquery("query", QueryParam(mode = "global", onlyNeedContext = true))
                    assertTrue("high-level relationship information" in global && document in global)
                    assertFalse("through edge(" in global)
                }
            }
        }

    @Test
    fun failedChunkEmbeddingDoesNotPublishGraphEvidence() =
        exercise { directory ->
            PathRAG(workingDir = directory).use { rag ->
                coEvery { pathrag.llm.openAiEmbedding(match { document in it }, any()) } throws IllegalStateException("embedding failed")
                assertFailsWith<IllegalStateException> { rag.ainsert(document) }
                assertTrue(rag.graph().nodes().isEmpty())
                assertTrue(rag.graph().edges().isEmpty())
            }
        }

    @Test
    fun retryPublishesCommittedDocumentAfterSpiWriteFails() =
        exercise { directory ->
            InMemoryPersistenceBackend().open(emptyMap()).use { backed ->
                var fail = true
                val session =
                    object : PersistenceSession by backed {
                        override fun graph(namespace: String): GraphStore {
                            val storage = backed.graph(namespace)
                            return object : GraphStore by storage {
                                override fun restore(snapshot: GraphSnapshot) {
                                    if (fail) {
                                        fail = false
                                        error("injected publication failure")
                                    }
                                    storage.restore(snapshot)
                                }
                            }
                        }
                    }
                PathRAG(workingDir = directory, persistenceSession = session, useUnifiedSpiForRetrievalAndIndex = true).use { rag ->
                    assertFailsWith<IllegalStateException> { rag.ainsert(document) }
                    rag.ainsert(document)
                    assertTrue(backed.graph("pathrag_graph").edges().isNotEmpty())
                    assertTrue(document in rag.aquery("evidence", QueryParam(onlyNeedContext = true)))
                }
            }
        }

    @Test
    fun shortBatchDoesNotPartiallyWriteVectorsAndCanBeRetried() =
        runBlocking {
            var short = true
            val embed = EmbeddingFunc(2, 100, { texts -> List(if (short) 1 else texts.size) { doubleArrayOf(1.0, 0.0) } })
            val store = NanoVectorDBStorage("test", emptyMap(), embed)
            val rows = mapOf("a" to mapOf("content" to "first"), "b" to mapOf("content" to "second"))
            assertFailsWith<IllegalStateException> { store.upsert(rows) }
            assertTrue(store.query("query", 2).isEmpty())
            short = false
            store.upsert(rows)
            assertEquals(setOf("first", "second"), store.query("query", 2).map { it["content"] }.toSet())
        }

    @Test
    fun collidingEndpointConcatenationsRetainBothRelationships() =
        exercise { directory ->
            coEvery { pathrag.llm.openAiComplete(any(), any(), any(), any(), any(), any(), any(), any(), any()) } returns
                """{"entities":[],"relationships":[
                {"src_id":"A","tgt_id":"BC","description":"first evidence","keywords":"link","weight":1},
                {"src_id":"AB","tgt_id":"C","description":"second evidence","keywords":"link","weight":1}
                ]}"""
            PathRAG(workingDir = directory).use { rag ->
                rag.ainsert(document)
                coEvery { pathrag.llm.openAiComplete(any(), any(), any(), any(), any(), any(), any(), any(), any()) } returns
                    """{"high_level_keywords":["link"],"low_level_keywords":[]}"""
                val query = QueryParam(mode = "global", onlyNeedContext = true)
                val original = rag.aquery("link", query)
                assertTrue("first evidence" in original)
                assertTrue("second evidence" in original)
                rag.adeleteEdge("A", "BC")
                val retained = rag.aquery("link", query)
                assertFalse("first evidence" in retained)
                assertTrue("second evidence" in retained)
            }
        }

    @Test
    fun retryRepairsRelationshipVectorEvenWhenExtractionReturnsNoEdges() =
        exercise { directory ->
            PathRAG(workingDir = directory).use { rag ->
                var fail = true
                coEvery { pathrag.llm.openAiEmbedding(match { "linkALPHABETArelated" in it }, any()) } answers {
                    check(!fail) { "injected relationship embedding failure" }
                    firstArg<List<String>>().map { DoubleArray(1536) { 0.02 } }
                }
                assertFailsWith<IllegalStateException> { rag.ainsert(document) }
                assertTrue(rag.graph().hasEdge("ALPHA", "BETA"))
                fail = false
                coEvery { pathrag.llm.openAiComplete(any(), any(), any(), any(), any(), any(), any(), any(), any()) } returns
                    """{"entities":[],"relationships":[]}"""
                rag.ainsert(document)
                coEvery { pathrag.llm.openAiComplete(any(), any(), any(), any(), any(), any(), any(), any(), any()) } returns
                    """{"high_level_keywords":["link"],"low_level_keywords":[]}"""
                assertTrue(document in rag.aquery("link", QueryParam(mode = "global", onlyNeedContext = true)))
            }
        }

    @Test
    fun documentExtractionRunsForExistingCustomChunksAndAfterGraphDrop() =
        exercise { directory ->
            PathRAG(workingDir = directory).use { rag ->
                rag.ainsertCustomKg(PathRAG.CustomKgPayload(chunks = listOf(PathRAG.CustomKgChunk(document))))
                assertTrue(rag.graph().edges().isEmpty())
                rag.ainsert(document)
                assertTrue(rag.graph().hasEdge("ALPHA", "BETA"))
                rag.adropGraph()
                assertTrue(rag.graph().edges().isEmpty())
                rag.ainsert(document)
                assertTrue(rag.graph().hasEdge("ALPHA", "BETA"))
            }
        }

    @Test
    fun retriesUseCommittedSnapshotAndLocalAnswersSurviveUnifiedEmbeddingFailure() =
        exercise { directory ->
            val secondDocument = "Evidence for Gamma and Delta."
            InMemoryPersistenceBackend().open(emptyMap()).use { session ->
                var failUnified = true
                var alphaRelationBatches = 0
                coEvery { pathrag.llm.openAiEmbedding(any(), any()) } answers {
                    val texts = firstArg<List<String>>()
                    if ("linkALPHABETArelated" in texts) {
                        alphaRelationBatches++
                        check(!failUnified || alphaRelationBatches == 1)
                    }
                    check("linkGAMMADELTAsecond" !in texts)
                    texts.map { DoubleArray(1536) { 0.02 } }
                }
                PathRAG(workingDir = directory, persistenceSession = session, useUnifiedSpiForRetrievalAndIndex = true).use { rag ->
                    assertFailsWith<IllegalStateException> { rag.ainsert(document) }
                    assertTrue(session.graph("pathrag_graph").edges().isEmpty())
                    coEvery { pathrag.llm.openAiComplete(any(), any(), any(), any(), any(), any(), any(), any(), any()) } returns
                        """{"entities":[],"relationships":[{"src_id":"GAMMA","tgt_id":"DELTA","description":"second","keywords":"link"}]}"""
                    assertFailsWith<IllegalStateException> { rag.ainsert(secondDocument) }
                    assertTrue(rag.graph().hasEdge("GAMMA", "DELTA"))

                    coEvery { pathrag.llm.openAiComplete(any(), any(), any(), any(), any(), any(), any(), any(), any()) } returns
                        """{"high_level_keywords":["link"],"low_level_keywords":[]}"""
                    coEvery {
                        pathrag.llm.openAiComplete(
                            any(),
                            any(),
                            match { document in it },
                            any(),
                            any(),
                            any(),
                            any(),
                            any(),
                            any(),
                        )
                    } returns
                        "local answer"
                    assertEquals("local answer", rag.aquery("link", QueryParam(mode = "global")))
                    assertFailsWith<IllegalStateException> { rag.aquery("link", QueryParam(onlyNeedContext = true)) }

                    failUnified = false
                    rag.ainsert(document)
                    assertEquals(
                        setOf("ALPHA" to "BETA"),
                        session
                            .graph("pathrag_graph")
                            .edges()
                            .map { it.source to it.target }
                            .toSet(),
                    )
                    val checkpoint = "$directory/graph.json"
                    rag.asaveGraph(checkpoint)
                    rag.aclear()
                    rag.aloadGraph(checkpoint)
                    assertTrue(rag.graph().hasEdge("ALPHA", "BETA"))
                    assertFalse(rag.graph().hasEdge("GAMMA", "DELTA"))
                }
            }
        }

    @Test
    fun localCheckpointReloadsGraphAndRelationshipVectors() =
        exercise { directory ->
            PathRAG(workingDir = directory).use { rag ->
                rag.ainsert(document)
                val checkpoint = "$directory/local-graph.json"
                rag.asaveGraph(checkpoint)
                rag.adropGraph()
                assertFalse(rag.graph().hasEdge("ALPHA", "BETA"))
                withTimeout(5_000) { rag.aloadGraph(checkpoint) }
                assertTrue(rag.graph().hasEdge("ALPHA", "BETA"))
                coEvery { pathrag.llm.openAiComplete(any(), any(), any(), any(), any(), any(), any(), any(), any()) } returns
                    """{"high_level_keywords":["link"],"low_level_keywords":[]}"""
                val context = rag.aquery("link", QueryParam(mode = "global", onlyNeedContext = true))
                assertTrue(document in context)
            }
        }

    @Test
    fun clearingCancelsPendingPublication() =
        exercise { directory ->
            InMemoryPersistenceBackend().open(emptyMap()).use { session ->
                var alphaRelationBatches = 0
                coEvery { pathrag.llm.openAiEmbedding(match { "linkALPHABETArelated" in it }, any()) } answers {
                    check(++alphaRelationBatches == 1) { "injected unified embedding failure" }
                    firstArg<List<String>>().map { DoubleArray(1536) { 0.02 } }
                }
                PathRAG(workingDir = directory, persistenceSession = session, useUnifiedSpiForRetrievalAndIndex = true).use { rag ->
                    assertFailsWith<IllegalStateException> { rag.ainsert(document) }
                    rag.aclear()
                    rag.aquery("link", QueryParam(onlyNeedContext = true))
                    assertTrue(session.graph("pathrag_graph").edges().isEmpty())
                }
            }
        }

    @Test
    fun answerGenerationDoesNotHoldTheIndexLock() =
        exercise { directory ->
            PathRAG(workingDir = directory).use { rag ->
                rag.ainsert(document)
                val answerStarted = CompletableDeferred<Unit>()
                val releaseAnswer = CompletableDeferred<Unit>()
                coEvery { pathrag.llm.openAiComplete(any(), any(), any(), any(), any(), any(), any(), any(), any()) } returns
                    """{"high_level_keywords":["link"],"low_level_keywords":[]}"""
                coEvery {
                    pathrag.llm.openAiComplete(
                        any(),
                        any(),
                        match { document in it },
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                    )
                } coAnswers
                    {
                        answerStarted.complete(Unit)
                        releaseAnswer.await()
                        "answer from the captured context"
                    }
                withTimeout(10_000) {
                    val answer = async { rag.aquery("link", QueryParam(mode = "global")) }
                    answerStarted.await()
                    rag.adropGraph()
                    releaseAnswer.complete(Unit)
                    assertEquals("answer from the captured context", answer.await())
                }
            }
        }

    @Test
    fun legacyRelationshipReplacementChecksOrderedEndpoints() =
        runBlocking {
            val embedding = EmbeddingFunc(2, 100, { texts -> texts.map { doubleArrayOf(1.0, 0.0) } })
            val store = NanoVectorDBStorage("test", emptyMap(), embedding, metaFields = setOf("src_id", "tgt_id"))
            val legacyId = computeMdHashId("ABC", prefix = "rel-")

            fun row(
                src: String,
                tgt: String,
                content: String,
            ) = mapOf("src_id" to src, "tgt_id" to tgt, "content" to content)
            store.upsert(mapOf(legacyId to row("A", "BC", "legacy")))
            val legacyWithCurrentEncoding = computeMdHashId("1:ABC", prefix = "rel-")
            store.upsert(mapOf(legacyWithCurrentEncoding to row("1:A", "BC", "different legacy")))
            store.upsert(mapOf(relationshipVectorId("AB", "C") to row("AB", "C", "other")))
            assertEquals(setOf("legacy", "other", "different legacy"), store.query("test", 10).map { it["content"] }.toSet())
            store.upsert(mapOf(relationshipVectorId("A", "BC") to row("A", "BC", "updated")))
            assertEquals(setOf("updated", "other", "different legacy"), store.query("test", 10).map { it["content"] }.toSet())
        }

    private fun exercise(test: suspend (String) -> Unit) =
        runBlocking {
            val dir = Files.createTempDirectory("pathrag-publication")
            mockkStatic("pathrag.llm.LlmKt")
            try {
                coEvery { pathrag.llm.openAiComplete(any(), any(), any(), any(), any(), any(), any(), any(), any()) } returns extraction
                coEvery { pathrag.llm.openAiEmbedding(any(), any()) } answers {
                    firstArg<List<String>>().map { DoubleArray(1536) { 0.02 } }
                }
                test(dir.toString())
            } finally {
                unmockkStatic("pathrag.llm.LlmKt")
                dir.toFile().deleteRecursively()
            }
        }
}
