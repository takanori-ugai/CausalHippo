package com.microsoft.graphrag

import com.microsoft.graphrag.index.CommunityReport
import com.microsoft.graphrag.index.DocumentChunker
import com.microsoft.graphrag.index.GraphRagConfig
import com.microsoft.graphrag.index.LocalVectorStore
import com.microsoft.graphrag.index.ParquetWriterHelper
import com.microsoft.graphrag.index.SimplePipeline
import com.microsoft.graphrag.index.TextEmbedding
import com.microsoft.graphrag.index.WorkflowResult
import com.microsoft.graphrag.index.defaultPipeline
import com.microsoft.graphrag.query.BasicQueryEngine
import com.microsoft.graphrag.query.DriftSearchEngine
import com.microsoft.graphrag.query.GlobalSearchEngine
import com.microsoft.graphrag.query.QueryCallbacks
import com.microsoft.graphrag.query.QueryResult
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import io.mockk.Called
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkStatic
import io.mockk.verify
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import shared.chunking.countTokensWithJTokKit
import shared.rag.unified.GraphRagUnifiedAdapter
import shared.rag.unified.UnifiedQuery
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class GraphRagIntegrityTest {
    @Test
    fun overlappingSaveAndEmptyRestorePreserveIndex() =
        withPipeline { root ->
            val rag = GraphRAG(root)
            rag.upsert("Original corpus")
            val output = root.resolve("output")
            assertFails { rag.saveGraph(output.toString()) }
            assertTrue(Files.isRegularFile(output.resolve("text_units.parquet")))
            val empty = Files.createDirectory(root.resolve("empty"))
            assertFails { rag.loadGraph(empty.toString()) }
            assertTrue(Files.isRegularFile(output.resolve("text_units.parquet")))
        }

    @Test
    fun subsequentUpsertUsesRestoredCorpus() =
        withPipeline { root ->
            val source = GraphRAG(root.resolve("source"))
            source.upsert("Original corpus")
            val snapshot = root.resolve("snapshot")
            source.saveGraph(snapshot.toString())
            val targetRoot = root.resolve("target")
            val target = GraphRAG(targetRoot)
            target.upsert("Unrelated corpus")
            target.loadGraph(snapshot.toString())
            target.upsert("Additional corpus")
            assertEquals(setOf("Original corpus", "Additional corpus"), inputTexts(targetRoot.resolve("input")))
        }

    @Test
    fun failedIndexDoesNotPublishOutputOrInput() =
        withPipeline { root ->
            val rag = GraphRAG(root)
            rag.upsert("Original corpus")
            assertFails { rag.upsert("Fail indexing") }
            assertEquals("Original corpus", Files.readString(root.resolve("output/corpus.txt")))
            assertEquals(setOf("Original corpus"), inputTexts(root.resolve("input")))
        }

    @Test
    fun constructorEmbeddingModelReachesIndexPipeline() =
        withPipeline { root ->
            GraphRAG(root, defaultEmbeddingModelName = "custom-model").upsert("Original corpus")
            assertEquals("custom-model", Files.readString(root.resolve("output/model.txt")))
        }

    @Test
    fun rejectsQueryEmbeddingModelDifferentFromStoredModel() =
        withPipeline { root ->
            val rag = GraphRAG(root, defaultEmbeddingModelName = "indexed-model")
            rag.upsert("Original corpus")
            val failure =
                assertFailsWith<IllegalArgumentException> {
                    rag.query("query", QueryParam(mode = "basic", embeddingModelName = "other-model"))
                }
            assertTrue("indexed-model" in failure.message.orEmpty())
        }

    @Test
    fun refusesUpsertAfterLoadingLegacySnapshotWithoutCorpus() =
        withPipeline { root ->
            val sourceRoot = root.resolve("source")
            GraphRAG(sourceRoot).upsert("Original corpus")
            val targetRoot = root.resolve("target")
            val target = GraphRAG(targetRoot)
            target.loadGraph(sourceRoot.resolve("output").toString())
            assertFailsWith<IllegalStateException> { target.upsert("Additional corpus") }
            assertEquals("Original corpus", Files.readString(targetRoot.resolve("output/corpus.txt")))
        }

    @Test
    fun incompatibleEntityEmbeddingsRejectRestoreBeforePublication() =
        withPipeline { root ->
            val sourceRoot = root.resolve("source")
            GraphRAG(sourceRoot).upsert("Original corpus")
            val output = sourceRoot.resolve("output")
            ParquetWriterHelper().writeEntityEmbeddings(
                output.resolve("entity_embeddings.parquet"),
                listOf(
                    com.microsoft.graphrag.index
                        .EntityEmbedding("entity", emptyList()),
                ),
            )
            val staged =
                com.microsoft.graphrag.query
                    .QueryIndexLoader(output)
                    .load()
            assertEquals(1, staged.entityEmbeddings.size)
            assertTrue(
                staged.entityEmbeddings
                    .single()
                    .vector
                    .isEmpty(),
            )
            val targetRoot = root.resolve("target")
            val target = GraphRAG(targetRoot)
            target.upsert("Existing corpus")
            assertFailsWith<IllegalArgumentException> { target.loadGraph(output.toString()) }
            assertEquals("Existing corpus", Files.readString(targetRoot.resolve("output/corpus.txt")))
        }

    @Test
    fun basicContextOnlyDoesNotCallChatModel() =
        withPipeline { root ->
            runBlocking {
                val chat = mockk<OpenAiStreamingChatModel>()
                val embedding = mockk<dev.langchain4j.model.embedding.EmbeddingModel>()
                every { embedding.embed(any<String>()) } returns
                    dev.langchain4j.model.output.Response.from(
                        dev.langchain4j.data.embedding
                            .Embedding(floatArrayOf(1.0f, 0.0f)),
                    )
                val text =
                    com.microsoft.graphrag.index
                        .TextUnit("unit", "chunk", "usable evidence", "document")
                val engine =
                    BasicQueryEngine(
                        chat,
                        embedding,
                        LocalVectorStore(root.resolve("vectors.json")),
                        listOf(text),
                        listOf(TextEmbedding("chunk", listOf(1.0, 0.0))),
                    )
                val result = engine.answer("query", "text", includeAnswer = false)
                assertEquals(listOf("usable evidence"), result.context.map { it.text })
                assertEquals(0, result.llmCalls)
                verify { chat wasNot Called }
            }
        }

    @Test
    fun globalSkipsOversizedReportsWithoutSendingHeaderOnlyBatch() =
        runBlocking {
            val contexts = mutableListOf<String>()
            val callback =
                object : QueryCallbacks {
                    override fun onMapResponseStart(mapContexts: List<String>) {
                        contexts.addAll(mapContexts)
                    }
                }
            val engine =
                GlobalSearchEngine(
                    model(),
                    listOf(CommunityReport(1, "oversized ".repeat(300)), CommunityReport(2, "usable evidence")),
                    callbacks = listOf(callback),
                    maxContextTokens = 60,
                )
            engine.search("query")
            assertEquals(1, contexts.size)
            assertTrue("usable evidence" in contexts.single())
            assertTrue(countTokensWithJTokKit(contexts.single()) <= 60)
        }

    @Test
    fun driftStreamsOnlySynthesis() =
        runBlocking {
            val engine =
                DriftSearchEngine(
                    model(),
                    listOf(CommunityReport(1, "evidence")),
                    localQueryEngine = mockk(),
                    maxIterations = 0,
                )
            assertEquals(listOf("""{"response":"final"}"""), engine.streamSearch("query").toList())
        }

    @Test
    fun unifiedFiltersExcludedReferencesAndRequestsContextOnly() =
        runBlocking {
            val delegate = mockk<GraphRAG>()
            val parameters = mutableListOf<QueryParam>()
            coEvery { delegate.aquery(any(), capture(parameters)) } returns
                QueryResult(
                    answer = "answer",
                    context = emptyList(),
                    contextRecords =
                        mapOf(
                            "actions" to listOf(mapOf("answer" to "generated action", "in_context" to "true")),
                            "sources" to
                                listOf(
                                    mapOf("text" to "included", "in_context" to "true"),
                                    mapOf("text" to "excluded", "in_context" to "false"),
                                ),
                        ),
                )
            val result = GraphRagUnifiedAdapter(delegate).aquery("query", UnifiedQuery(includeAnswer = false))
            assertEquals(listOf("included"), result.references.map { it.snippet })
            assertTrue("includeAnswer=false" in parameters.single().toString())
        }

    private fun model(): OpenAiStreamingChatModel =
        mockk {
            every { chat(any<String>(), any<StreamingChatResponseHandler>()) } answers {
                complete(secondArg(), """{"response":"primer","follow_up_queries":[]}""")
            }
            every { chat(any<List<ChatMessage>>(), any<StreamingChatResponseHandler>()) } answers {
                complete(secondArg(), """{"response":"final"}""")
            }
        }

    private fun complete(
        handler: StreamingChatResponseHandler,
        response: String,
    ) {
        handler.onPartialResponse(response)
        handler.onCompleteResponse(mockk())
    }

    private fun withPipeline(test: (Path) -> Unit) {
        val root = Files.createTempDirectory("graphrag-integrity")
        mockkStatic(::defaultPipeline)
        try {
            every { defaultPipeline() } answers {
                SimplePipeline(
                    listOf(
                        "fixture" to { config: GraphRagConfig, _ ->
                            val documents = inputTexts(config.inputDir)
                            Files.writeString(config.outputDir.resolve("corpus.txt"), documents.sorted().joinToString("|"))
                            Files.writeString(
                                config.outputDir.resolve("model.txt"),
                                config.embeddingModelName,
                            )
                            check("Fail indexing" !in documents) { "injected publication failure" }
                            val chunker = DocumentChunker()
                            val chunks = chunker.loadAndChunk(config.inputDir)
                            val writer = ParquetWriterHelper()
                            writer.writeTextUnits(config.outputDir.resolve("text_units.parquet"), chunker.toTextUnits(chunks))
                            writer.writeTextEmbeddings(
                                config.outputDir.resolve("text_embeddings.parquet"),
                                chunks.map { TextEmbedding(it.id, listOf(1.0, 0.0)) },
                            )
                            WorkflowResult()
                        },
                    ),
                )
            }
            test(root)
        } finally {
            unmockkStatic(::defaultPipeline)
            root.toFile().deleteRecursively()
        }
    }

    private fun inputTexts(input: Path): Set<String> =
        Files.list(input).use { files -> files.map { Files.readString(it) }.toList().toSet() }
}
