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
import com.microsoft.graphrag.query.LocalQueryEngine
import com.microsoft.graphrag.query.QueryCallbacks
import com.microsoft.graphrag.query.QueryResult
import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import dev.langchain4j.model.output.Response
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
import shared.rag.unified.UnifiedMode
import shared.rag.unified.UnifiedQuery
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
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
    fun globalContextOnlyPacksMultipleBatchesAndIncludesConversationOnce() =
        runBlocking {
            val chat = mockk<OpenAiStreamingChatModel>()
            val first = "alpha ".repeat(22).trim()
            val second = "beta ".repeat(22).trim()
            val history = "Earlier question"
            val engine = GlobalSearchEngine(chat, listOf(CommunityReport(1, first), CommunityReport(2, second)), maxContextTokens = 58)

            val result = engine.retrieveContext("question", listOf(history))
            assertEquals("", result.answer)
            assertEquals(0, result.llmCalls)
            assertEquals(2, result.contextText.split("-----Reports-----").size - 1)
            assertEquals(1, result.contextText.split(history).size - 1)
            assertEquals(
                setOf(first, second),
                result.contextRecords
                    .getValue("reports")
                    .map { it.getValue("summary") }
                    .toSet(),
            )
            assertEquals(listOf(history), result.contextRecords.getValue("conversation").map { it.getValue("content") })
            verify { chat wasNot Called }
        }

    @Test
    fun localContextOnlyReturnsEvidenceWithoutCallingChat() =
        withPipeline { root ->
            runBlocking {
                val chat = mockk<OpenAiStreamingChatModel>()
                val embedding = mockk<EmbeddingModel>()
                every { embedding.embed(any<String>()) } returns Response.from(Embedding(floatArrayOf(1.0f, 0.0f)))
                val source =
                    com.microsoft.graphrag.index
                        .TextUnit("unit", "chunk", "usable local evidence", "source")
                val engine =
                    LocalQueryEngine(
                        streamingModel = chat,
                        embeddingModel = embedding,
                        vectorStore = LocalVectorStore(root.resolve("vectors.json")),
                        textUnits = listOf(source),
                        textEmbeddings = listOf(TextEmbedding("chunk", listOf(1.0, 0.0))),
                        entities = emptyList(),
                        entitySummaries = emptyList(),
                        relationships = emptyList(),
                        claims = emptyList(),
                        covariates = emptyMap(),
                        communities = emptyList(),
                        communityReports = emptyList(),
                    )
                val result = engine.answer("question", "text", includeAnswer = false)
                assertEquals("", result.answer)
                assertEquals(0, result.llmCalls)
                assertTrue("usable local evidence" in result.contextText)
                verify { chat wasNot Called }
            }
        }

    @Test
    fun dropClearsCachedIndexAndPermitsRebuildingFromNewDocuments() =
        withPipeline { root ->
            val rag = GraphRAG(root)
            rag.upsert("Old corpus")
            rag.inspectGraph()
            rag.drop()
            val inspection = rag.inspectGraph()
            assertEquals(emptyList<Any>(), inspection["nodes"])
            assertTrue(Files.notExists(root.resolve("output/corpus.txt")))
            rag.upsert("New corpus")
            assertEquals("New corpus", Files.readString(root.resolve("output/corpus.txt")))
        }

    @Test
    fun queryModesReturnTheirEvidenceWithoutCallingChat() =
        withPipeline { root ->
            val chat = mockk<OpenAiStreamingChatModel>()
            val models = mutableListOf<String>()
            val rag = ragWithModels(root, chat, models)
            rag.upsert("Local evidence")
            Files.writeString(root.resolve("output/community_reports.json"), """[{"communityId":1,"summary":"Community evidence"}]""")

            for (mode in listOf("basic", "local", "global", "drift")) {
                val result = rag.query("question", QueryParam(mode = mode, includeAnswer = false, streaming = true))
                assertEquals("", result.answer, mode)
                assertEquals(0, result.llmCalls, mode)
                if (mode != "global") {
                    assertTrue(result.context.any { it.text.contains("Local evidence") } || "Local evidence" in result.contextText, mode)
                }
                if (mode == "global" || mode == "drift") {
                    assertTrue("Community evidence" in result.contextText, mode)
                    assertTrue(result.contextRecords.getValue("reports").isNotEmpty(), mode)
                }
            }
            assertEquals(List(4) { "fixture-embedding" }, models)
            verify { chat wasNot Called }
        }

    @Test
    fun queryStreamsAnswersFromEachMode() =
        withPipeline { root ->
            val rag = ragWithModels(root, model())
            rag.upsert("Local evidence")
            Files.writeString(root.resolve("output/community_reports.json"), """[{"communityId":1,"summary":"Community evidence"}]""")
            for (mode in listOf("basic", "local", "global", "drift")) {
                val result = rag.query("question", QueryParam(mode = mode, maxIterations = 0, streaming = true))
                assertTrue(result.answer.isNotBlank(), mode)
            }
        }

    @Test
    fun vectorOperationsPersistRowsAndDeleteOnlyTheRequestedEntity() =
        withPipeline { root ->
            runBlocking {
                val models = mutableListOf<String>()
                val rag = ragWithModels(root, mockk(), models)
                rag.upsert("Local evidence")
                val original = rag.query("question", 10)
                assertEquals(listOf("Local evidence"), original.map { it["content"] })
                rag.upsert(
                    mapOf(
                        "fixture-entity" to mapOf("kind" to "entity", "vector" to listOf(1.0, 0.0)),
                        "fixture-text" to mapOf("vector" to listOf(1.0, 0.0)),
                    ),
                )
                val store = LocalVectorStore(root.resolve("output/vector_store.json"))
                rag.deleteEntityRelation("fixture-entity")
                assertEquals(listOf("fixture-entity"), store.load()?.entityEmbeddings?.map { it.entityId })
                assertEquals(original.map { it["id"] }.toSet() + "fixture-text", rag.query("question", 10).map { it["id"] }.toSet())
                rag.deleteEntity("fixture-entity")
                assertTrue(
                    store
                        .load()
                        ?.entityEmbeddings
                        .orEmpty()
                        .none { it.entityId == "fixture-entity" },
                )
                assertEquals(original.map { it["id"] }.toSet() + "fixture-text", rag.query("question", 10).map { it["id"] }.toSet())
                assertTrue(models.all { it == "fixture-embedding" })
            }
        }

    @Test
    fun invalidSnapshotPathsAndTextVectorsPreserveTheCommittedIndex() =
        withPipeline { root ->
            val sourceRoot = root.resolve("source")
            GraphRAG(sourceRoot).upsert("Original corpus")
            val source = sourceRoot.resolve("output")
            val embeddings = source.resolve("text_embeddings.parquet")
            val targetRoot = root.resolve("target")
            val target = GraphRAG(targetRoot)
            target.upsert("Existing corpus")
            assertFailsWith<IllegalArgumentException> { target.saveGraph(" ") }
            assertFailsWith<IllegalArgumentException> { target.loadGraph(" ") }
            assertFailsWith<IllegalArgumentException> { target.loadGraph(root.resolve("missing").toString()) }
            assertFailsWith<IllegalArgumentException> { target.saveGraph(targetRoot.toString()) }

            val chunkId =
                com.microsoft.graphrag.query
                    .QueryIndexLoader(source)
                    .load()
                    .textUnits
                    .single()
                    .chunkId
            for ((vectors, message) in listOf(
                listOf(TextEmbedding("another-chunk", listOf(1.0, 0.0))) to "missing text units or aligned embeddings",
                listOf(TextEmbedding(chunkId, emptyList())) to "invalid text embeddings",
                listOf(TextEmbedding(chunkId, listOf(Double.NaN, 0.0))) to "invalid text embeddings",
            )) {
                Files.delete(embeddings)
                ParquetWriterHelper().writeTextEmbeddings(embeddings, vectors)
                val error = assertFailsWith<IllegalArgumentException> { target.loadGraph(source.toString()) }
                assertTrue(message in error.message.orEmpty(), error.message)
                assertEquals("Existing corpus", Files.readString(targetRoot.resolve("output/corpus.txt")))
            }
        }

    private fun ragWithModels(
        root: Path,
        chat: OpenAiStreamingChatModel,
        models: MutableList<String> = mutableListOf(),
    ): GraphRAG {
        val embedding = mockk<EmbeddingModel>()
        every { embedding.embed(any<String>()) } returns Response.from(Embedding(floatArrayOf(1.0f, 0.0f)))
        return GraphRAG(
            root,
            root.resolve("input"),
            root.resolve("output"),
            root.resolve("update_output"),
            defaultChatModelName = "fixture-chat",
            defaultEmbeddingModelName = "fixture-embedding",
            queryEmbeddingModelFactory = { name ->
                models += name
                embedding
            },
            streamingModelFactory = { chat },
        )
    }

    @Test
    fun blankBundledCorpusRestoresInQueryOnlyMode() =
        withPipeline { root ->
            val source = GraphRAG(root.resolve("source"))
            source.upsert("Original corpus")
            val snapshot = root.resolve("snapshot")
            source.saveGraph(snapshot.toString())
            val input = snapshot.resolve(".graphrag-input")
            Files.list(input).use { paths -> paths.forEach { Files.writeString(it, "   ") } }
            val target = ragWithModels(root.resolve("target"), mockk())
            target.loadGraph(snapshot.toString())
            assertFailsWith<IllegalStateException> { target.upsert("Additional corpus") }
            assertEquals(listOf("Original corpus"), runBlocking { target.query("question", 1) }.map { it["content"] })
        }

    @Test
    fun nestedIndexUsesItsRecordedModelAndRejectsConflictingModels() =
        withPipeline { root ->
            val nested = root.resolve("nested")
            GraphRAG(nested, defaultEmbeddingModelName = "text-embedding-3-large").upsert("Nested evidence")
            val models = mutableListOf<String>()
            val target = ragWithModels(root.resolve("target"), mockk(), models)
            target.loadGraph(nested.toString())
            val param = QueryParam(mode = "basic", includeAnswer = false, embeddingModelName = "text-embedding-3-large")
            assertEquals(listOf("Nested evidence"), target.query("question", param).context.map { it.text })
            assertEquals(listOf("Nested evidence"), runBlocking { target.query("question", 1) }.map { it["content"] })
            assertEquals(List(2) { "text-embedding-3-large" }, models)

            GraphRAG(nested.resolve("second")).upsert("Other evidence")
            assertFailsWith<IllegalArgumentException> { target.loadGraph(nested.toString()) }
            assertEquals(listOf("Nested evidence"), target.query("question", param).context.map { it.text })
        }

    @Test
    fun dynamicGlobalContextReportsCommunityRatingUsage() =
        runBlocking {
            val chat = mockk<OpenAiStreamingChatModel>()
            every { chat.chat(any<List<ChatMessage>>(), any<StreamingChatResponseHandler>()) } answers {
                complete(secondArg(), """{"rating":5}""")
            }
            val engine = GlobalSearchEngine(chat, listOf(CommunityReport(1, "Relevant evidence")), dynamicCommunitySelection = true)
            val result = engine.retrieveContext("question", emptyList())
            assertEquals(1, result.llmCalls)
            assertTrue(result.promptTokens > 0)
            assertTrue(result.outputTokens > 0)
            assertEquals(mapOf("build_context" to 1), result.llmCallsCategories)
            assertEquals(mapOf("build_context" to result.promptTokens), result.promptTokensCategories)
            assertEquals(mapOf("build_context" to result.outputTokens), result.outputTokensCategories)
        }

    @Test
    fun localStreamingRequestRetainsFollowUpsWithoutReturningTheAnswer() =
        withPipeline { root ->
            val chat = mockk<OpenAiStreamingChatModel>()
            every { chat.chat(any<List<ChatMessage>>(), any<StreamingChatResponseHandler>()) } answers {
                complete(secondArg(), """{"response":"answer","follow_up_queries":["What comes next?"]}""")
            }
            val rag = ragWithModels(root, chat)
            rag.upsert("Original evidence")
            val result =
                GraphRagUnifiedAdapter(rag).query(
                    "question",
                    UnifiedQuery(mode = UnifiedMode.LOCAL, includeAnswer = false, includeFollowUps = true, streaming = true),
                )
            assertNull(result.answer)
            assertEquals(listOf("What comes next?"), result.followUpQueries)
        }

    @Test
    fun driftStreamsSynthesisAfterCompletingALocalAction() =
        runBlocking {
            val local = mockk<LocalQueryEngine>()
            coEvery { local.answer(any(), any(), any(), any(), any()) } returns
                QueryResult(answer = "Completed local action", context = emptyList())
            val engine =
                DriftSearchEngine(model(), listOf(CommunityReport(1, "evidence")), localQueryEngine = local, maxIterations = 1)
            assertEquals(listOf("""{"response":"final"}"""), engine.streamSearch("query", listOf("follow-up")).toList())
            io.mockk.coVerify(exactly = 1) { local.answer("follow-up", any(), any(), "query", any()) }
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
