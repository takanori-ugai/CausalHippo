package lightrag.operate

import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.StreamingChatModel
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.runBlocking
import lightrag.core.QueryParam
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.utils.Prompts
import lightrag.utils.computeMd5
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** Unit tests validating naive query behavior, caching, and streaming paths. */
class NaiveQueryTest {
    private fun defaultParams(
        query: String = "What is AI?",
        chatModel: ChatModel? = mockk(),
        chunks: List<Map<String, Any?>> = listOf(mapOf("content" to "AI is Artificial Intelligence", "reference_id" to "ref1")),
        tokenizer: (String) -> List<Int> = { it.map { c -> c.code } },
        decoder: (List<Int>) -> String = { list -> list.map { it.toChar() }.joinToString("") },
        globalConfig: Map<String, Any?> = emptyMap(),
        onlyNeedContext: Boolean = false,
        onlyNeedPrompt: Boolean = false,
        stream: Boolean = false,
        hashingKv: BaseKVStorage? = null,
    ): NaiveQueryParams {
        val chunksVdb = mockk<BaseVectorStorage>()
        coEvery { chunksVdb.query(any(), any(), any()) } returns
            chunks.map {
                it.filterValues { v -> v != null }.mapValues { entry -> entry.value!! }
            }
        every { chunksVdb.cosineBetterThanThreshold } returns 0.8

        return NaiveQueryParams(
            query = query,
            chatModel = chatModel,
            globalConfig = globalConfig,
            tokenizer = tokenizer,
            decoder = decoder,
            chunksVdb = chunksVdb,
            queryParam =
                QueryParam(
                    onlyNeedContext = onlyNeedContext,
                    onlyNeedPrompt = onlyNeedPrompt,
                    stream = stream,
                    mode = "naive",
                ),
            hashingKv = hashingKv,
            systemPrompt = null,
        )
    }

    /** Ensures blank queries return the fail response prompt. */
    @Test
    fun `returns fail response when query is blank`() =
        runBlocking {
            val params = defaultParams(query = "")
            val result = naiveQuery(params)
            assertNotNull(result)
            assertEquals(Prompts.FAIL_RESPONSE, result.content)
        }

    /** Verifies an error is surfaced when no chat model is configured. */
    @Test
    fun `returns error when chat model is missing`() =
        runBlocking {
            val params = defaultParams(chatModel = null)
            val result = naiveQuery(params)
            assertNotNull(result)
            assertNotNull(result.content)
            assertTrue(result.content.contains("Error: No LLM model configured"))
        }

    /** Confirms null is returned when no chunks are retrieved for a query. */
    @Test
    fun `returns null when no chunks found`() =
        runBlocking {
            val params = defaultParams(chunks = emptyList())
            val result = naiveQuery(params)
            assertEquals(null, result)
        }

    /** Checks that context-only requests return chunk context and references. */
    @Test
    fun `returns context only when onlyNeedContext is true`() {
        runBlocking {
            val params = defaultParams(onlyNeedContext = true)
            val result = naiveQuery(params)
            assertNotNull(result)
            assertNotNull(result.content)
            assertTrue(result.content.contains("ref1") || result.content.contains("AI is Artificial Intelligence"))
            assertNotNull(result.rawData)
        }
    }

    /** Checks that prompt-only requests return the constructed prompt and references. */
    @Test
    fun `returns prompt only when onlyNeedPrompt is true`() {
        runBlocking {
            val params = defaultParams(onlyNeedPrompt = true)
            val result = naiveQuery(params)
            assertNotNull(result)
            assertNotNull(result.content)
            assertTrue(result.content.contains("---User Query---"))
            assertNotNull(result.rawData)
        }
    }

    /** Ensures cached responses are returned when LLM cache is enabled and populated. */
    @Test
    fun `returns cached result if present`() =
        runBlocking {
            val mockKv = mockk<BaseKVStorage>()
            val params = defaultParams(globalConfig = mapOf("enable_llm_cache" to true), hashingKv = mockKv)

            val maxTotalTokens = params.queryParam.maxTotalTokens.coerceAtMost(DEFAULT_MAX_TOTAL_TOKENS)
            val expectedHash =
                computeMd5(
                    listOf(
                        params.queryParam.mode,
                        params.query,
                        params.queryParam.responseType ?: "",
                        params.queryParam.topK.toString(),
                        params.queryParam.chunkTopK.toString(),
                        params.queryParam.maxEntityTokens.toString(),
                        params.queryParam.maxRelationTokens.toString(),
                        maxTotalTokens.toString(),
                        params.queryParam.userPrompt ?: "",
                        params.queryParam.enableRerank.toString(),
                    ).joinToString("|"),
                )

            coEvery { mockKv.getById(expectedHash) } returns
                mapOf(
                    "content" to "Cached response",
                    "create_time" to 123456789L,
                )

            val result = naiveQuery(params)
            assertNotNull(result)
            assertEquals("Cached response", result.content)
        }

    /** Confirms the LLM is invoked and its response is surfaced when no cache exists. */
    @Test
    fun `calls LLM and returns its response`() =
        runBlocking {
            val mockModel = mockk<ChatModel>()
            every { mockModel.chat(any<List<ChatMessage>>()) } returns ChatResponse.builder().aiMessage(AiMessage("LLM answer")).build()
            val params = defaultParams(chatModel = mockModel)
            val result = naiveQuery(params)
            assertNotNull(result)
            assertNotNull(result.content)
            assertTrue(result.content.contains("LLM answer"))
        }

    /** Verifies streaming responses are produced when a streaming-capable model is provided. */
    @Test
    fun `returns streaming response when stream is true and model supports it`() =
        runBlocking {
            val streamingModel = mockk<ChatModel>(moreInterfaces = arrayOf(StreamingChatModel::class))
            val streamingModelCast = streamingModel as StreamingChatModel

            val handlerSlot = slot<StreamingChatResponseHandler>()
            every { streamingModelCast.chat(any<List<ChatMessage>>(), capture(handlerSlot)) } answers {
                handlerSlot.captured.onPartialResponse("token1")
                handlerSlot.captured.onCompleteResponse(ChatResponse.builder().aiMessage(AiMessage("streamed")).build())
            }

            val params = defaultParams(chatModel = streamingModel, stream = true)
            val result = naiveQuery(params)
            assertNotNull(result)
            assertTrue(result.isStreaming)
        }
}
