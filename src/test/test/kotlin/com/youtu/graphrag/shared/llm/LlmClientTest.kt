package com.youtu.graphrag.shared.llm

import com.sun.net.httpserver.HttpServer
import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.request.ChatRequest
import dev.langchain4j.model.chat.response.ChatResponse
import java.net.InetSocketAddress
import java.util.concurrent.Executors
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LlmClientTest {
    @Test
    fun `chat model client delegates to ChatModel and returns ai text`() {
        val model = RecordingChatModel(response = "model response")
        val client = ChatModelLlmClient(model)

        val output = client.complete("test prompt")

        assertEquals("model response", output)
        assertEquals("test prompt", model.lastPrompt)
    }

    @Test
    fun `chat model client strips markdown fence and json prefix from content`() {
        val model = RecordingChatModel(response = "```json\n{\"answer\":\"ok\"}\n```")
        val client = ChatModelLlmClient(model)

        val output = client.complete("test prompt")

        assertEquals("{\"answer\":\"ok\"}", output)
    }

    @Test
    fun `factory falls back to noop when no api credentials exist`() {
        val client = LlmClientFactory.fromEnvironment(env = emptyMap())
        assertTrue(client is NoopLlmClient)
        assertEquals("", client.complete("any prompt"))
    }

    @Test
    fun `factory openai client works against mock chat completions server`() {
        var requestBody = ""
        val server =
            createMockServer("/v1/chat/completions") { exchange ->
                requestBody = exchange.requestBody.bufferedReader().readText()
                writeChatCompletionResponse(exchange, "mocked completion")
            }
        server.start()

        try {
            val port = server.address.port
            val client =
                LlmClientFactory.fromEnvironment(
                    env =
                        mapOf(
                            "OPENAI_PROVIDER" to "openai",
                            "LLM_API_KEY" to "test-key",
                            "LLM_MODEL" to "test-model",
                            "LLM_BASE_URL" to "http://127.0.0.1:$port/v1",
                        ),
                )

            val output = client.complete("who is the leader?")

            assertEquals("mocked completion", output)
            assertTrue(requestBody.contains("test-model"))
            assertTrue(requestBody.contains("who is the leader?"))
        } finally {
            stopMockServer(server)
        }
    }

    @Test
    fun `factory uses openai provider by default when OPENAI_PROVIDER is missing`() {
        var requestBody = ""
        val server =
            createMockServer("/v1/chat/completions") { exchange ->
                requestBody = exchange.requestBody.bufferedReader().readText()
                writeChatCompletionResponse(exchange, "default provider response")
            }
        server.start()

        try {
            val port = server.address.port
            val client =
                LlmClientFactory.fromEnvironment(
                    env =
                        mapOf(
                            "LLM_API_KEY" to "test-key",
                            "LLM_BASE_URL" to "http://127.0.0.1:$port/v1",
                        ),
                )

            val output = client.complete("default provider prompt")

            assertEquals("default provider response", output)
            assertTrue(requestBody.contains("gpt-4.1-mini"))
            assertTrue(requestBody.contains("default provider prompt"))
        } finally {
            stopMockServer(server)
        }
    }

    @Test
    fun `factory openai compatible provider uses deepseek defaults`() {
        var requestBody = ""
        val server =
            createMockServer("/v1/chat/completions") { exchange ->
                requestBody = exchange.requestBody.bufferedReader().readText()
                writeChatCompletionResponse(exchange, "compat response")
            }
        server.start()

        try {
            val port = server.address.port
            val client =
                LlmClientFactory.fromEnvironment(
                    env =
                        mapOf(
                            "OPENAI_PROVIDER" to "openai_compatible",
                            "LLM_API_KEY" to "test-key",
                            "LLM_BASE_URL" to "http://127.0.0.1:$port/v1",
                        ),
                )

            val output = client.complete("compat provider prompt")

            assertEquals("compat response", output)
            assertTrue(requestBody.contains("deepseek-chat"))
            assertTrue(requestBody.contains("compat provider prompt"))
        } finally {
            stopMockServer(server)
        }
    }

    @Test
    fun `factory azure provider uses deployment path api-version and api-key header`() {
        var requestPath = ""
        var requestQuery = ""
        var apiKeyHeader = ""

        val server =
            createMockServer("/openai/deployments/deployment-x/chat/completions") { exchange ->
                requestPath = exchange.requestURI.path
                requestQuery = exchange.requestURI.query.orEmpty()
                apiKeyHeader = exchange.requestHeaders.getFirst("api-key").orEmpty()
                writeChatCompletionResponse(exchange, "azure response")
            }
        server.start()

        try {
            val port = server.address.port
            val client =
                LlmClientFactory.fromEnvironment(
                    env =
                        mapOf(
                            "OPENAI_PROVIDER" to "azure",
                            "AZURE_OPENAI_API_KEY" to "azure-key",
                            "AZURE_OPENAI_ENDPOINT" to "http://127.0.0.1:$port",
                            "AZURE_OPENAI_DEPLOYMENT" to "deployment-x",
                            "API_VERSION" to "2025-01-01-preview",
                        ),
                )

            val output = client.complete("azure prompt")

            assertEquals("azure response", output)
            assertEquals("/openai/deployments/deployment-x/chat/completions", requestPath)
            assertTrue(requestQuery.contains("api-version=2025-01-01-preview"))
            assertEquals("azure-key", apiKeyHeader)
        } finally {
            stopMockServer(server)
        }
    }
}

private class RecordingChatModel(
    private val response: String,
) : ChatModel {
    var lastPrompt: String? = null

    override fun doChat(request: ChatRequest): ChatResponse {
        val lastUserMessage = UserMessage.findLast(request.messages()).orElse(null)
        lastPrompt = lastUserMessage?.singleText()
        return ChatResponse.builder().aiMessage(AiMessage.from(response)).build()
    }
}

private fun createMockServer(
    path: String,
    handler: (com.sun.net.httpserver.HttpExchange) -> Unit,
): HttpServer {
    val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
    server.executor = Executors.newSingleThreadExecutor()
    server.createContext(path, handler)
    return server
}

private fun stopMockServer(server: HttpServer) {
    val executor = server.executor as? java.util.concurrent.ExecutorService
    server.stop(0)
    executor?.shutdownNow()
}

private fun writeChatCompletionResponse(
    exchange: com.sun.net.httpserver.HttpExchange,
    content: String,
) {
    val response =
        """
        {
          "id": "chatcmpl-test",
          "object": "chat.completion",
          "created": 1,
          "model": "test-model",
          "choices": [
            {
              "index": 0,
              "message": {
                "role": "assistant",
                "content": "$content"
              },
              "finish_reason": "stop"
            }
          ],
          "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2
          }
        }
        """.trimIndent()
    exchange.responseHeaders.add("Content-Type", "application/json")
    exchange.sendResponseHeaders(200, response.toByteArray().size.toLong())
    exchange.responseBody.use { body ->
        body.write(response.toByteArray())
    }
}
