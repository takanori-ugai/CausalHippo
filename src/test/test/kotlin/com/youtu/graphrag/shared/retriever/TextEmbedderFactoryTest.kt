package com.youtu.graphrag.shared.retriever

import com.sun.net.httpserver.HttpServer
import com.youtu.graphrag.shared.config.ConfigManager
import java.net.InetSocketAddress
import java.util.concurrent.Executors
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TextEmbedderFactoryTest {
    @Test
    fun `openai text embedder requests embeddings endpoint and parses vector`() {
        var requestBody = ""
        var authHeader = ""
        val server =
            createMockServer("/v1/embeddings") { exchange ->
                requestBody = exchange.requestBody.bufferedReader().readText()
                authHeader = exchange.requestHeaders.getFirst("Authorization").orEmpty()
                writeEmbeddingResponse(exchange, listOf(0.1f, 0.2f, 0.3f))
            }
        server.start()

        try {
            val port = server.address.port
            val fallback = HashTextEmbedder(dimensions = 3, modelName = "fallback")
            val embedder =
                OpenAiTextEmbedder(
                    modelName = "text-embedding-3-small",
                    baseUrl = "http://127.0.0.1:$port/v1",
                    apiKey = "test-key",
                    dimensions = 3,
                    fallback = fallback,
                )

            val vector = embedder.embed("who discovered gravity?")

            assertTrue(vector.contentEquals(floatArrayOf(0.1f, 0.2f, 0.3f)))
            assertTrue(requestBody.contains("\"model\":\"text-embedding-3-small\""))
            assertTrue(requestBody.contains("who discovered gravity?"))
            assertEquals("Bearer test-key", authHeader)
        } finally {
            stopMockServer(server)
        }
    }

    @Test
    fun `openai text embedder falls back to hash embedding on request failure`() {
        val server =
            createMockServer("/v1/embeddings") { exchange ->
                exchange.sendResponseHeaders(500, 0)
                exchange.responseBody.close()
            }
        server.start()

        try {
            val port = server.address.port
            val fallback = HashTextEmbedder(dimensions = 5, modelName = "fallback")
            val embedder =
                OpenAiTextEmbedder(
                    modelName = "text-embedding-3-small",
                    baseUrl = "http://127.0.0.1:$port/v1",
                    apiKey = "test-key",
                    dimensions = 5,
                    fallback = fallback,
                )

            val input = "alpha beta gamma"
            val vector = embedder.embed(input)
            val expected = fallback.embed(input)

            assertTrue(vector.contentEquals(expected))
        } finally {
            stopMockServer(server)
        }
    }

    @Test
    fun `factory falls back to hash when openai provider has no api key`() {
        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
                "embeddings" to
                    mapOf(
                        "provider" to "openai",
                        "model_name" to "text-embedding-3-small",
                        "dimensions" to 8,
                    ),
            ),
        )

        val embedder =
            RetrieverEmbedderFactory.fromConfig(
                config = config,
                env =
                    mapOf(
                        "EMBEDDING_PROVIDER" to "openai",
                    ),
            )

        assertTrue(embedder is HashTextEmbedder)
        assertEquals(8, embedder.dimensions)
        assertTrue(embedder.modelTag.startsWith("hash:"))
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

private fun writeEmbeddingResponse(
    exchange: com.sun.net.httpserver.HttpExchange,
    embedding: List<Float>,
) {
    val embeddingPayload = embedding.joinToString(separator = ",") { value -> value.toString() }
    val response =
        """
        {
          "object": "list",
          "data": [
            {
              "object": "embedding",
              "embedding": [$embeddingPayload],
              "index": 0
            }
          ],
          "model": "test-embedding-model"
        }
        """.trimIndent()
    exchange.responseHeaders.add("Content-Type", "application/json")
    exchange.sendResponseHeaders(200, response.toByteArray().size.toLong())
    exchange.responseBody.use { body ->
        body.write(response.toByteArray())
    }
}
