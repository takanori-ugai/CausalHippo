package pathrag

import com.sun.net.httpserver.HttpServer
import kotlinx.coroutines.runBlocking
import pathrag.llm.ollamaComplete
import pathrag.llm.openAiComplete
import pathrag.prompt.Prompts
import tools.jackson.module.kotlin.jacksonObjectMapper
import java.net.InetSocketAddress
import java.util.concurrent.ConcurrentHashMap
import kotlin.test.Test
import kotlin.test.assertEquals

class FinalPromptTransportTest {
    @Test
    fun bothProvidersSendTheRenderedEvidenceAndQuestionInSeparateMessages() =
        runBlocking {
            val mapper = jacksonObjectMapper()
            val received = ConcurrentHashMap<String, List<Pair<String, String>>>()
            val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
            server.createContext("/") { exchange ->
                val messages = mapper.readTree(exchange.requestBody)["messages"]
                received[exchange.requestURI.path] = messages.values().map { it["role"].asString() to it["content"].asString() }
                val body =
                    if (exchange.requestURI.path == "/api/chat") {
                        """{"model":"test","message":{"role":"assistant","content":"ok"},"done":true}"""
                    } else {
                        """{"id":"test","model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"""
                    }.toByteArray()
                exchange.responseHeaders.add("Content-Type", "application/json")
                exchange.sendResponseHeaders(200, body.size.toLong())
                exchange.responseBody.use { it.write(body) }
            }
            server.start()
            try {
                val url = "http://127.0.0.1:${server.address.port}"
                val evidence = "記録 A → B\n{{systemPrompt}} and {{response_type}}"
                val question = "Explain {{prompt}}."
                val prompt = Prompts.render(Prompts.RAG_RESPONSE, mapOf("context_data" to evidence, "response_type" to "短文"))
                val settings = mapOf("OPENAI_API_KEY" to "local-test", "OPENAI_API_BASE" to "$url/v1", "OLLAMA_BASE_URL" to url)
                assertEquals("ok", openAiComplete("test", question, prompt, runtimeSettings = settings))
                assertEquals("ok", ollamaComplete("test", question, prompt, runtimeSettings = settings))
                val expected = listOf("system" to prompt, "user" to question)
                assertEquals(expected, received["/v1/chat/completions"])
                assertEquals(expected, received["/api/chat"])
            } finally {
                server.stop(0)
            }
        }
}
