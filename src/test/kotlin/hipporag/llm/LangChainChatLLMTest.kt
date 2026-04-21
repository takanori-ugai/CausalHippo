package hipporag.llm

import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.response.ChatResponse
import hipporag.utils.Message
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import shared.llm.supportsTemperature
import kotlin.test.Test
import kotlin.test.assertEquals

class LangChainChatLLMTest {
    @Test
    fun inferUsesConfiguredModelWithoutFallback() {
        val model = mockk<ChatModel>()
        val response =
            ChatResponse
                .builder()
                .aiMessage(AiMessage.from("response"))
                .build()

        every { model.chat(any<List<ChatMessage>>()) } returns response

        val llm = LangChainChatLLM(model)

        val result = llm.infer(listOf(Message("user", "hello")))

        assertEquals("response", result.response)
        verify(exactly = 1) { model.chat(any<List<ChatMessage>>()) }
    }

    @Test
    fun supportsTemperatureUsesConfiguredModelMap() {
        assertEquals(false, supportsTemperature("gpt-5-mini"))
        assertEquals(true, supportsTemperature("gpt-4o-mini"))
        assertEquals(true, supportsTemperature("unknown-model"))
    }
}
