package causalrag

import causalrag.generator.llm.LLMInterface
import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.model.chat.response.ChatResponse
import shared.llm.supportsTemperature
import kotlin.test.Test
import kotlin.test.assertEquals

class LLMInterfaceTest {
    @Test
    fun extractResponseTextReturnsEmptyStringWhenAiMessageTextIsMissing() {
        val response =
            ChatResponse
                .builder()
                .aiMessage(AiMessage.builder().thinking("intermediate reasoning").build())
                .build()

        val llm = LLMInterface()

        assertEquals("", llm.extractResponseText(response))
    }

    @Test
    fun supportsTemperatureMatchesKnownOpenAiModelCapabilities() {
        assertEquals(false, supportsTemperature("gpt-5-mini"))
        assertEquals(true, supportsTemperature("gpt-4o-mini"))
    }
}
