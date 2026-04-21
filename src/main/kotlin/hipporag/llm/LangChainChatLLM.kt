package hipporag.llm

import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatModel
import hipporag.utils.LlmResult
import hipporag.utils.Message

/**
 * Adapter for LangChain4j chat models to the [BaseLLM] interface.
 */
class LangChainChatLLM(
    private val model: ChatModel,
) : BaseLLM {
    /**
     * Sends [messages] to the underlying LangChain4j model.
     */
    override fun infer(messages: List<Message>): LlmResult {
        val chatMessages = messages.map { toChatMessage(it) }
        val response = chat(chatMessages)
        val aiMessage = response.aiMessage()
        val responseText = aiMessage.text()

        val metadata = mutableMapOf<String, Any>()
        val tokenUsage = response.tokenUsage()
        if (tokenUsage != null) {
            metadata["prompt_tokens"] = tokenUsage.inputTokenCount()
            metadata["completion_tokens"] = tokenUsage.outputTokenCount()
            metadata["total_tokens"] = tokenUsage.totalTokenCount()
        }

        return LlmResult(response = responseText ?: "", metadata = metadata)
    }

    private fun chat(chatMessages: List<ChatMessage>) = model.chat(chatMessages)

    private fun toChatMessage(message: Message): ChatMessage =
        when (message.role.lowercase()) {
            "system" -> SystemMessage.from(message.content)
            "assistant", "ai" -> AiMessage.from(message.content)
            "user" -> UserMessage.from(message.content)
            else -> throw IllegalArgumentException("Unsupported message role: '${message.role}'")
        }
}
