package lightrag.llm

import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.model.ModelProvider
import dev.langchain4j.model.chat.Capability
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.StreamingChatModel
import dev.langchain4j.model.chat.listener.ChatModelListener
import dev.langchain4j.model.chat.request.ChatRequest
import dev.langchain4j.model.chat.request.ChatRequestParameters
import dev.langchain4j.model.chat.response.ChatResponse

/**
 * Synchronous chat adapter paired with a separate streaming implementation.
 *
 * LangChain4j 1.20 declares overlapping methods with incompatible return types on
 * [ChatModel] and [StreamingChatModel], so a single JVM class cannot implement both.
 */
class DualChatModel(
    private val chatModel: ChatModel,
    val streamingChatModel: StreamingChatModel,
) : ChatModel {
    // ChatModel delegate
    override fun chat(request: ChatRequest): ChatResponse = chatModel.chat(request)

    override fun doChat(request: ChatRequest): ChatResponse = chatModel.doChat(request)

    override fun defaultRequestParameters(): ChatRequestParameters = chatModel.defaultRequestParameters()

    override fun listeners(): List<ChatModelListener> = chatModel.listeners()

    override fun provider(): ModelProvider = chatModel.provider()

    override fun chat(message: String): String = chatModel.chat(message)

    override fun chat(vararg messages: ChatMessage): ChatResponse = chatModel.chat(*messages)

    override fun chat(messages: List<ChatMessage>): ChatResponse = chatModel.chat(messages)

    override fun supportedCapabilities(): Set<Capability> = chatModel.supportedCapabilities()
}
