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
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler

/**
 * Adapter that exposes both [ChatModel] and [StreamingChatModel] interfaces using the provided implementations.
 */
class DualChatModel(
    private val chatModel: ChatModel,
    private val streamingChatModel: StreamingChatModel,
) : ChatModel,
    StreamingChatModel {
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

    // StreamingChatModel delegate
    override fun chat(
        request: ChatRequest,
        handler: StreamingChatResponseHandler,
    ) = streamingChatModel.chat(request, handler)

    override fun doChat(
        request: ChatRequest,
        handler: StreamingChatResponseHandler,
    ) = streamingChatModel.doChat(request, handler)

    override fun chat(
        message: String,
        handler: StreamingChatResponseHandler,
    ) = streamingChatModel.chat(message, handler)

    override fun chat(
        messages: List<ChatMessage>,
        handler: StreamingChatResponseHandler,
    ) = streamingChatModel.chat(messages, handler)
}
