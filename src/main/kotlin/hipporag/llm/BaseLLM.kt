package hipporag.llm

import hipporag.utils.LlmResult
import hipporag.utils.Message

/**
 * Abstraction for chat-oriented language models.
 */
interface BaseLLM {
    /**
     * Performs inference for the given [messages].
     */
    fun infer(messages: List<Message>): LlmResult
}
