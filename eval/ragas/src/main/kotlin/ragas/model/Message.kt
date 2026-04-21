package ragas.model

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement

/**
 * Structured representation of one tool call requested by an AI message.
 *
 * @property name Tool name or identifier.
 * @property args JSON-serializable tool arguments.
 */
@Serializable
data class ToolCall(
    val name: String,
    val args: Map<String, JsonElement> = emptyMap(),
)

/**
 * Base type for messages in a multi-turn conversation sample.
 *
 * @property content Main text content for the message.
 * @property metadata Optional structured metadata for downstream integrations.
 */
@Serializable
sealed interface ConversationMessage {
    val content: String
    val metadata: Map<String, JsonElement>?

    /** Returns a human-readable representation of the message. */
    fun prettyRepr(): String
}

/**
 * User-authored conversation message.
 *
 * @property content Natural-language user text.
 * @property metadata Optional message metadata.
 */
@Serializable
data class HumanMessage(
    override val content: String,
    override val metadata: Map<String, JsonElement>? = null,
) : ConversationMessage {
    /** Returns `"Human: <content>"`. */
    override fun prettyRepr(): String = "Human: $content"
}

/**
 * Tool output message emitted after a tool invocation.
 *
 * @property content Tool output text.
 * @property metadata Optional message metadata.
 */
@Serializable
data class ToolMessage(
    override val content: String,
    override val metadata: Map<String, JsonElement>? = null,
) : ConversationMessage {
    /** Returns `"ToolOutput: <content>"`. */
    override fun prettyRepr(): String = "ToolOutput: $content"
}

/**
 * Assistant message with optional tool-call requests.
 *
 * @property content Assistant response text.
 * @property toolCalls Optional tool calls requested by the assistant.
 * @property metadata Optional message metadata.
 */
@Serializable
data class AiMessage(
    override val content: String,
    val toolCalls: List<ToolCall>? = null,
    override val metadata: Map<String, JsonElement>? = null,
) : ConversationMessage {
    /**
     * Returns a human-readable representation including content and tool calls.
     *
     * Output can span multiple lines when tool calls are present.
     */
    override fun prettyRepr(): String {
        val lines = mutableListOf<String>()
        if (content.isNotBlank()) {
            lines += "AI: $content"
        }
        if (!toolCalls.isNullOrEmpty()) {
            lines += "Tools:"
            toolCalls.forEach { tool ->
                lines += "  ${tool.name}: ${tool.args}"
            }
        }
        return lines.joinToString("\n")
    }
}
