package ragas.model

import kotlinx.serialization.Serializable

/** Base contract for evaluation samples that can be serialized to row maps. */
@Serializable
sealed interface Sample {
    /** Converts the sample into a backend-friendly key/value map. */
    fun toMap(): Map<String, Any?>
}

/**
 * Single-turn sample used by RAG and response-quality metrics.
 *
 * @property userInput User question or prompt.
 * @property retrievedContexts Contexts retrieved by the system under test.
 * @property referenceContexts Ground-truth supporting contexts.
 * @property retrievedContextIds Stable IDs for retrieved contexts.
 * @property referenceContextIds Stable IDs for reference contexts.
 * @property response Model response to evaluate.
 * @property multiResponses Optional alternative responses for list-based metrics.
 * @property reference Ground-truth answer text.
 * @property rubrics Optional rubric labels and descriptions.
 * @property personaName Optional persona metadata.
 * @property queryStyle Optional query style metadata.
 * @property queryLength Optional query length bucket metadata.
 */
@Serializable
data class SingleTurnSample(
    val userInput: String? = null,
    val retrievedContexts: List<String>? = null,
    val referenceContexts: List<String>? = null,
    val retrievedContextIds: List<String>? = null,
    val referenceContextIds: List<String>? = null,
    val response: String? = null,
    val multiResponses: List<String>? = null,
    val reference: String? = null,
    val rubrics: Map<String, String>? = null,
    val personaName: String? = null,
    val queryStyle: String? = null,
    val queryLength: String? = null,
) : Sample {
    /** Serializes this sample using snake_case keys expected by metric pipelines. */
    override fun toMap(): Map<String, Any?> =
        mapOf(
            "user_input" to userInput,
            "retrieved_contexts" to retrievedContexts,
            "reference_contexts" to referenceContexts,
            "retrieved_context_ids" to retrievedContextIds,
            "reference_context_ids" to referenceContextIds,
            "response" to response,
            "multi_responses" to multiResponses,
            "reference" to reference,
            "rubrics" to rubrics,
            "persona_name" to personaName,
            "query_style" to queryStyle,
            "query_length" to queryLength,
        ).filterValues { it != null }
}

/**
 * Multi-turn conversation sample used by agent and dialogue metrics.
 *
 * @property userInput Ordered conversation messages.
 * @property reference Optional reference answer or conversation outcome.
 * @property referenceToolCalls Optional expected tool-call traces.
 * @property rubrics Optional rubric labels and descriptions.
 * @property referenceTopics Optional expected topic list for topic-adherence metrics.
 */
@Serializable
data class MultiTurnSample(
    val userInput: List<ConversationMessage>,
    val reference: String? = null,
    val referenceToolCalls: List<ToolCall>? = null,
    val rubrics: Map<String, String>? = null,
    val referenceTopics: List<String>? = null,
) : Sample {
    init {
        validateUserInput(userInput)
    }

    /** Converts typed messages into normalized map payloads used by evaluators. */
    fun toMessages(): List<Map<String, Any?>> =
        userInput.map { message ->
            when (message) {
                is HumanMessage -> {
                    mapOf(
                        "type" to "human",
                        "content" to message.content,
                        "metadata" to message.metadata,
                    )
                }

                is ToolMessage -> {
                    mapOf(
                        "type" to "tool",
                        "content" to message.content,
                        "metadata" to message.metadata,
                    )
                }

                is AiMessage -> {
                    val content: Any =
                        if (message.toolCalls.isNullOrEmpty()) {
                            message.content
                        } else {
                            mapOf(
                                "text" to message.content,
                                "tool_calls" to message.toolCalls,
                            )
                        }
                    mapOf(
                        "type" to "ai",
                        "content" to content,
                        "metadata" to message.metadata,
                    )
                }
            }
        }

    /** Returns a readable multi-line conversation transcript. */
    fun prettyRepr(): String = userInput.joinToString(separator = "\n") { it.prettyRepr() }

    /** Serializes this sample using snake_case keys expected by metric pipelines. */
    override fun toMap(): Map<String, Any?> =
        mapOf(
            "user_input" to toMessages(),
            "reference" to reference,
            "reference_tool_calls" to referenceToolCalls,
            "rubrics" to rubrics,
            "reference_topics" to referenceTopics,
        ).filterValues { it != null }

    private fun validateUserInput(messages: List<ConversationMessage>) {
        var hasSeenAiMessage = false

        messages.forEachIndexed { index, message ->
            when (message) {
                is AiMessage -> {
                    hasSeenAiMessage = true
                }

                is ToolMessage -> {
                    require(hasSeenAiMessage) {
                        "ToolMessage must be preceded by an AiMessage somewhere in the conversation."
                    }

                    if (index > 0) {
                        val previousMessage = messages[index - 1]
                        when (previousMessage) {
                            is AiMessage -> {
                                require(!previousMessage.toolCalls.isNullOrEmpty()) {
                                    "ToolMessage must follow an AiMessage where tools were called."
                                }
                            }

                            is ToolMessage -> {
                                Unit
                            }

                            is HumanMessage -> {
                                error("ToolMessage must follow an AiMessage or another ToolMessage.")
                            }
                        }
                    }
                }

                is HumanMessage -> {
                    Unit
                }
            }
        }
    }
}
