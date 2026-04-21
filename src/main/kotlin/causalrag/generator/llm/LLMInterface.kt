package causalrag.generator.llm

import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiChatRequestParameters
import dev.langchain4j.model.openaiofficial.OpenAiOfficialChatModel
import io.github.oshai.kotlinlogging.KotlinLogging
import shared.llm.supportsTemperature

private val logger = KotlinLogging.logger {}

/**
 * Thin wrapper around provider-specific chat model implementations.
 *
 * @property modelName Provider model identifier.
 * @property systemMessage Optional system prompt prepended to every request.
 * @property baseUrl Optional base URL override for self-hosted backends.
 */
@Suppress("TooGenericExceptionCaught")
class LLMInterface(
    private val modelName: String = "gpt-4o-mini",
    apiKey: String? = null,
    provider: String = "openai",
    private val systemMessage: String? = null,
    private val baseUrl: String? = null,
) {
    private val providerName = provider.lowercase()
    private val apiKeyValue = apiKey ?: System.getenv("OPENAI_API_KEY")

    private var chatModel: ChatModel? = null
    private var lastTemperature: Double? = null
    private var lastJsonMode: Boolean? = null
    private var lastMaxTokens: Int? = null
    private var lastJsonArrayMode: Boolean? = null

    /**
     * Generates a chat completion for the supplied prompt.
     *
     * @param prompt User prompt.
     * @param temperature Sampling temperature. Currently applied only to the OpenAI provider.
     * @param maxTokens Maximum completion token budget. Currently applied only to the OpenAI provider.
     * @param stream Whether streaming was requested. Streaming is currently ignored for all providers.
     * @param jsonMode Whether to request JSON output when supported. Currently only the OpenAI provider uses this flag.
     * @param jsonArrayMode Whether the prompt expects a JSON array response. For OpenAI this only affects prompt handling;
     * response format remains provider-limited, and other providers currently ignore it.
     * @return Model response text.
     */
    fun generate(
        prompt: String,
        temperature: Double = 1.0,
        maxTokens: Int = 800,
        stream: Boolean = false,
        jsonMode: Boolean = false,
        jsonArrayMode: Boolean = false,
    ): String {
        if (stream) {
            logger.warn { "Streaming not implemented; falling back to non-streaming response." }
        }
        return try {
            when (providerName) {
                "openai" -> {
                    chatOpenAi(prompt, temperature, maxTokens, jsonMode, jsonArrayMode)
                }

                "ollama" -> {
                    val model = getOllamaModel()
                    extractResponseText(model.chat(buildMessages(prompt)))
                }

                else -> {
                    throw IllegalArgumentException("Unsupported provider: $providerName")
                }
            }
        } catch (ex: Exception) {
            logger.error(ex) { "Error generating completion" }
            throw ex
        }
    }

    private fun buildMessages(prompt: String): List<ChatMessage> =
        if (!systemMessage.isNullOrBlank()) {
            listOf(SystemMessage.from(systemMessage), UserMessage.from(prompt))
        } else {
            listOf(UserMessage.from(prompt))
        }

    private fun chatOpenAi(
        prompt: String,
        temperature: Double,
        maxTokens: Int,
        jsonMode: Boolean,
        jsonArrayMode: Boolean,
    ): String = extractResponseText(getOpenAiModel(temperature, maxTokens, jsonMode, jsonArrayMode).chat(buildMessages(prompt)))

    internal fun extractResponseText(response: ChatResponse): String {
        val aiMessage = response.aiMessage()
        val text = aiMessage.text()
        if (text != null) {
            return text
        }

        logger.warn {
            buildString {
                append("LLM response did not include text content")
                if (aiMessage.hasToolExecutionRequests()) {
                    append("; tool execution requests were returned instead")
                }
                aiMessage.thinking()?.takeIf { it.isNotBlank() }?.let {
                    append("; reasoning content was present but no final text was provided")
                }
            }
        }
        return ""
    }

    private fun getOpenAiModel(
        temperature: Double,
        maxTokens: Int,
        jsonMode: Boolean,
        jsonArrayMode: Boolean,
    ): ChatModel {
        check(!apiKeyValue.isNullOrBlank()) { "OPENAI_API_KEY is not configured." }
        val includeTemperature = supportsTemperature(modelName)
        val effectiveTemperature = if (includeTemperature) temperature else null
        if (
            chatModel == null ||
            lastTemperature != effectiveTemperature ||
            lastJsonMode != jsonMode ||
            lastMaxTokens != maxTokens ||
            lastJsonArrayMode != jsonArrayMode
        ) {
            val builder =
                OpenAiOfficialChatModel
                    .builder()
                    .apiKey(apiKeyValue)
                    .modelName(modelName)
                    .defaultRequestParameters(
                        OpenAiChatRequestParameters
                            .builder()
                            .modelName(modelName)
                            .also { builder ->
                                if (includeTemperature) {
                                    builder.temperature(temperature)
                                }
                            }.maxCompletionTokens(maxTokens)
                            .build(),
                    )
            if (baseUrl != null) {
                builder.baseUrl(baseUrl)
            }
            if (jsonMode) {
                if (jsonArrayMode) {
                    logger.debug { "jsonArrayMode enabled; relying on prompt since OpenAI responseFormat is json_object-only." }
                } else {
                    builder.responseFormat("json_object")
                    builder.strictJsonSchema(true)
                }
            }
            chatModel = builder.build()
            lastTemperature = effectiveTemperature
            lastJsonMode = jsonMode
            lastMaxTokens = maxTokens
            lastJsonArrayMode = jsonArrayMode
        }
        return chatModel!!
    }

    private fun getOllamaModel(): ChatModel {
        if (chatModel == null) {
            val builder = OllamaChatModel.builder().modelName(modelName)
            if (baseUrl != null) {
                builder.baseUrl(baseUrl)
            }
            chatModel = builder.build()
        }
        return chatModel!!
    }
}
