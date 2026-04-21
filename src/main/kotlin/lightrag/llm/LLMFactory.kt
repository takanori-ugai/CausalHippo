package lightrag.llm

import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.StreamingChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.model.ollama.OllamaEmbeddingModel
import dev.langchain4j.model.ollama.OllamaStreamingChatModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import java.time.Duration

/**
 * A factory for creating [ChatModel]s and [EmbeddingModel]s.
 */
object LLMFactory {
    /**
     * Build a [ChatModel] for the given binding.
     *
     * Example using an OllamaModel served locally:
     * ```
     * val chatModel = LLMFactory.createChatModel(
     *     binding = "ollama",
     *     modelName = "llama3",
     *     baseUrl = "http://localhost:11434",
     *     temperature = 0.1,
     *     logRequests = false,
     *     logResponses = false,
     * )
     * ```
     * @param binding The binding to use (e.g., "openai", "ollama").
     * @param modelName The name of the model to use.
     * @param baseUrl The base URL of the model server.
     * @param apiKey The API key to use.
     * @param timeout The timeout in seconds.
     * @param temperature The temperature to use.
     * @param logRequests Whether to log requests.
     * @param logResponses Whether to log responses.
     * @return A [ChatModel] instance.
     */
    fun createChatModel(
        binding: String,
        modelName: String,
        baseUrl: String? = null,
        apiKey: String? = null,
        timeout: Long = 60,
        temperature: Double? = null,
        logRequests: Boolean = true,
        logResponses: Boolean = true,
    ): ChatModel =
        when (binding) {
            "openai" -> {
                val builder =
                    OpenAiChatModel
                        .builder()
                        .modelName(modelName)
                        .apiKey(apiKey ?: "demo")
                        .logRequests(logRequests)
                        .logResponses(logResponses)
                        .timeout(Duration.ofSeconds(timeout))
                if (baseUrl != null) {
                    builder.baseUrl(baseUrl)
                }
                builder.build()
            }

            "ollama" -> {
                val builder =
                    OllamaChatModel
                        .builder()
                        .modelName(modelName)
                        .baseUrl(baseUrl ?: "http://localhost:11434")
                        .timeout(Duration.ofSeconds(timeout))
                        .logRequests(logRequests)
                        .logResponses(logResponses)
                if (temperature != null) {
                    builder.temperature(temperature)
                }
                builder.build()
            }

            else -> {
                throw IllegalArgumentException("Unsupported LLM binding: $binding")
            }
        }

    /**
     * Build a [StreamingChatModel] for the given binding.
     */
    fun createStreamingChatModel(
        binding: String,
        modelName: String,
        baseUrl: String? = null,
        apiKey: String? = null,
        timeout: Long = 60,
        temperature: Double? = null,
        logRequests: Boolean = true,
        logResponses: Boolean = true,
    ): StreamingChatModel =
        when (binding) {
            "openai" -> {
                val builder =
                    OpenAiStreamingChatModel
                        .builder()
                        .modelName(modelName)
                        .apiKey(apiKey ?: "demo")
                        .logRequests(logRequests)
                        .logResponses(logResponses)
                        .timeout(Duration.ofSeconds(timeout))
                if (baseUrl != null) {
                    builder.baseUrl(baseUrl)
                }
                builder.build()
            }

            "ollama" -> {
                val builder =
                    OllamaStreamingChatModel
                        .builder()
                        .modelName(modelName)
                        .baseUrl(baseUrl ?: "http://localhost:11434")
                        .timeout(Duration.ofSeconds(timeout))
                        .logRequests(logRequests)
                        .logResponses(logResponses)
                if (temperature != null) {
                    builder.temperature(temperature)
                }
                builder.build()
            }

            else -> {
                throw IllegalArgumentException("Unsupported streaming LLM binding: $binding")
            }
        }

    /**
     * Build an [EmbeddingModel] for the given binding.
     * @param binding The binding to use (e.g., "openai", "ollama").
     * @param modelName The name of the model to use.
     * @param baseUrl The base URL of the model server.
     * @param apiKey The API key to use.
     * @param timeout The timeout in seconds.
     * @return An [EmbeddingModel] instance.
     */
    fun createEmbeddingModel(
        binding: String,
        modelName: String,
        baseUrl: String? = null,
        apiKey: String? = null,
        timeout: Long = 60,
    ): EmbeddingModel =
        when (binding) {
            "openai" -> {
                val builder =
                    OpenAiEmbeddingModel
                        .builder()
                        .modelName(modelName)
                        .apiKey(apiKey ?: "demo")
                        .logRequests(true)
                        .timeout(Duration.ofSeconds(timeout))
                if (baseUrl != null) {
                    builder.baseUrl(baseUrl)
                }
                builder.build()
            }

            "ollama" -> {
                val builder =
                    OllamaEmbeddingModel
                        .builder()
                        .modelName(modelName)
                        .baseUrl(baseUrl ?: "http://localhost:11434")
                        .timeout(Duration.ofSeconds(timeout))
                builder.build()
            }

            else -> {
                throw IllegalArgumentException("Unsupported embedding binding: $binding")
            }
        }
}
