package com.youtu.graphrag.shared.llm

import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.model.openai.OpenAiChatModel
import io.github.oshai.kotlinlogging.KotlinLogging

interface LlmClient {
    fun complete(prompt: String): String
}

class NoopLlmClient : LlmClient {
    override fun complete(prompt: String): String = ""
}

class ChatModelLlmClient(
    private val chatModel: ChatModel,
) : LlmClient {
    private val logger = KotlinLogging.logger {}

    override fun complete(prompt: String): String {
        if (prompt.isBlank()) {
            return ""
        }

        return runCatching {
            val rawText =
                chatModel
                    .chat(UserMessage.from(prompt))
                    .aiMessage()
                    .text()
                    .orEmpty()
            LlmOutputParser.cleanLlmContent(rawText)
        }.getOrElse { error ->
            logger.warn(error) { "ChatModel completion failed" }
            ""
        }
    }
}

object LlmClientFactory {
    private val logger = KotlinLogging.logger {}
    private const val DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
    private const val DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
    private const val DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
    private const val DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    private const val DEFAULT_OPENAI_COMPATIBLE_MODEL = "deepseek-chat"
    private const val DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://api.deepseek.com"

    fun fromEnvironment(env: Map<String, String> = System.getenv()): LlmClient {
        val provider = env.firstNonBlank("OPENAI_PROVIDER", "LLM_PROVIDER")?.lowercase() ?: "openai"
        return when (provider) {
            "ollama" -> {
                createOllamaClient(env)
            }

            "azure",
            "azure_openai",
            -> {
                createAzureOpenAiCompatibleClient(env)
            }

            "openai" -> {
                createOpenAiClient(
                    env = env,
                    defaultModel = DEFAULT_OPENAI_MODEL,
                    defaultBaseUrl = DEFAULT_OPENAI_BASE_URL,
                )
            }

            "gemini", "google", "google_gemini" -> {
                createGeminiClient(env)
            }

            "openai_compatible",
            "deepseek",
            -> {
                createOpenAiClient(
                    env = env,
                    defaultModel = DEFAULT_OPENAI_COMPATIBLE_MODEL,
                    defaultBaseUrl = DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
                )
            }

            else -> {
                logger.warn { "Unsupported OPENAI_PROVIDER='$provider'. Falling back to OpenAI-compatible mode." }
                createOpenAiClient(
                    env = env,
                    defaultModel = DEFAULT_OPENAI_COMPATIBLE_MODEL,
                    defaultBaseUrl = DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
                )
            }
        }
    }

    private fun createOpenAiClient(
        env: Map<String, String>,
        defaultModel: String,
        defaultBaseUrl: String,
        apiKeyEnvKeys: Array<String> = arrayOf("LLM_API_KEY", "OPENAI_API_KEY"),
    ): LlmClient {
        val apiKey = env.firstNonBlank(*apiKeyEnvKeys) ?: return NoopLlmClient()
        val modelName = env.firstNonBlank("LLM_MODEL", "OPENAI_MODEL") ?: defaultModel
        val baseUrl = env.firstNonBlank("LLM_BASE_URL", "OPENAI_BASE_URL") ?: defaultBaseUrl

        val builder =
            OpenAiChatModel
                .builder()
                .apiKey(apiKey)
                .modelName(modelName)
                .baseUrl(baseUrl)

        env["LLM_TEMPERATURE"]?.toDoubleOrNull()?.let(builder::temperature)
            ?: builder.temperature(0.3)

        return runCatching {
            ChatModelLlmClient(builder.build())
        }.getOrElse { error ->
            logger.warn(error) { "Failed to initialize OpenAI ChatModel. Falling back to NoopLlmClient." }
            NoopLlmClient()
        }
    }

    private fun createAzureOpenAiCompatibleClient(env: Map<String, String>): LlmClient {
        val apiKey =
            env.firstNonBlank(
                "LLM_API_KEY",
                "AZURE_OPENAI_API_KEY",
                "OPENAI_API_KEY",
            ) ?: return NoopLlmClient()
        val deploymentName = env.firstNonBlank("LLM_MODEL", "OPENAI_MODEL", "AZURE_OPENAI_DEPLOYMENT") ?: return NoopLlmClient()
        val endpoint = env.firstNonBlank("LLM_BASE_URL", "AZURE_OPENAI_ENDPOINT", "OPENAI_BASE_URL") ?: return NoopLlmClient()
        val apiVersion = env.firstNonBlank("API_VERSION", "AZURE_OPENAI_API_VERSION") ?: "2025-01-01-preview"

        val baseUrl = "${endpoint.trimEnd('/')}/openai/deployments/$deploymentName"
        val builder =
            OpenAiChatModel
                .builder()
                .apiKey(apiKey)
                .modelName(deploymentName)
                .baseUrl(baseUrl)
                .customHeaders(mapOf("api-key" to apiKey))
                .customQueryParams(mapOf("api-version" to apiVersion))

        env["LLM_TEMPERATURE"]?.toDoubleOrNull()?.let(builder::temperature)
            ?: builder.temperature(0.3)

        return runCatching {
            ChatModelLlmClient(builder.build())
        }.getOrElse { error ->
            logger.warn(error) { "Failed to initialize Azure OpenAI-compatible ChatModel. Falling back to NoopLlmClient." }
            NoopLlmClient()
        }
    }

    private fun createOllamaClient(env: Map<String, String>): LlmClient {
        val modelName = env.firstNonBlank("LLM_MODEL", "OLLAMA_MODEL") ?: "llama3"
        val baseUrl = env.firstNonBlank("LLM_BASE_URL", "OLLAMA_BASE_URL") ?: "http://localhost:11434"

        val builder =
            OllamaChatModel
                .builder()
                .baseUrl(baseUrl)
                .modelName(modelName)

        env["LLM_TEMPERATURE"]?.toDoubleOrNull()?.let(builder::temperature)
            ?: builder.temperature(0.3)

        return runCatching {
            ChatModelLlmClient(builder.build())
        }.getOrElse { error ->
            logger.warn(error) { "Failed to initialize Ollama ChatModel. Falling back to NoopLlmClient." }
            NoopLlmClient()
        }
    }

    private fun createGeminiClient(env: Map<String, String>): LlmClient {
        val apiKey = env.firstNonBlank("LLM_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY") ?: return NoopLlmClient()
        val modelName = env.firstNonBlank("LLM_MODEL", "OPENAI_MODEL") ?: DEFAULT_GEMINI_MODEL
        val baseUrl = env.firstNonBlank("LLM_BASE_URL", "OPENAI_BASE_URL") ?: DEFAULT_GEMINI_BASE_URL

        val builder =
            GoogleAiGeminiChatModel
                .builder()
                .apiKey(apiKey)
                .modelName(modelName)
                .baseUrl(baseUrl)

        env["LLM_TEMPERATURE"]?.toDoubleOrNull()?.let(builder::temperature)
            ?: builder.temperature(0.3)

        return runCatching {
            ChatModelLlmClient(builder.build())
        }.getOrElse { error ->
            logger.warn(error) { "Failed to initialize Gemini ChatModel. Falling back to NoopLlmClient." }
            NoopLlmClient()
        }
    }

    private fun Map<String, String>.firstNonBlank(vararg keys: String): String? =
        keys
            .asSequence()
            .mapNotNull { key -> this[key]?.trim() }
            .firstOrNull { value -> value.isNotBlank() }
}
