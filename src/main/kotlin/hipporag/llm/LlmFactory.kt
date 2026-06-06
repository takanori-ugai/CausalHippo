package hipporag.llm

import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.model.openai.OpenAiChatRequestParameters
import dev.langchain4j.model.openaiofficial.OpenAiOfficialChatModel
import hipporag.config.BaseConfig
import io.github.oshai.kotlinlogging.KotlinLogging
import shared.llm.supportsTemperature

private val logger = KotlinLogging.logger {}
private const val DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

/**
 * Builds an LLM client from [globalConfig].
 */
fun getLlm(globalConfig: BaseConfig): BaseLLM {
    val provider = globalConfig.llmProvider?.lowercase()
    val modelName = globalConfig.llmName
    val temperature = globalConfig.temperature

    val unsupportedProviders =
        setOf(
            "azure",
            "bedrock",
            "bedrock_llm",
            "aws_bedrock",
            "transformers",
            "transformers_llm",
            "transformers_offline",
            "transformers-offline",
            "vllm",
            "vllm_offline",
            "vllm-offline",
        )

    if (provider in unsupportedProviders) {
        error(
            "LLM provider '$provider' is not supported in the Kotlin port yet. " +
                "Supported providers: OpenAI-compatible (default)," +
                "and Ollama (set llmProvider=ollama or use an Ollama base URL).",
        )
    } else if (provider != null && provider !in setOf("openai", "ollama", "gemini", "google", "google_gemini")) {
        logger.warn { "Unknown LLM provider '$provider'. Falling back to OpenAI-compatible settings." }
    }

    return when {
        provider == "azure" -> {
            error(
                "Azure OpenAI chat is not supported by this factory after the SDK switch. " +
                    "Restore Azure handling or use llmProvider=openai/ollama with matching settings.",
            )
        }

        provider == "ollama" || (globalConfig.llmBaseUrl?.contains("ollama") == true) -> {
            val baseUrl = globalConfig.ollamaBaseUrl ?: globalConfig.llmBaseUrl ?: "http://localhost:11434"
            val ollamaModel = globalConfig.ollamaModelName ?: modelName
            val model =
                OllamaChatModel
                    .builder()
                    .baseUrl(baseUrl)
                    .modelName(ollamaModel)
                    .temperature(temperature)
                    .build()
            LangChainChatLLM(model)
        }

        provider in setOf("gemini", "google", "google_gemini") -> {
            val apiKey =
                globalConfig.openAiApiKey
                    ?: System.getenv("GEMINI_API_KEY")
                    ?: System.getenv("OPENAI_API_KEY")
                    ?: error("Gemini API key not configured. Set GEMINI_API_KEY/OPENAI_API_KEY or openAiApiKey in config.")
            val includeTemperature = supportsTemperature(modelName)
            val builder =
                GoogleAiGeminiChatModel
                    .builder()
                    .apiKey(apiKey)
                    .modelName(modelName)
                    .baseUrl(globalConfig.llmBaseUrl ?: DEFAULT_GEMINI_BASE_URL)
            if (includeTemperature) {
                builder.temperature(temperature)
            }
            if (globalConfig.maxNewTokens != null) {
                builder.maxOutputTokens(globalConfig.maxNewTokens)
            }
            val model = builder.build()
            LangChainChatLLM(model)
        }

        else -> {
            val apiKey =
                globalConfig.openAiApiKey ?: System.getenv("OPENAI_API_KEY")
                    ?: error("OpenAI API key not configured. Set openAiApiKey in config or OPENAI_API_KEY env var.")
            val model =
                buildOpenAiChatModel(
                    globalConfig = globalConfig,
                    apiKey = apiKey,
                    includeTemperature = supportsTemperature(modelName),
                )
            LangChainChatLLM(model)
        }
    }
}

private fun buildOpenAiChatModel(
    globalConfig: BaseConfig,
    apiKey: String,
    includeTemperature: Boolean,
): OpenAiOfficialChatModel {
    val requestParametersBuilder =
        OpenAiChatRequestParameters
            .builder()
            .modelName(globalConfig.llmName)
            .also { builder ->
                if (includeTemperature) {
                    builder.temperature(globalConfig.temperature)
                }
            }
    val builder =
        OpenAiOfficialChatModel
            .builder()
            .apiKey(apiKey)
            .modelName(globalConfig.llmName)
            .also { chatBuilder ->
                if (includeTemperature) {
                    chatBuilder.temperature(globalConfig.temperature)
                }
                if (globalConfig.maxNewTokens != null) {
                    chatBuilder.maxCompletionTokens(globalConfig.maxNewTokens)
                }
            }
    if (globalConfig.llmBaseUrl != null) {
        builder.baseUrl(globalConfig.llmBaseUrl)
    }
    if (globalConfig.maxNewTokens != null) {
        requestParametersBuilder.maxCompletionTokens(globalConfig.maxNewTokens)
    }
    builder.defaultRequestParameters(requestParametersBuilder.build())
    return builder.build()
}
