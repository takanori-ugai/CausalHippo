package hipporag.embeddingmodel

import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel
import dev.langchain4j.model.ollama.OllamaEmbeddingModel
import dev.langchain4j.model.openaiofficial.OpenAiOfficialEmbeddingModel
import hipporag.config.BaseConfig
import io.github.oshai.kotlinlogging.KotlinLogging
import java.net.URI

/**
 * Builds LangChain4j-backed embedding models based on [BaseConfig].
 */
class LangChainEmbeddingFactory : EmbeddingModelFactory {
    private val logger = KotlinLogging.logger {}
    private val defaultGeminiBaseUrl = "https://generativelanguage.googleapis.com/v1beta/openai"

    /**
     * Creates a LangChain4j embedding model wrapped in [BaseEmbeddingModel].
     */
    override fun create(
        globalConfig: BaseConfig,
        embeddingModelName: String,
    ): BaseEmbeddingModel {
        val model = buildEmbeddingModel(globalConfig, embeddingModelName)
        return LangChainEmbeddingModel(model)
    }

    private fun buildEmbeddingModel(
        globalConfig: BaseConfig,
        embeddingModelName: String,
    ): EmbeddingModel {
        val provider = globalConfig.embeddingProvider?.lowercase()
        val unsupportedProviders =
            setOf(
                "cohere",
                "contriever",
                "gritlm",
                "nvembedv2",
                "nv-embed-v2",
                "transformers",
                "vllm",
                "vllm_offline",
                "vllm-offline",
                "huggingface",
                "azure",
            )

        if (provider in unsupportedProviders) {
            error(
                "Embedding provider '$provider' is not supported in the Kotlin port yet. " +
                    "Supported providers: OpenAI-compatible (default), " +
                    "and Ollama (set embeddingProvider=ollama or use an Ollama base URL).",
            )
        } else if (provider != null && provider !in setOf("openai", "ollama", "gemini", "google", "google_gemini")) {
            logger.warn { "Unknown embedding provider '$provider'. Falling back to OpenAI-compatible settings." }
        }
        return when {
            provider == "azure" -> {
                error(
                    "Azure OpenAI embeddings are not supported by this factory after the SDK switch. " +
                        "Restore Azure handling or use embeddingProvider=openai/ollama with matching settings.",
                )
            }

            provider == "ollama" -> {
                val baseUrl =
                    globalConfig.ollamaBaseUrl
                        ?: globalConfig.embeddingBaseUrl
                        ?: "http://localhost:11434"
                val model = globalConfig.ollamaEmbeddingModelName ?: embeddingModelName
                OllamaEmbeddingModel
                    .builder()
                    .baseUrl(baseUrl)
                    .modelName(model)
                    .build()
            }

            provider in setOf("gemini", "google", "google_gemini") -> {
                val apiKey =
                    globalConfig.openAiApiKey
                        ?: System.getenv("GEMINI_API_KEY")
                        ?: System.getenv("OPENAI_API_KEY")
                        ?: error("Gemini API key not found. Set GEMINI_API_KEY/OPENAI_API_KEY or openAiApiKey in config.")
                val builder =
                    GoogleAiEmbeddingModel
                        .builder()
                        .apiKey(apiKey)
                        .modelName(embeddingModelName)
                builder.baseUrl(globalConfig.embeddingBaseUrl ?: defaultGeminiBaseUrl)
                builder.build()
            }

            isLikelyOllamaBaseUrl(globalConfig.embeddingBaseUrl) -> {
                logger.warn {
                    "Ambiguous embedding provider. 'ollama' detected in embeddingBaseUrl, " +
                        "but embeddingProvider is not explicitly 'ollama'. Defaulting to Ollama."
                }
                val baseUrl =
                    globalConfig.ollamaBaseUrl
                        ?: globalConfig.embeddingBaseUrl
                        ?: "http://localhost:11434"
                val model = globalConfig.ollamaEmbeddingModelName ?: embeddingModelName
                OllamaEmbeddingModel
                    .builder()
                    .baseUrl(baseUrl)
                    .modelName(model)
                    .build()
            }

            else -> {
                val apiKey =
                    globalConfig.openAiApiKey ?: System.getenv("OPENAI_API_KEY")
                        ?: error("OpenAI API key not found. Set openAiApiKey in config or OPENAI_API_KEY env var.")
                val builder =
                    OpenAiOfficialEmbeddingModel
                        .builder()
                        .apiKey(apiKey)
                        .modelName(embeddingModelName)
                if (globalConfig.embeddingBaseUrl != null) {
                    builder.baseUrl(globalConfig.embeddingBaseUrl)
                }
                builder.build()
            }
        }
    }

    private fun isLikelyOllamaBaseUrl(baseUrl: String?): Boolean {
        if (baseUrl.isNullOrBlank()) return false
        return runCatching {
            val uri = URI(baseUrl)
            uri.port == 11434
        }.getOrDefault(false)
    }
}
