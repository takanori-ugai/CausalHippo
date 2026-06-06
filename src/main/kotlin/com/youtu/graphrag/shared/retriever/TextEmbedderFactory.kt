package com.youtu.graphrag.shared.retriever

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.youtu.graphrag.shared.config.ConfigManager
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel
import io.github.oshai.kotlinlogging.KotlinLogging
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration

class OpenAiTextEmbedder(
    private val modelName: String,
    private val baseUrl: String,
    private val apiKey: String,
    override val dimensions: Int,
    private val fallback: TextEmbedder,
    private val timeout: Duration = Duration.ofSeconds(30),
    private val httpClient: HttpClient = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build(),
    private val objectMapper: ObjectMapper = ObjectMapper().registerKotlinModule(),
) : TextEmbedder {
    private val logger = KotlinLogging.logger {}
    override val modelTag: String = "openai:$modelName:$dimensions@${baseUrl.trimEnd('/')}"

    override fun embed(text: String): FloatArray {
        if (text.isBlank()) {
            return FloatArray(dimensions)
        }

        return runCatching {
            val payload =
                linkedMapOf<String, Any>(
                    "model" to modelName,
                    "input" to text,
                )
            val body = objectMapper.writeValueAsString(payload)
            val endpoint = toEmbeddingsEndpoint(baseUrl)
            val request =
                HttpRequest
                    .newBuilder()
                    .uri(URI.create(endpoint))
                    .timeout(timeout)
                    .header("Authorization", "Bearer $apiKey")
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build()
            val response = httpClient.send(request, HttpResponse.BodyHandlers.ofString())
            if (response.statusCode() !in 200..299) {
                error("Embedding endpoint returned status=${response.statusCode()}")
            }
            parseEmbedding(response.body())
        }.getOrElse { error ->
            logger.warn(error) { "OpenAI-compatible embedding request failed. Falling back to hash embedder." }
            fallback.embed(text)
        }
    }

    private fun parseEmbedding(responseBody: String): FloatArray {
        val root = objectMapper.readTree(responseBody)
        val data = root.path("data")
        if (!data.isArray || data.isEmpty) {
            error("Embedding response contains empty data field")
        }
        val first = data.first()
        val embeddingNode = first.path("embedding")
        if (!embeddingNode.isArray || embeddingNode.isEmpty) {
            error("Embedding response contains empty embedding vector")
        }

        val vector = FloatArray(embeddingNode.size())
        for (index in 0 until embeddingNode.size()) {
            vector[index] = embeddingNode[index].asDouble().toFloat()
        }
        return normalizeDimensions(vector)
    }

    private fun normalizeDimensions(vector: FloatArray): FloatArray {
        if (vector.size == dimensions) {
            return vector
        }
        if (vector.size > dimensions) {
            return vector.copyOf(dimensions)
        }

        val normalized = FloatArray(dimensions)
        vector.copyInto(normalized)
        return normalized
    }

    private fun toEmbeddingsEndpoint(rawBaseUrl: String): String {
        val trimmed = rawBaseUrl.trimEnd('/')
        return if (trimmed.endsWith("/embeddings")) trimmed else "$trimmed/embeddings"
    }
}

class GeminiTextEmbedder(
    private val modelName: String,
    private val baseUrl: String,
    private val apiKey: String,
    override val dimensions: Int,
    private val fallback: TextEmbedder,
    private val timeout: Duration = Duration.ofSeconds(30),
) : TextEmbedder {
    private val logger = KotlinLogging.logger {}
    private val delegate =
        GoogleAiEmbeddingModel
            .builder()
            .modelName(modelName)
            .apiKey(apiKey)
            .baseUrl(baseUrl)
            .timeout(timeout)
            .logRequests(true)
            .build()

    override val modelTag: String = "gemini:$modelName:$dimensions@${baseUrl.trimEnd('/')}"

    override fun embed(text: String): FloatArray {
        if (text.isBlank()) {
            return FloatArray(dimensions)
        }

        return runCatching {
            val vector = delegate.embed(text).content().vector()
            normalizeDimensions(vector)
        }.getOrElse { error ->
            logger.warn(error) { "Gemini embedding request failed. Falling back to hash embedder." }
            fallback.embed(text)
        }
    }

    private fun normalizeDimensions(vector: FloatArray): FloatArray {
        if (vector.size == dimensions) {
            return vector
        }
        if (vector.size > dimensions) {
            return vector.copyOf(dimensions)
        }

        val normalized = FloatArray(dimensions)
        vector.copyInto(normalized)
        return normalized
    }
}

object RetrieverEmbedderFactory {
    private val logger = KotlinLogging.logger {}
    private const val DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

    fun fromConfig(
        config: ConfigManager,
        env: Map<String, String> = System.getenv(),
    ): TextEmbedder {
        val modelName = env.firstNonBlank("EMBEDDING_MODEL") ?: config.embeddings.modelName
        val dimensions =
            env.firstNonBlank("EMBEDDING_DIMENSIONS")?.toIntOrNull()?.coerceAtLeast(1)
                ?: config.embeddings.dimensions.coerceAtLeast(1)
        val fallback = HashTextEmbedder(dimensions = dimensions, modelName = modelName)
        val provider = (env.firstNonBlank("EMBEDDING_PROVIDER") ?: config.embeddings.provider).lowercase()

        return when (provider) {
            "openai",
            "openai_compatible",
            -> createOpenAiEmbedder(config, env, modelName, dimensions, fallback)

            "gemini",
            "google",
            "google_gemini",
            -> createGeminiEmbedder(config, env, modelName, dimensions, fallback)

            else -> fallback
        }
    }

    private fun createOpenAiEmbedder(
        config: ConfigManager,
        env: Map<String, String>,
        modelName: String,
        dimensions: Int,
        fallback: TextEmbedder,
    ): TextEmbedder {
        val apiKey = env.firstNonBlank("EMBEDDING_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY")
        if (apiKey.isNullOrBlank()) {
            logger.warn { "EMBEDDING_PROVIDER=openai but no embedding API key was configured. Using hash embedder fallback." }
            return fallback
        }

        val baseUrl =
            env.firstNonBlank("EMBEDDING_BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL")
                ?: if ((env.firstNonBlank("EMBEDDING_PROVIDER") ?: config.embeddings.provider).lowercase() in
                    setOf("gemini", "google", "google_gemini")
                ) {
                    DEFAULT_GEMINI_BASE_URL
                } else {
                    config.embeddings.baseUrl
                }

        return OpenAiTextEmbedder(
            modelName = modelName,
            baseUrl = baseUrl,
            apiKey = apiKey,
            dimensions = dimensions,
            fallback = fallback,
        )
    }

    private fun createGeminiEmbedder(
        config: ConfigManager,
        env: Map<String, String>,
        modelName: String,
        dimensions: Int,
        fallback: TextEmbedder,
    ): TextEmbedder {
        val apiKey = env.firstNonBlank("EMBEDDING_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY")
        if (apiKey.isNullOrBlank()) {
            logger.warn { "EMBEDDING_PROVIDER=gemini but no embedding API key was configured. Using hash embedder fallback." }
            return fallback
        }

        val baseUrl =
            env.firstNonBlank("EMBEDDING_BASE_URL", "LLM_BASE_URL", "OPENAI_BASE_URL")
                ?: config.embeddings.baseUrl.ifBlank { DEFAULT_GEMINI_BASE_URL }

        return GeminiTextEmbedder(
            modelName = modelName,
            baseUrl = baseUrl,
            apiKey = apiKey,
            dimensions = dimensions,
            fallback = fallback,
        )
    }

    private fun Map<String, String>.firstNonBlank(vararg keys: String): String? =
        keys
            .asSequence()
            .mapNotNull { key -> this[key]?.trim() }
            .firstOrNull { value -> value.isNotBlank() }
}
