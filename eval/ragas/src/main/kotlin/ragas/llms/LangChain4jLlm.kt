package ragas.llms

import dev.langchain4j.data.message.ImageContent
import dev.langchain4j.data.message.TextContent
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.request.ChatRequest
import dev.langchain4j.service.AiServices
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import ragas.prompt.PromptContentPart
import ragas.runtime.RunConfig

/**
 * BaseRagasLlm adapter for LangChain4j chat models.
 *
 * @property model Underlying model instance.
 * @property runConfig Runtime configuration.
 */
class LangChain4jLlm(
    private val model: ChatModel,
    override var runConfig: RunConfig = RunConfig(),
) : BaseRagasLlm,
    StructuredOutputRagasLlm,
    MultiModalRagasLlm {
    private val numericService: NumericStructuredService by lazy {
        AiServices.create(NumericStructuredService::class.java, model)
    }
    private val discreteService: DiscreteStructuredService by lazy {
        AiServices.create(DiscreteStructuredService::class.java, model)
    }
    private val rankingService: RankingStructuredService by lazy {
        AiServices.create(RankingStructuredService::class.java, model)
    }

    /**
     * Generates one or more text completions for the supplied prompt.
     */
    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult = generateContent(listOf(PromptContentPart.Text(prompt)), n, temperature, stop)

    /**
     * Generates and parses a numeric value from model output.
     */
    override suspend fun generateNumericValue(prompt: String): Double? =
        withTimeout(runConfig.timeoutSeconds * 1_000) {
            withContext(Dispatchers.IO) {
                numericService.evaluate(prompt).value
            }
        }

    /**
     * Generates and parses one label from a fixed set of choices.
     */
    override suspend fun generateDiscreteValue(prompt: String): String? =
        withTimeout(runConfig.timeoutSeconds * 1_000) {
            withContext(Dispatchers.IO) {
                discreteService.evaluate(prompt).value
            }
        }

    /**
     * Generates and parses ranked items from model output.
     */
    override suspend fun generateRankingItems(prompt: String): List<String>? =
        withTimeout(runConfig.timeoutSeconds * 1_000) {
            withContext(Dispatchers.IO) {
                rankingService.evaluate(prompt)?.items
            }
        }

    /**
     * Generates output from multimodal prompt content parts.
     */
    override suspend fun generateContent(
        content: List<PromptContentPart>,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult =
        coroutineScope {
            require(n > 0) { "n must be greater than zero." }

            val langchain4jContent =
                content.map { part ->
                    when (part) {
                        is PromptContentPart.Text -> TextContent.from(part.text)
                        is PromptContentPart.ImageDataUri -> ImageContent.from(part.dataUri)
                        is PromptContentPart.ImageUrl -> ImageContent.from(part.url)
                    }
                }
            val userMessage = UserMessage.from(langchain4jContent)

            val jobs =
                (0 until n).map {
                    async {
                        withTimeout(runConfig.timeoutSeconds * 1_000) {
                            withContext(Dispatchers.IO) {
                                val requestBuilder =
                                    ChatRequest
                                        .builder()
                                        .messages(userMessage)
                                if (temperature != null) {
                                    requestBuilder.temperature(temperature)
                                }
                                if (!stop.isNullOrEmpty()) {
                                    requestBuilder.stopSequences(stop)
                                }
                                val response = model.chat(requestBuilder.build())
                                val text = response.aiMessage().text()
                                val finish = response.finishReason()?.name
                                LlmGeneration(
                                    text = applyStop(text, stop),
                                    finishReason = finish,
                                )
                            }
                        }
                    }
                }

            LlmResult(generations = jobs.awaitAll())
        }

    private fun applyStop(
        text: String,
        stop: List<String>?,
    ): String {
        if (stop.isNullOrEmpty()) {
            return text
        }

        val firstMatchIndex =
            stop
                .map { marker -> text.indexOf(marker) }
                .filter { index -> index >= 0 }
                .minOrNull()

        return if (firstMatchIndex == null) text else text.substring(0, firstMatchIndex)
    }

    private interface NumericStructuredService {
        fun evaluate(userMessage: String): NumericStructuredOutput
    }

    private interface DiscreteStructuredService {
        fun evaluate(userMessage: String): DiscreteStructuredOutput
    }

    private interface RankingStructuredService {
        fun evaluate(userMessage: String): RankingStructuredOutput?
    }

    private data class NumericStructuredOutput(
        val value: Double? = null,
    )

    private data class DiscreteStructuredOutput(
        val value: String? = null,
    )

    private data class RankingStructuredOutput(
        val items: List<String> = emptyList(),
    )
}
