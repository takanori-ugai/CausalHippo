package com.microsoft.graphrag.index

import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.openai.OpenAiChatModel
import io.github.oshai.kotlinlogging.KotlinLogging
import tools.jackson.databind.ObjectMapper
import tools.jackson.databind.json.JsonMapper
import tools.jackson.module.kotlin.KotlinModule

/**
 * Extracts claims from document chunks using an LLM.
 *
 * @property chatModel Chat model used for claim extraction.
 * @property prompts Repository that provides prompt templates.
 * @property objectMapper Mapper used to parse JSON responses.
 */
class ClaimsWorkflow(
    private val chatModel: OpenAiChatModel,
    private val prompts: PromptRepository = PromptRepository(),
    private val objectMapper: ObjectMapper =
        JsonMapper
            .builder()
            .addModule(KotlinModule.Builder().build())
            .findAndAddModules()
            .build(),
) {
    /**
     * Extracts claims from the provided document chunks.
     *
     * @param chunks Document chunks to process.
     * @param entitySpecs Entity type spec string inserted into the prompt.
     * @param claimDescription Claim description inserted into the prompt.
     * @return List of extracted claims.
     */
    fun extractClaims(
        chunks: List<DocumentChunk>,
        entitySpecs: String = "ORGANIZATION,PERSON,GPE",
        claimDescription: String = "Any claims or facts that could be relevant to information discovery.",
    ): List<Claim> {
        val claims = mutableListOf<Claim>()
        val basePrompt = prompts.loadExtractClaimsPrompt()
        for (chunk in chunks) {
            val prompt =
                basePrompt
                    .replace("{entity_specs}", entitySpecs)
                    .replace("{claim_description}", claimDescription)
                    .replace("{input_text}", chunk.text)
            val extracted =
                runCatching { extractClaimsJson(prompt) }
                    .getOrElse { throw IllegalStateException("Claim extraction failed for chunk ${chunk.id}", it) }
            logger.debug { "Claims parsed for chunk ${chunk.id}: ${extracted.size}" }
            claims += extracted
        }
        return claims
    }

    @Suppress("ReturnCount")
    private fun extractClaimsJson(prompt: String): List<Claim> {
        val messages =
            listOf<ChatMessage>(
                UserMessage(prompt),
            )
        val response = chat(messages)
        val json = checkNotNull(extractJsonArray(response)) { "Missing claim extraction JSON array" }
        return runCatching {
            objectMapper.readValue(json, Array<Claim>::class.java).toList()
        }.getOrElse {
            throw IllegalStateException("Invalid claim extraction JSON", it)
        }
    }

    private fun extractJsonArray(text: String): String? {
        val start = text.indexOf('[')
        val end = text.lastIndexOf(']')
        if (start == -1 || end == -1 || end <= start) return null
        return text.substring(start, end + 1)
    }

    private fun chat(messages: List<ChatMessage>): String =
        runCatching { chatModel.chat(messages).aiMessage().text() }
            .getOrElse {
                throw IllegalStateException("Claim extraction chat failed", it)
            }

    companion object {
        private val logger = KotlinLogging.logger {}
    }
}
