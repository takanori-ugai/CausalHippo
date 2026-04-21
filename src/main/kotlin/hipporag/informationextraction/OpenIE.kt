package hipporag.informationextraction

import hipporag.llm.BaseLLM
import hipporag.prompts.PromptTemplateManager
import hipporag.utils.EmbeddingRow
import hipporag.utils.NerRawOutput
import hipporag.utils.TripleRawOutput
import hipporag.utils.extractJsonObjectWithKey
import hipporag.utils.filterInvalidTriples
import hipporag.utils.jsonWithDefaults
import io.github.oshai.kotlinlogging.KLogger
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Common interface for OpenIE extraction implementations.
 */
interface OpenIEBase {
    /**
     * Extracts named entities and triples for each row keyed by chunk ID.
     */
    fun batchOpenie(rows: Map<String, EmbeddingRow>): Pair<Map<String, NerRawOutput>, Map<String, TripleRawOutput>>
}

/**
 * Online OpenIE extractor that calls an LLM for NER and triple extraction.
 */
class OpenIE(
    private val llmModel: BaseLLM,
) : OpenIEBase {
    private val logger = KotlinLogging.logger {}
    private val promptTemplateManager =
        PromptTemplateManager(
            roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
        )
    private val json = jsonWithDefaults { ignoreUnknownKeys = true }

    private fun formatMessagesForLog(messages: List<hipporag.utils.Message>): String =
        messages.joinToString(separator = "\n") { msg ->
            "[${msg.role}] ${msg.content}"
        }

    private fun ner(
        chunkKey: String,
        passage: String,
    ): NerRawOutput {
        val messages = promptTemplateManager.render("ner", mapOf("passage" to passage))
        if (logger.isDebugEnabled()) {
            logger.debug { "OpenIE NER prompt for chunk $chunkKey:\n${formatMessagesForLog(messages)}" }
        }
        return safeExtract(
            logger = logger,
            chunkKey = chunkKey,
            operation = "NER",
            fallback = { e ->
                NerRawOutput(
                    chunkId = chunkKey,
                    response = null,
                    uniqueEntities = emptyList(),
                    metadata = mapOf("error" to e.message.orEmpty()),
                )
            },
        ) {
            val result = llmModel.infer(messages)
            if (logger.isDebugEnabled()) {
                logger.debug { "OpenIE NER response for chunk $chunkKey:\n${result.response}" }
            }
            val entities = extractNamedEntitiesFromResponse(result.response, json)
            NerRawOutput(
                chunkId = chunkKey,
                response = result.response,
                uniqueEntities = entities.distinct(),
                metadata = result.metadata,
            )
        }
    }

    private fun tripleExtraction(
        chunkKey: String,
        passage: String,
        namedEntities: List<String>,
    ): TripleRawOutput {
        val namedEntityJson = buildNamedEntityJson(namedEntities, json)
        val messages =
            promptTemplateManager.render(
                "triple_extraction",
                mapOf("passage" to passage, "named_entity_json" to namedEntityJson),
            )
        if (logger.isDebugEnabled()) {
            logger.debug { "OpenIE triple extraction prompt for chunk $chunkKey:\n${formatMessagesForLog(messages)}" }
        }
        return safeExtract(
            logger = logger,
            chunkKey = chunkKey,
            operation = "Triple extraction",
            fallback = { e ->
                TripleRawOutput(
                    chunkId = chunkKey,
                    response = null,
                    triples = emptyList(),
                    metadata = mapOf("error" to e.message.orEmpty()),
                )
            },
        ) {
            val result = llmModel.infer(messages)
            if (logger.isDebugEnabled()) {
                logger.debug { "OpenIE triple extraction response for chunk $chunkKey:\n${result.response}" }
            }
            val triples = extractTriplesFromResponse(result.response, json)
            TripleRawOutput(
                chunkId = chunkKey,
                response = result.response,
                triples = filterInvalidTriples(triples),
                metadata = result.metadata,
            )
        }
    }

    /**
     * Runs NER followed by triple extraction for each provided row.
     */
    override fun batchOpenie(rows: Map<String, EmbeddingRow>): Pair<Map<String, NerRawOutput>, Map<String, TripleRawOutput>> {
        val nerResults = mutableMapOf<String, NerRawOutput>()
        val tripleResults = mutableMapOf<String, TripleRawOutput>()

        for ((chunkKey, row) in rows) {
            val nerOutput = ner(chunkKey, row.content)
            nerResults[chunkKey] = nerOutput
            val tripleOutput = tripleExtraction(chunkKey, row.content, nerOutput.uniqueEntities)
            tripleResults[chunkKey] = tripleOutput
        }

        return nerResults to tripleResults
    }
}

private fun twoPhaseOpenie(
    llmModel: BaseLLM,
    logger: io.github.oshai.kotlinlogging.KLogger,
    promptTemplateManager: PromptTemplateManager,
    json: Json,
    rows: Map<String, EmbeddingRow>,
): Pair<Map<String, NerRawOutput>, Map<String, TripleRawOutput>> {
    val nerResults = mutableMapOf<String, NerRawOutput>()
    val tripleResults = mutableMapOf<String, TripleRawOutput>()

    val namedEntitiesByChunk = mutableMapOf<String, List<String>>()
    for ((chunkKey, row) in rows) {
        val messages = promptTemplateManager.render("ner", mapOf("passage" to row.content))
        val nerOutput =
            safeExtract(
                logger = logger,
                chunkKey = chunkKey,
                operation = "NER",
                fallback = { e ->
                    NerRawOutput(
                        chunkId = chunkKey,
                        response = null,
                        uniqueEntities = emptyList(),
                        metadata = mapOf("error" to e.message.orEmpty()),
                    ) to emptyList()
                },
            ) {
                val result = llmModel.infer(messages)
                val entities = extractNamedEntitiesFromResponse(result.response, json)
                if (entities.isEmpty()) {
                    logger.warn { "No entities extracted for chunk_id: $chunkKey" }
                }
                NerRawOutput(
                    chunkId = chunkKey,
                    response = result.response,
                    uniqueEntities = entities.distinct(),
                    metadata = result.metadata,
                ) to entities
            }
        nerResults[chunkKey] = nerOutput.first
        namedEntitiesByChunk[chunkKey] = nerOutput.second
    }

    for ((chunkKey, row) in rows) {
        val namedEntities = namedEntitiesByChunk[chunkKey] ?: emptyList()
        val namedEntityJson = buildNamedEntityJson(namedEntities, json)
        val messages =
            promptTemplateManager.render(
                "triple_extraction",
                mapOf("passage" to row.content, "named_entity_json" to namedEntityJson),
            )
        val tripleOutput =
            safeExtract(
                logger = logger,
                chunkKey = chunkKey,
                operation = "Triple extraction",
                fallback = { e ->
                    TripleRawOutput(
                        chunkId = chunkKey,
                        response = null,
                        triples = emptyList(),
                        metadata = mapOf("error" to e.message.orEmpty()),
                    )
                },
            ) {
                val result = llmModel.infer(messages)
                val triples = extractTriplesFromResponse(result.response, json)
                if (triples.isEmpty()) {
                    logger.warn { "No triples extracted for chunk_id: $chunkKey" }
                }
                TripleRawOutput(
                    chunkId = chunkKey,
                    response = result.response,
                    triples = filterInvalidTriples(triples),
                    metadata = result.metadata,
                )
            }
        tripleResults[chunkKey] = tripleOutput
    }

    return nerResults to tripleResults
}

open class OfflineOpenIEBase(
    private val llmModel: BaseLLM,
) : OpenIEBase {
    private val logger = KotlinLogging.logger {}
    private val promptTemplateManager =
        PromptTemplateManager(
            roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
        )
    private val json = jsonWithDefaults { ignoreUnknownKeys = true }

    /**
     * Runs two-phase OpenIE extraction for each provided row.
     */
    override fun batchOpenie(rows: Map<String, EmbeddingRow>): Pair<Map<String, NerRawOutput>, Map<String, TripleRawOutput>> =
        twoPhaseOpenie(llmModel, logger, promptTemplateManager, json, rows)
}

/**
 * Offline OpenIE extractor that performs two-phase extraction with a local model.
 */
class VllmOfflineOpenIE(
    llmModel: BaseLLM,
) : OfflineOpenIEBase(llmModel)

/**
 * Offline OpenIE extractor that targets transformer-based local models.
 *
 * Extension point: add model-specific prompt setup or decoding behavior here.
 */
class TransformersOfflineOpenIE(
    llmModel: BaseLLM,
) : OfflineOpenIEBase(llmModel)

private fun buildNamedEntityJson(
    namedEntities: List<String>,
    json: Json,
): String =
    json.encodeToString(
        JsonObject.serializer(),
        JsonObject(mapOf("named_entities" to JsonArray(namedEntities.map { JsonPrimitive(it) }))),
    )

private fun extractNamedEntitiesFromResponse(
    response: String,
    json: Json,
): List<String> {
    val directArray =
        runCatching { json.parseToJsonElement(response).jsonArray }
            .getOrNull()
    if (directArray != null) {
        return directArray
            .mapNotNull { (it as? JsonPrimitive)?.contentOrNull?.trim() }
            .filter { it.isNotEmpty() }
    }
    val jsonObject = extractJsonObjectWithKey(response, "named_entities", json) ?: return emptyList()
    val entities = jsonObject["named_entities"]?.jsonArray ?: return emptyList()
    return entities
        .mapNotNull { (it as? JsonPrimitive)?.contentOrNull?.trim() }
        .filter { it.isNotEmpty() }
}

private fun extractTriplesFromResponse(
    response: String,
    json: Json,
): List<List<String>> {
    val jsonObject = extractJsonObjectWithKey(response, "triples", json) ?: return emptyList()
    val triplesArray = jsonObject["triples"]?.jsonArray ?: return emptyList()
    val triples = mutableListOf<List<String>>()
    for (tripleEl in triplesArray) {
        val tripleArray = tripleEl.asJsonArrayOrNull() ?: continue
        val triple = tripleArray.mapNotNull { it.jsonPrimitive.contentOrNull?.trim() }.filter { it.isNotEmpty() }
        if (triple.size == 3) {
            triples.add(triple)
        }
    }
    return triples
}

private fun JsonElement.asJsonArrayOrNull(): JsonArray? = runCatching { this.jsonArray }.getOrNull()

private inline fun <T> safeExtract(
    logger: KLogger,
    chunkKey: String,
    operation: String,
    fallback: (Exception) -> T,
    block: () -> T,
): T =
    runCatching { block() }.getOrElse { e ->
        when (e) {
            is Error -> {
                throw e
            }

            is Exception -> {
                logger.warn(e) { "$operation failed for chunk $chunkKey" }
                fallback(e)
            }

            else -> {
                throw e
            }
        }
    }
