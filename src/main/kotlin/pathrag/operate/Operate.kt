package pathrag.operate

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import pathrag.base.BaseGraphStorage
import pathrag.base.BaseKVStorage
import pathrag.base.BaseVectorStorage
import pathrag.base.QueryParam
import pathrag.prompt.Prompts
import pathrag.utils.ResponseCache
import pathrag.utils.Tokenizer
import pathrag.utils.computeArgsHash
import pathrag.utils.computeMdHashId
import pathrag.utils.entityVectorData
import pathrag.utils.relationshipVectorData
import pathrag.utils.relationshipVectorId
import shared.chunking.chunkByTokenSizeWithOverlap
import shared.chunking.hardTruncateStringsByTokenBudget
import kotlin.math.pow

private val logger = KotlinLogging.logger("PathRAG-Operate")
private typealias LlmFunc =
    suspend (
        prompt: String,
        systemPrompt: String?,
        historyMessages: List<Map<String, String>>,
        keywordExtraction: Boolean,
        stream: Boolean,
        maxTokens: Int?,
        hashingKv: Any?,
    ) -> String

@Serializable
private data class LlmEntity(
    @SerialName("entity_name") val entityName: String = "",
    @SerialName("entity_type") val entityType: String = "UNKNOWN",
    val description: String = "",
    @SerialName("source_id") val sourceId: String? = null,
)

@Serializable
private data class LlmRelationship(
    @SerialName("src_id") val srcId: String = "",
    @SerialName("tgt_id") val tgtId: String = "",
    val description: String = "",
    val keywords: String = "",
    val weight: Double = 1.0,
    @SerialName("source_id") val sourceId: String? = null,
)

@Serializable
private data class ExtractionPayload(
    val entities: List<LlmEntity> = emptyList(),
    val relationships: List<LlmRelationship> = emptyList(),
)

@Serializable
private data class KeywordPayload(
    @SerialName("high_level_keywords") val highLevel: List<String> = emptyList(),
    @SerialName("low_level_keywords") val lowLevel: List<String> = emptyList(),
)

/**
 * Split text into overlapping token-based chunks suitable for token-limited processing.
 *
 * @param content The source text to split.
 * @param overlapTokenSize Number of tokens that each chunk should overlap with the previous chunk.
 * @param maxTokenSize Maximum number of tokens per chunk; must be greater than `overlapTokenSize`.
 * @param tiktokenModel Tokenizer model identifier used to encode/decode the text.
 * @return A list of maps, each containing:
 *   - "tokens": Int — the token count for the chunk,
 *   - "content": String — the decoded text of the chunk,
 *   - "chunk_order_index": Int — the zero-based sequence index of the chunk.
 */
fun chunkingByTokenSize(
    content: String,
    overlapTokenSize: Int = 128,
    maxTokenSize: Int = 1024,
    tiktokenModel: String = "gpt-4o-mini",
): List<Map<String, Any>> {
    val chunks =
        chunkByTokenSizeWithOverlap(
            content = content,
            chunkTokenSize = maxTokenSize,
            chunkOverlapTokenSize = overlapTokenSize,
            model = tiktokenModel,
        )
    return chunks.map { chunk ->
        mapOf(
            "tokens" to chunk.tokens,
            "content" to chunk.content,
            "chunk_order_index" to chunk.chunkOrderIndex,
        )
    }
}

/**
 * Basic entity/relationship extraction that mirrors the Python flow shape:
 * - Uses LLM to parse entities/relationships from text chunks
 * - Merges entities by name, ensures graph + VDB consistency
 */
suspend fun extractEntities(
    chunks: Map<String, Map<String, Any>>,
    knowledgeGraphInst: BaseGraphStorage,
    entityVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    globalConfig: Map<String, Any?>,
    repairExistingVectors: Boolean = false,
    beforeGraphWrite: suspend () -> Unit = {},
): BaseGraphStorage {
    logger.info { "Extracting entities for ${chunks.size} chunks via LLM." }

    @Suppress("UNCHECKED_CAST")
    val llm =
        globalConfig["llm_model_func"] as? LlmFunc ?: run {
            beforeGraphWrite()
            return knowledgeGraphInst
        }

    @Suppress("UNCHECKED_CAST")
    val addonParams = globalConfig["addon_params"] as? Map<String, Any?> ?: emptyMap()

    val language =
        addonParams["language"]?.toString()?.takeIf { it.isNotBlank() } ?: Prompts.DEFAULT_LANGUAGE
    val entityTypes =
        (addonParams["entity_types"] as? List<*>)
            ?.mapNotNull { it?.toString() }
            ?.filter { it.isNotBlank() }
            ?.takeIf { it.isNotEmpty() }
            ?: Prompts.DEFAULT_ENTITY_TYPES
    val exampleNumber = (addonParams["example_number"] as? Number)?.toInt()
    val examples =
        if (exampleNumber != null && exampleNumber > 0 && exampleNumber < Prompts.ENTITY_EXTRACTION_EXAMPLES.size) {
            Prompts.ENTITY_EXTRACTION_EXAMPLES.take(exampleNumber).joinToString("\n")
        } else {
            Prompts.ENTITY_EXTRACTION_EXAMPLES.joinToString("\n")
        }
    val entityExtractMaxGleaning = (globalConfig["entity_extract_max_gleaning"] as? Number)?.toInt() ?: 1

    val orderedChunks = chunks.entries.toList()
    val maxTokensForExtraction = (globalConfig["max_tokens_for_extraction"] as? Int) ?: 2048

    val concurrencyLimit = (globalConfig["max_concurrent_extractions"] as? Number)?.toInt() ?: 16
    val semaphore = kotlinx.coroutines.sync.Semaphore(concurrencyLimit)

    val results =
        coroutineScope {
            orderedChunks
                .map { entry ->
                    async(Dispatchers.Default) {
                        semaphore.withPermit {
                            val (chunkId, chunk) = entry
                            val content = chunk["content"]?.toString().orEmpty()
                            if (content.isBlank()) {
                                return@withPermit Pair(
                                    emptyMap<String, List<Map<String, String>>>(),
                                    emptyMap<Pair<String, String>, List<Map<String, Any?>>>(),
                                )
                            }
                            val prompt =
                                Prompts.render(
                                    Prompts.ENTITY_REL_JSON,
                                    mapOf(
                                        "text" to content,
                                        "language" to language,
                                        "entity_types" to entityTypes.joinToString(", "),
                                        "examples" to examples,
                                    ),
                                )

                            suspend fun requestExtraction(
                                prompt: String,
                                history: List<Map<String, String>>,
                            ): Pair<String, ExtractionPayload> {
                                val response = llm(prompt, null, history, false, false, maxTokensForExtraction, null)
                                return try {
                                    response to Json.decodeFromString<ExtractionPayload>(extractJsonPayload(response))
                                } catch (first: IllegalArgumentException) {
                                    logger.warn(first) { "Retrying malformed extraction for chunk $chunkId" }
                                    val retry = llm(prompt, null, history, false, false, maxTokensForExtraction, null)
                                    try {
                                        retry to Json.decodeFromString<ExtractionPayload>(extractJsonPayload(retry))
                                    } catch (second: IllegalArgumentException) {
                                        throw IllegalStateException("Invalid PathRAG extraction JSON for chunk $chunkId", second)
                                    }
                                }
                            }

                            fun appendHistory(
                                history: MutableList<Map<String, String>>,
                                role: String,
                                message: String,
                            ) {
                                history.add(mapOf("role" to role, "content" to message))
                            }

                            val (response, initialPayload) = requestExtraction(prompt, emptyList())
                            var payload = initialPayload
                            val history = mutableListOf<Map<String, String>>()
                            appendHistory(history, "user", prompt)
                            appendHistory(history, "assistant", response)
                            if (entityExtractMaxGleaning > 0) {
                                for (gleanIndex in 0 until entityExtractMaxGleaning) {
                                    val (gleanResponse, gleanPayload) =
                                        requestExtraction(Prompts.ENTITY_CONTINUE_JSON, history)
                                    appendHistory(history, "user", Prompts.ENTITY_CONTINUE_JSON)
                                    appendHistory(history, "assistant", gleanResponse)
                                    payload =
                                        ExtractionPayload(
                                            entities = payload.entities + gleanPayload.entities,
                                            relationships = payload.relationships + gleanPayload.relationships,
                                        )

                                    if (gleanIndex == entityExtractMaxGleaning - 1) break
                                    val ifLoopResponse =
                                        llm(
                                            Prompts.ENTITY_IF_LOOP_JSON,
                                            null,
                                            history,
                                            false,
                                            false,
                                            maxTokensForExtraction,
                                            null,
                                        ).trim().trim('"', '\'').lowercase()
                                    appendHistory(history, "user", Prompts.ENTITY_IF_LOOP_JSON)
                                    appendHistory(history, "assistant", ifLoopResponse)
                                    if (ifLoopResponse != "yes") break
                                }
                            }

                            val entities = payload.entities.filter { it.entityName.isNotBlank() }
                            val relationships = payload.relationships.filter { it.srcId.isNotBlank() && it.tgtId.isNotBlank() }

                            val nodes =
                                entities
                                    .mapNotNull { ent ->
                                        val name = normalizeId(ent.entityName)
                                        if (name.isBlank()) return@mapNotNull null
                                        val entityType = ent.entityType.ifBlank { "UNKNOWN" }.uppercase()
                                        mapOf(
                                            "entity_type" to entityType,
                                            "description" to ent.description,
                                            "source_id" to chunkId,
                                            "entity_name" to name,
                                        )
                                    }.groupBy { it["entity_name"].orEmpty() }

                            val edges =
                                relationships
                                    .mapNotNull { rel ->
                                        val src = normalizeId(rel.srcId)
                                        val tgt = normalizeId(rel.tgtId)
                                        if (src.isBlank() || tgt.isBlank()) return@mapNotNull null
                                        mapOf(
                                            "src_id" to src,
                                            "tgt_id" to tgt,
                                            "weight" to rel.weight,
                                            "description" to rel.description,
                                            "keywords" to rel.keywords,
                                            "source_id" to chunkId,
                                        )
                                    }.groupBy {
                                        Pair(
                                            it["src_id"]?.toString().orEmpty(),
                                            it["tgt_id"]?.toString().orEmpty(),
                                        )
                                    }

                            Pair(nodes, edges)
                        }
                    }
                }.awaitAll()
        }

    val maybeNodes = mutableMapOf<String, MutableList<Map<String, String>>>()
    val maybeEdges = mutableMapOf<Pair<String, String>, MutableList<Map<String, Any?>>>()
    for ((nodes, edges) in results) {
        nodes.forEach { (k, v) ->
            if (k.isBlank()) return@forEach
            val bucket = maybeNodes.getOrPut(k) { mutableListOf() }
            bucket.addAll(v)
        }
        edges.forEach { (k, v) ->
            if (k.first.isBlank() || k.second.isBlank()) return@forEach
            val bucket = maybeEdges.getOrPut(k) { mutableListOf() }
            bucket.addAll(v)
        }
    }

    beforeGraphWrite()

    val allEntitiesData =
        coroutineScope {
            maybeNodes
                .map { (name, data) ->
                    async(Dispatchers.Default) {
                        mergeNodesThenUpsert(name, data, knowledgeGraphInst, globalConfig, llm, language)
                    }
                }.awaitAll()
        }

    val placeholderDescriptions = mutableMapOf<String, MutableList<String>>()
    val placeholderSourceIds = mutableMapOf<String, MutableList<String>>()
    for ((key, data) in maybeEdges) {
        val descriptions =
            data
                .mapNotNull { it["description"]?.toString() }
                .filter { it.isNotBlank() }
        val sourceIds =
            data
                .mapNotNull { it["source_id"]?.toString() }
                .filter { it.isNotBlank() }
        for (nodeId in listOf(key.first, key.second)) {
            placeholderDescriptions.getOrPut(nodeId) { mutableListOf() }.addAll(descriptions)
            placeholderSourceIds.getOrPut(nodeId) { mutableListOf() }.addAll(sourceIds)
        }
    }

    val placeholderNodeIds = (placeholderDescriptions.keys + placeholderSourceIds.keys).toSet()
    for (nodeId in placeholderNodeIds) {
        if (knowledgeGraphInst.getNode(nodeId) != null) continue
        val description =
            placeholderDescriptions
                .getOrDefault(nodeId, mutableListOf())
                .filter { it.isNotBlank() }
                .toSortedSet()
                .joinToString(Prompts.GRAPH_FIELD_SEP)
        val sourceId =
            placeholderSourceIds
                .getOrDefault(nodeId, mutableListOf())
                .filter { it.isNotBlank() }
                .toSet()
                .joinToString(Prompts.GRAPH_FIELD_SEP)
        knowledgeGraphInst.upsertNode(
            nodeId,
            mapOf(
                "source_id" to sourceId,
                "description" to description,
                "entity_type" to "UNKNOWN",
            ),
        )
    }

    val allRelationshipsData =
        coroutineScope {
            maybeEdges
                .map { (key, data) ->
                    async(Dispatchers.Default) {
                        mergeEdgesThenUpsert(key.first, key.second, data, knowledgeGraphInst, globalConfig, llm, language)
                    }
                }.awaitAll()
        }

    fun belongsToRetry(row: Map<String, Any?>): Boolean =
        splitBySep(row["source_id"]?.toString().orEmpty(), Prompts.GRAPH_FIELD_SEP).any { it in chunks }

    val entitiesToIndex =
        if (repairExistingVectors) {
            val updated = allEntitiesData.mapNotNull { it["entity_name"] }.toSet()
            allEntitiesData +
                knowledgeGraphInst.nodes().filterNot { it in updated }.mapNotNull { name ->
                    knowledgeGraphInst.getNode(name)?.takeIf(::belongsToRetry)?.let {
                        mapOf(
                            "entity_name" to name,
                            "description" to it["description"]?.toString().orEmpty(),
                            "source_id" to it["source_id"]?.toString().orEmpty(),
                        )
                    }
                }
        } else {
            allEntitiesData
        }
    val relationshipsToIndex =
        if (repairExistingVectors) {
            val updated = allRelationshipsData.map { it["src_id"] to it["tgt_id"] }.toSet()
            allRelationshipsData +
                knowledgeGraphInst.edges().filterNot { it in updated }.mapNotNull { (src, tgt) ->
                    knowledgeGraphInst.getEdge(src, tgt)?.takeIf(::belongsToRetry)?.plus(mapOf("src_id" to src, "tgt_id" to tgt))
                }
        } else {
            allRelationshipsData
        }

    if (entitiesToIndex.isNotEmpty()) {
        val toStore =
            entitiesToIndex
                .mapNotNull { ent ->
                    val entityName = ent["entity_name"] ?: return@mapNotNull null
                    val id = computeMdHashId(entityName, prefix = "ent-")
                    id to entityVectorData(entityName, ent)
                }.toMap()
        entityVdb.upsert(toStore)
    }

    if (relationshipsToIndex.isNotEmpty()) {
        val toStore =
            relationshipsToIndex
                .mapNotNull { edge ->
                    val src = edge["src_id"]?.toString() ?: return@mapNotNull null
                    val tgt = edge["tgt_id"]?.toString() ?: return@mapNotNull null
                    relationshipVectorId(src, tgt) to relationshipVectorData(src, tgt, edge)
                }.toMap()
        relationshipsVdb.upsert(toStore)
    }

    return knowledgeGraphInst
}

private suspend fun mergeNodesThenUpsert(
    entityName: String,
    nodesData: List<Map<String, String>>,
    knowledgeGraphInst: BaseGraphStorage,
    globalConfig: Map<String, Any?>,
    llm: LlmFunc,
    language: String,
): Map<String, String> {
    val alreadyEntityTypes = mutableListOf<String>()
    val alreadySourceIds = mutableListOf<String>()
    val alreadyDescriptions = mutableListOf<String>()

    val alreadyNode = knowledgeGraphInst.getNode(entityName)
    if (alreadyNode != null) {
        alreadyEntityTypes.add(alreadyNode["entity_type"]?.toString().orEmpty())
        alreadySourceIds.addAll(splitBySep(alreadyNode["source_id"]?.toString().orEmpty(), Prompts.GRAPH_FIELD_SEP))
        alreadyDescriptions.addAll(splitBySep(alreadyNode["description"]?.toString().orEmpty(), Prompts.GRAPH_FIELD_SEP))
    }

    val entityType =
        (nodesData.mapNotNull { it["entity_type"] } + alreadyEntityTypes)
            .groupingBy { it }
            .eachCount()
            .maxByOrNull { it.value }
            ?.key
            .orEmpty()
            .ifBlank { "UNKNOWN" }
    var description =
        (nodesData.mapNotNull { it["description"] } + alreadyDescriptions)
            .filter { it.isNotBlank() }
            .toSortedSet()
            .joinToString(Prompts.GRAPH_FIELD_SEP)
    val sourceId =
        (nodesData.mapNotNull { it["source_id"] } + alreadySourceIds)
            .filter { it.isNotBlank() }
            .toSet()
            .joinToString(Prompts.GRAPH_FIELD_SEP)
    description = handleEntityRelationSummary(entityName, description, globalConfig, llm, language)

    val nodeData =
        mapOf(
            "entity_type" to entityType,
            "description" to description,
            "source_id" to sourceId,
        )
    knowledgeGraphInst.upsertNode(entityName, nodeData)
    return nodeData + mapOf("entity_name" to entityName)
}

private suspend fun mergeEdgesThenUpsert(
    srcId: String,
    tgtId: String,
    edgesData: List<Map<String, Any?>>,
    knowledgeGraphInst: BaseGraphStorage,
    globalConfig: Map<String, Any?>,
    llm: LlmFunc,
    language: String,
): Map<String, Any?> {
    val alreadyWeights = mutableListOf<Double>()
    val alreadySourceIds = mutableListOf<String>()
    val alreadyDescriptions = mutableListOf<String>()
    val alreadyKeywords = mutableListOf<String>()

    val alreadyEdge = knowledgeGraphInst.getEdge(srcId, tgtId)
    if (alreadyEdge != null) {
        alreadyWeights.add((alreadyEdge["weight"] as? Number)?.toDouble() ?: 0.0)
        alreadySourceIds.addAll(splitBySep(alreadyEdge["source_id"]?.toString().orEmpty(), Prompts.GRAPH_FIELD_SEP))
        alreadyDescriptions.addAll(splitBySep(alreadyEdge["description"]?.toString().orEmpty(), Prompts.GRAPH_FIELD_SEP))
        alreadyKeywords.addAll(splitBySep(alreadyEdge["keywords"]?.toString().orEmpty(), Prompts.GRAPH_FIELD_SEP))
    }

    val weight =
        edgesData
            .filterNot { it["source_id"] in alreadySourceIds }
            .mapNotNull { (it["weight"] as? Number)?.toDouble() } + alreadyWeights
    var description =
        (edgesData.mapNotNull { it["description"]?.toString() } + alreadyDescriptions)
            .filter { it.isNotBlank() }
            .toSortedSet()
            .joinToString(Prompts.GRAPH_FIELD_SEP)
    val keywords =
        (edgesData.mapNotNull { it["keywords"]?.toString() } + alreadyKeywords)
            .filter { it.isNotBlank() }
            .toSortedSet()
            .joinToString(Prompts.GRAPH_FIELD_SEP)
    val sourceId =
        (edgesData.mapNotNull { it["source_id"]?.toString() } + alreadySourceIds)
            .filter { it.isNotBlank() }
            .toSet()
            .joinToString(Prompts.GRAPH_FIELD_SEP)

    description = handleEntityRelationSummary("($srcId, $tgtId)", description, globalConfig, llm, language)
    knowledgeGraphInst.upsertEdge(
        srcId,
        tgtId,
        mapOf(
            "weight" to weight.sum(),
            "description" to description,
            "keywords" to keywords,
            "source_id" to sourceId,
        ),
    )

    return mapOf(
        "src_id" to srcId,
        "tgt_id" to tgtId,
        "description" to description,
        "keywords" to keywords,
        "source_id" to sourceId,
        "weight" to weight.sum(),
    )
}

private suspend fun handleEntityRelationSummary(
    entityOrRelationName: String,
    description: String,
    globalConfig: Map<String, Any?>,
    llm: LlmFunc,
    language: String,
): String {
    if (description.isBlank()) return description
    val llmMaxTokens = (globalConfig["llm_model_max_token_size"] as? Number)?.toInt() ?: 32768
    val tiktokenModelName = globalConfig["tiktoken_model_name"]?.toString()?.ifBlank { "gpt-4o-mini" } ?: "gpt-4o-mini"
    val summaryMaxTokens = (globalConfig["entity_summary_to_max_tokens"] as? Number)?.toInt() ?: 500

    val tokens = Tokenizer.encode(description, tiktokenModelName)
    if (tokens.size < summaryMaxTokens) return description
    val useDescription = Tokenizer.decode(tokens.take(llmMaxTokens), tiktokenModelName)
    val context =
        mapOf(
            "entity_name" to entityOrRelationName,
            "description_list" to splitBySep(useDescription, Prompts.GRAPH_FIELD_SEP).toString(),
            "language" to language,
        )
    val prompt = Prompts.render(Prompts.SUMMARIZE_ENTITY_DESCRIPTIONS, context)
    logger.debug { "Trigger summary: $entityOrRelationName" }
    return llm(prompt, null, emptyList(), false, false, summaryMaxTokens, null)
}

private fun splitBySep(
    value: String,
    sep: String,
): List<String> =
    value
        .split(sep)
        .map { it.trim() }
        .filter { it.isNotBlank() }

private fun extractJsonPayload(response: String): String {
    val trimmed = response.trim()
    val fencedRegex = Regex("```(?:json)?\\s*([\\s\\S]*?)\\s*```", setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL))
    val fencedMatch = fencedRegex.find(trimmed)
    if (fencedMatch != null) {
        return fencedMatch.groupValues[1].trim()
    }
    val firstBrace = trimmed.indexOf('{')
    val lastBrace = trimmed.lastIndexOf('}')
    if (firstBrace != -1 && lastBrace > firstBrace) {
        return trimmed.substring(firstBrace, lastBrace + 1).trim()
    }
    return trimmed
}

/**
 * Normalize an identifier by removing surrounding double quotes and converting to uppercase.
 *
 * @param id The identifier to normalize; may include surrounding double quotes.
 * @return The normalized identifier with no surrounding double quotes and in uppercase.
 */
private fun normalizeId(id: String): String =
    id
        .trim()
        .trim('"')
        .trim()
        .uppercase()

/**
 * Run a retrieval-augmented generation (RAG) query backed by the knowledge graph using the configured mode.
 *
 * Dispatches the query to one of the mode-specific runners ("local", "global", "hybrid"), extracts keywords,
 * composes the system context, and caches the result when a ResponseCache is provided.
 *
 * @param query The user's query text.
 * @param knowledgeGraphInst Graph storage instance used for node/edge lookups and path computations.
 * @param entitiesVdb Vector database storing entity embeddings and metadata.
 * @param relationshipsVdb Vector database storing relationship embeddings and metadata.
 * @param textChunksDb Key-value storage containing text chunks referenced by source IDs.
 * @param queryParam Parameters controlling retrieval and response behavior (mode, topK, streaming, etc.).
 * @param globalConfig Global configuration map used for keyword extraction and other runtime options.
 * @param llmModel LLM call function used to generate keywords and final responses. It receives prompt, optional system prompt,
 *                 conversation history, keywordExtraction flag, streaming flag, optional maxTokens, and an optional hashingKv.
 * @param hashingKv Optional response cache used to read/write cached responses keyed by the query and mode.
 *
 * @return The generated response string for the query.
 * @throws IllegalArgumentException If `queryParam.mode` is not one of "local", "global", or "hybrid".
 */
suspend fun kgQuery(
    query: String,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
    queryParam: QueryParam,
    globalConfig: Map<String, Any?>,
    llmModel: suspend (
        prompt: String,
        systemPrompt: String?,
        historyMessages: List<Map<String, String>>,
        keywordExtraction: Boolean,
        stream: Boolean,
        maxTokens: Int?,
        hashingKv: Any?,
    ) -> String,
    hashingKv: ResponseCache? = null,
    contextLock: Mutex? = null,
): String =
    withContext(Dispatchers.Default) {
        val (llKeywords, hlKeywords) = extractKeywords(llmModel, query, globalConfig)
        val mode = queryParam.mode.lowercase()

        suspend fun buildContext(): Pair<String, Int> {
            val sections =
                when (mode) {
                    "local" -> {
                        listOf(getNodeData(llKeywords, knowledgeGraphInst, entitiesVdb, textChunksDb, queryParam))
                    }

                    "global" -> {
                        listOf(getEdgeData(hlKeywords, knowledgeGraphInst, relationshipsVdb, textChunksDb, queryParam))
                    }

                    "hybrid" -> {
                        listOf(
                            getEdgeData(hlKeywords, knowledgeGraphInst, relationshipsVdb, textChunksDb, queryParam),
                            getNodeData(llKeywords, knowledgeGraphInst, entitiesVdb, textChunksDb, queryParam),
                        )
                    }

                    else -> {
                        throw IllegalArgumentException("Unknown query mode: ${queryParam.mode}. Supported modes: local, global, hybrid")
                    }
                }
            val context =
                if (mode == "hybrid") {
                    renderWithinBudget(sections, queryParam.maxTokenForTextUnit)
                } else {
                    sections.single().render()
                }
            return context to if (queryParam.onlyNeedContext) 0 else knowledgeGraphInst.nodes().size
        }
        val (context, nodeCount) = contextLock?.withLock { buildContext() } ?: buildContext()
        if (queryParam.onlyNeedContext) return@withContext context

        val requestPrompt =
            Prompts.render(
                Prompts.RAG_RESPONSE,
                mapOf("context_data" to context, "response_type" to queryParam.responseType),
            )
        val systemPrompt = "PathRAG (Kotlin) | nodes=$nodeCount | mode=$mode\n$requestPrompt"
        if (queryParam.onlyNeedPrompt) return@withContext systemPrompt

        val responseCache = hashingKv.takeUnless { queryParam.stream }
        val argsHash = computeArgsHash(queryParam, query, systemPrompt)
        val cached = responseCache?.handleCache(argsHash, query, queryParam.mode, allowSimilar = false)
        if (cached != null) return@withContext cached

        val maxOutputTokens = if (mode == "global") queryParam.maxTokenForGlobalContext else queryParam.maxTokenForTextUnit
        val response = llmModel(query, systemPrompt, emptyList(), false, queryParam.stream, maxOutputTokens, null)
        responseCache?.upsert(queryParam.mode, argsHash, response, query)
        response
    }

private data class ContextTable(
    val title: String,
    val headers: List<String>,
    val rows: List<List<String>> = emptyList(),
    val sourceIds: List<String> = emptyList(),
    val reverseForPrompt: Boolean = false,
) {
    private val serializedRows = rows.map(::serializeCsvRow)
    private val header = serializeCsvRow(headers)

    fun render(indices: List<Int> = rows.indices.toList()): String =
        buildString {
            append("-----$title-----\n```csv\n")
            append(header)
            for (index in if (reverseForPrompt) indices.asReversed() else indices) append('\n').append(serializedRows[index])
            append("\n```")
        }

    fun rowTokens(index: Int): Int = Tokenizer.count("\n${serializedRows[index]}")
}

private data class ContextSection(
    val name: String,
    val tables: List<ContextTable>,
) {
    fun render(): String = "-----$name-information-----\n" + tables.joinToString("\n") { it.render() }
}

private fun renderWithinBudget(
    sections: List<ContextSection>,
    maxTokens: Int,
): String {
    val tables = sections.flatMap { it.tables }
    val seenSources = mutableSetOf<String>()
    val candidates =
        tables.map { table ->
            table.rows.indices.filter { table.sourceIds.isEmpty() || seenSources.add(table.sourceIds[it]) }
        }

    fun render(selection: List<List<Int>>): String {
        var index = 0
        return sections.joinToString("\n") { section ->
            "-----${section.name}-information-----\n" +
                section.tables.joinToString("\n") { table -> table.render(selection[index++]) }
        }
    }

    val full = render(candidates)
    if (Tokenizer.count(full) <= maxTokens) return full

    val selected = tables.map { mutableListOf<Int>() }
    var remaining = maxTokens - Tokenizer.count(render(selected))
    if (remaining < 0) return ""
    val order = tables.indices.sortedBy { if (tables[it].sourceIds.isNotEmpty()) 0 else 1 }
    val positions = IntArray(tables.size)
    val additions = mutableListOf<Int>()
    while (remaining > 0) {
        var hasCandidate = false
        for (index in order) {
            val row = candidates[index].getOrNull(positions[index]++) ?: continue
            hasCandidate = true
            val cost = tables[index].rowTokens(row)
            if (cost > remaining) continue
            selected[index].add(row)
            additions.add(index)
            remaining -= cost
        }
        if (!hasCandidate) break
    }

    // Token counts can change where CSV rows meet. Check the complete context before returning it.
    var context = render(selected)
    while (Tokenizer.count(context) > maxTokens && additions.isNotEmpty()) {
        selected[additions.removeLast()].removeLast()
        context = render(selected)
    }
    return context
}

private fun firstSourceId(vararg values: Any?): String =
    values.firstNotNullOfOrNull { it?.toString()?.takeIf(String::isNotBlank) }.orEmpty()

private suspend fun getNodeData(
    keywords: String,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
    queryParam: QueryParam,
): ContextSection {
    val results = entitiesVdb.query(keywords, topK = queryParam.topK)
    val nodeDatas =
        results.mapNotNull { res ->
            val name = res["entity_name"]?.toString() ?: res["content"]?.toString()
            if (name != null) {
                val node = knowledgeGraphInst.getNode(name)
                val degree = knowledgeGraphInst.nodeDegree(name)
                val desc = node?.get("description") ?: res["content"]
                mapOf(
                    "entity_name" to name,
                    "entity_type" to (node?.get("entity_type") ?: "UNKNOWN"),
                    "description" to (desc ?: ""),
                    "rank" to degree,
                    "source_id" to firstSourceId(node?.get("source_id"), res["source_id"], res["full_doc_id"]),
                )
            } else {
                null
            }
        }

    val textUnits = findMostRelatedTextUnitFromEntities(nodeDatas, queryParam, textChunksDb)
    val paths = buildPathRelations(nodeDatas, knowledgeGraphInst, queryParam)

    val entities =
        ContextTable(
            "low-level entity information",
            listOf("id", "entity", "type", "description", "rank"),
            nodeDatas.mapIndexed { idx, n ->
                listOf(
                    idx.toString(),
                    n["entity_name"].toString(),
                    n["entity_type"].toString(),
                    n["description"].toString(),
                    n["rank"].toString(),
                )
            },
        )
    val relations =
        ContextTable(
            "low-level relationship information",
            listOf("context"),
            paths.map { listOf(it) },
            reverseForPrompt = true,
        )
    val sources =
        ContextTable(
            "Sources",
            listOf("id", "content"),
            textUnits.mapIndexed { idx, t ->
                listOf(idx.toString(), t["content"].toString())
            },
            sourceIds = textUnits.map { it.getValue("_chunk_id").toString() },
        )

    return ContextSection("local", listOf(entities, relations, sources))
}

private suspend fun findMostRelatedTextUnitFromEntities(
    nodeDatas: List<Map<String, Any>>,
    queryParam: QueryParam,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
): List<Map<String, Any>> {
    val ids =
        nodeDatas.flatMap { node ->
            node["source_id"]
                ?.toString()
                ?.split(Prompts.GRAPH_FIELD_SEP, ",")
                ?.map { it.trim() }
                ?.filter { it.isNotEmpty() }
                ?: emptyList()
        }
    val uniqueIds = ids.distinct()
    val chunks = textChunksDb.getByIds(uniqueIds)
    val valid = uniqueIds.zip(chunks).mapNotNull { (id, chunk) -> chunk?.plus("_chunk_id" to id) }.take(queryParam.topK)
    return truncateByToken(valid, queryParam.maxTokenForTextUnit)
}

private suspend fun buildPathRelations(
    nodeDatas: List<Map<String, Any>>,
    knowledgeGraphInst: BaseGraphStorage,
    queryParam: QueryParam,
): List<String> {
    val targetNodes = nodeDatas.mapNotNull { it["entity_name"]?.toString() }.distinct()
    if (targetNodes.size < 2) return emptyList()

    val edgesList = knowledgeGraphInst.edges()
    val adjacency = mutableMapOf<String, MutableSet<String>>()
    edgesList.forEach { (u, v) ->
        adjacency.computeIfAbsent(u) { mutableSetOf() }.add(v)
        adjacency.computeIfAbsent(v) { mutableSetOf() }.add(u)
    }
    if (adjacency.isEmpty()) return emptyList()

    data class PathBucket(
        val paths: MutableList<List<String>> = mutableListOf(),
        val edges: MutableSet<Pair<String, String>> = mutableSetOf(),
    )
    val result = mutableMapOf<Pair<String, String>, PathBucket>()
    val oneHopPaths = mutableListOf<List<String>>()
    val twoHopPaths = mutableListOf<List<String>>()
    val threeHopPaths = mutableListOf<List<String>>()

    fun dfs(
        current: String,
        target: String,
        path: List<String>,
        depth: Int,
    ) {
        if (depth > 3) return
        if (current == target) {
            val key = path.first() to target
            val bucket = result.getOrPut(key) { PathBucket() }
            bucket.paths.add(path)
            path.windowed(2).forEach { (u, v) -> bucket.edges.add(if (u <= v) u to v else v to u) }
            when (depth) {
                1 -> oneHopPaths.add(path)
                2 -> twoHopPaths.add(path)
                3 -> threeHopPaths.add(path)
            }
            return
        }
        adjacency[current].orEmpty().forEach { n ->
            if (n !in path) {
                dfs(n, target, path + n, depth + 1)
            }
        }
    }

    for (n1 in targetNodes) {
        for (n2 in targetNodes) {
            if (n1 != n2) {
                dfs(n1, n2, listOf(n1), 0)
            }
        }
    }

    fun bfsWeightedPaths(
        source: String,
        target: String,
        paths: List<List<String>>,
        threshold: Double,
        alpha: Double,
    ): List<Pair<List<String>, Double>> {
        if (paths.isEmpty()) return emptyList()
        val follow = mutableMapOf<String, MutableSet<String>>()
        paths.forEach { p ->
            p.windowed(2).forEach { (u, v) ->
                follow.computeIfAbsent(u) { mutableSetOf() }.add(v)
            }
        }
        val edgeWeights = mutableMapOf<Pair<String, String>, Double>()
        val results = mutableListOf<List<String>>()

        fun incEdge(
            u: String,
            v: String,
            add: Double,
        ) {
            edgeWeights[u to v] = (edgeWeights[u to v] ?: 0.0) + add
        }

        for (n in follow[source].orEmpty()) {
            incEdge(source, n, 1.0 / follow[source]!!.size)
            if (n == target) {
                results.add(listOf(source, n))
                continue
            }
            if ((edgeWeights[source to n] ?: 0.0) > threshold) {
                for (m in follow[n].orEmpty()) {
                    val w = (edgeWeights[source to n] ?: 0.0) * alpha / follow[n]!!.size
                    incEdge(n, m, w)
                    if (m == target) {
                        results.add(listOf(source, n, m))
                        continue
                    }
                    if ((edgeWeights[n to m] ?: 0.0) > threshold) {
                        for (k in follow[m].orEmpty()) {
                            val w2 = (edgeWeights[n to m] ?: 0.0) * alpha / follow[m]!!.size
                            incEdge(m, k, w2)
                            if (k == target) {
                                results.add(listOf(source, n, m, k))
                            }
                        }
                    }
                }
            }
        }
        return paths.map { p ->
            val pw =
                if (p.size < 2) {
                    0.0
                } else {
                    var sum = 0.0
                    p.windowed(2).forEach { (u, v) -> sum += edgeWeights[u to v] ?: 0.0 }
                    sum / (p.size - 1)
                }
            p to pw
        }
    }

    val threshold = 0.3
    val alpha = 0.8
    val allResults = mutableListOf<Pair<List<String>, Double>>()
    for (n1 in targetNodes) {
        for (n2 in targetNodes) {
            if (n1 != n2) {
                val bucket = result[n1 to n2] ?: continue
                val paths = bucket.paths
                val scored = bfsWeightedPaths(n1, n2, paths, threshold, alpha)
                allResults.addAll(scored)
            }
        }
    }
    val sortedResults = allResults.sortedByDescending { it.second }
    val seen = mutableSetOf<List<String>>()
    val resultEdge =
        sortedResults.filter { (path, _) ->
            val canonicalPath = if (path.first() <= path.last()) path else path.reversed()
            seen.add(canonicalPath)
        }

    val length1 = oneHopPaths.size / 2
    val length2 = twoHopPaths.size / 2
    val length3 = threeHopPaths.size / 2
    val baseResults = mutableListOf<List<String>>()
    if (oneHopPaths.isNotEmpty()) baseResults.addAll(oneHopPaths.take(length1))
    if (twoHopPaths.isNotEmpty()) baseResults.addAll(twoHopPaths.take(length2))
    if (threeHopPaths.isNotEmpty()) baseResults.addAll(threeHopPaths.take(length3))

    var totalEdges = 15
    if (baseResults.size < totalEdges) totalEdges = baseResults.size
    val sortResult =
        if (resultEdge.isNotEmpty()) {
            if (resultEdge.size > totalEdges) resultEdge.take(totalEdges) else resultEdge
        } else {
            emptyList()
        }
    val finalPaths = sortResult.map { it.first }

    suspend fun describePath(path: List<String>): String? {
        suspend fun nodeDesc(id: String): String {
            val n = knowledgeGraphInst.getNode(id) ?: return "The entity $id"
            val t = n["entity_type"] ?: "UNKNOWN"
            val d = n["description"] ?: ""
            return "The entity $id is a $t with the description($d)"
        }

        suspend fun edgeDesc(
            u: String,
            v: String,
        ): String? {
            val e = knowledgeGraphInst.getEdge(u, v) ?: knowledgeGraphInst.getEdge(v, u)
            val kw = e?.get("keywords")?.toString().orEmpty()
            val desc = e?.get("description")?.toString().orEmpty()
            if (kw.isBlank() && desc.isBlank()) return null
            val edgeInfo = listOfNotNull(desc.takeIf { it.isNotBlank() }, kw.takeIf { it.isNotBlank() }).joinToString("; ")
            return "through edge($edgeInfo) to connect to $u and $v."
        }
        return when (path.size) {
            2 -> {
                val (s, t) = path
                val e = edgeDesc(s, t) ?: return null
                "${nodeDesc(s)} $e ${nodeDesc(t)}"
            }

            3 -> {
                val (s, b, t) = path
                val e1 = edgeDesc(s, b) ?: return null
                val e2 = edgeDesc(b, t) ?: return null
                "${nodeDesc(s)} $e1 ${nodeDesc(b)} and ${nodeDesc(b)} $e2 ${nodeDesc(t)}"
            }

            4 -> {
                val s = path[0]
                val b1 = path[1]
                val b2 = path[2]
                val t = path[3]
                val e1 = edgeDesc(s, b1) ?: return null
                val e2 = edgeDesc(b1, b2) ?: return null
                val e3 = edgeDesc(b2, t) ?: return null
                "${nodeDesc(s)} $e1 ${nodeDesc(b1)} and ${nodeDesc(b1)} $e2 ${nodeDesc(b2)} and ${nodeDesc(b2)} $e3 ${nodeDesc(t)}"
            }

            else -> {
                null
            }
        }
    }

    val described = mutableListOf<String>()
    for (p in finalPaths) {
        val d = describePath(p)
        if (d != null) described.add(d)
    }

    val truncated =
        truncateByToken(described.map { mapOf("content" to it) }, queryParam.maxTokenForLocalContext)
            .map { it["content"].toString() }
    return truncated
}

private fun truncateByToken(
    list: List<Map<String, Any>>,
    maxToken: Int,
    model: String = "gpt-4o-mini",
): List<Map<String, Any>> {
    val keptContents =
        hardTruncateStringsByTokenBudget(
            items = list.map { it["content"]?.toString().orEmpty() },
            maxTokenSize = maxToken,
            model = model,
            includePartialLastItem = false,
        )
    return list.take(keptContents.size)
}

private fun serializeCsvRow(row: List<String>): String = row.joinToString(",") { escapeCsvField(it) }

private fun escapeCsvField(value: String): String {
    val needsQuotes = value.contains(',') || value.contains('\n') || value.contains('"')
    if (!needsQuotes) return value
    val escaped = value.replace("\"", "\"\"")
    return "\"$escaped\""
}

private suspend fun extractKeywords(
    llmModel: suspend (
        prompt: String,
        systemPrompt: String?,
        historyMessages: List<Map<String, String>>,
        keywordExtraction: Boolean,
        stream: Boolean,
        maxTokens: Int?,
        hashingKv: Any?,
    ) -> String,
    query: String,
    globalConfig: Map<String, Any?>,
): Pair<String, String> {
    val configuredHigh =
        (globalConfig["fixed_high_level_keywords"] as? Collection<*>)
            ?.mapNotNull { it?.toString()?.takeIf { s -> s.isNotBlank() } }
            ?: emptyList()
    val configuredLow =
        (globalConfig["fixed_low_level_keywords"] as? Collection<*>)
            ?.mapNotNull { it?.toString()?.takeIf { s -> s.isNotBlank() } }
            ?: emptyList()
    if (configuredHigh.isNotEmpty() || configuredLow.isNotEmpty()) {
        val hl = configuredHigh.joinToString(", ")
        val ll = configuredLow.joinToString(", ")
        return ll to hl
    }

    val examples = (globalConfig["keywords_examples"] as? String).orEmpty()
    val prompt =
        Prompts.render(
            Prompts.KEYWORDS_EXTRACTION,
            mapOf(
                "query" to query,
                "examples" to examples,
            ),
        )
    val raw = llmModel(prompt, null, emptyList(), true, false, 512, null)
    val parsed =
        runCatching { Json.decodeFromString<KeywordPayload>(extractJsonPayload(raw)) }
            .getOrElse { KeywordPayload() }
    val hl = parsed.highLevel.joinToString(", ")
    val ll = parsed.lowLevel.joinToString(", ")
    return ll to hl
}

private suspend fun getEdgeData(
    keywords: String,
    knowledgeGraphInst: BaseGraphStorage,
    relationshipsVdb: BaseVectorStorage,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
    queryParam: QueryParam,
): ContextSection {
    val results = relationshipsVdb.query(keywords, topK = queryParam.topK)
    val edges =
        results
            .mapNotNull { res ->
                val src = res["src_id"]?.toString()
                val tgt = res["tgt_id"]?.toString()
                if (src != null && tgt != null) {
                    val edge = knowledgeGraphInst.getEdge(src, tgt)
                    if (edge != null) {
                        edge +
                            mapOf(
                                "src_id" to src,
                                "tgt_id" to tgt,
                                "source_id" to firstSourceId(edge["source_id"], res["source_id"], res["full_doc_id"]),
                            )
                    } else {
                        null
                    }
                } else {
                    null
                }
            }.distinctBy { it["src_id"] to it["tgt_id"] }

    val entityIds =
        edges
            .flatMap { e ->
                listOfNotNull(
                    e["src_id"]?.toString(),
                    e["tgt_id"]?.toString(),
                )
            }.distinct()
    val nodeDatas =
        entityIds.mapNotNull { name ->
            val node = knowledgeGraphInst.getNode(name)
            val degree = knowledgeGraphInst.nodeDegree(name)
            node?.let {
                mapOf(
                    "entity_name" to name,
                    "entity_type" to (it["entity_type"] ?: "UNKNOWN"),
                    "description" to (it["description"] ?: ""),
                    "rank" to degree,
                    "source_id" to (it["source_id"] ?: ""),
                )
            }
        }

    val textUnits = findRelatedTextUnitFromRelationships(edges, textChunksDb, queryParam)

    val entities =
        ContextTable(
            "high-level entity information",
            listOf("id", "entity", "type", "description", "rank"),
            nodeDatas.mapIndexed { idx, n ->
                listOf(
                    idx.toString(),
                    n["entity_name"].toString(),
                    n["entity_type"].toString(),
                    n["description"].toString(),
                    n["rank"].toString(),
                )
            },
        )

    val relations =
        ContextTable(
            "high-level relationship information",
            listOf("id", "source", "target", "description", "keywords", "weight"),
            edges.mapIndexed { idx, e ->
                listOf(
                    idx.toString(),
                    e["src_id"].toString(),
                    e["tgt_id"].toString(),
                    e["description"]?.toString() ?: "",
                    e["keywords"]?.toString() ?: "",
                    e["weight"]?.toString() ?: "1.0",
                )
            },
        )

    val sources =
        ContextTable(
            "Sources",
            listOf("id", "content"),
            textUnits.mapIndexed { idx, t ->
                listOf(idx.toString(), t["content"].toString())
            },
            sourceIds = textUnits.map { it.getValue("_chunk_id").toString() },
        )

    return ContextSection("global", listOf(entities, relations, sources))
}

private suspend fun findRelatedTextUnitFromRelationships(
    edges: List<Map<String, Any?>>,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
    queryParam: QueryParam,
): List<Map<String, Any>> {
    val ids =
        edges.flatMap { e ->
            e["source_id"]
                ?.toString()
                ?.split(Prompts.GRAPH_FIELD_SEP, ",")
                ?.map { it.trim() }
                ?.filter { it.isNotEmpty() }
                ?: emptyList()
        }
    val unique = ids.distinct()
    val chunks = textChunksDb.getByIds(unique)
    val valid = unique.zip(chunks).mapNotNull { (id, chunk) -> chunk?.plus("_chunk_id" to id) }.take(queryParam.topK)
    return truncateByToken(valid, queryParam.maxTokenForTextUnit)
}
