package lightrag.operate

import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.StreamingChatModel
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler // Corrected import
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import lightrag.core.CacheData
import lightrag.core.QueryParam
import lightrag.core.QueryParamCache
import lightrag.core.QueryResult
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.utils.Prompts
import lightrag.utils.computeMd5

private val logger = KotlinLogging.logger {}

/**
 * Represents the context of a chunk.
 * @property referenceId The reference ID of the chunk.
 * @property content The content of the chunk.
 */
@Serializable
data class ChunkContext(
    @SerialName("reference_id")
    val referenceId: String?,
    val content: String?,
)

/**
 * The default maximum number of total tokens.
 */
const val DEFAULT_MAX_TOTAL_TOKENS = 30000 // From python/constants.py

/**
 * Parameters for a naive query.
 * @property query The query string.
 * @property chunksVdb The vector storage for chunks.
 * @property queryParam The query parameters.
 * @property globalConfig The global configuration.
 * @property hashingKv The key-value storage for hashing.
 * @property systemPrompt The system prompt.
 * @property chatModel The chat model to use.
 * @property tokenizer The tokenizer to use.
 * @property decoder The decoder to use.
 */
data class NaiveQueryParams(
    val query: String,
    val chunksVdb: BaseVectorStorage,
    val queryParam: QueryParam,
    val globalConfig: Map<String, Any?>,
    val hashingKv: BaseKVStorage? = null,
    val systemPrompt: String? = null,
    val chatModel: ChatModel? = null,
    val tokenizer: ((String) -> List<Int>),
    val decoder: ((List<Int>) -> String),
)

// Equivalent to python's _get_vector_context
private suspend fun getVectorContext(
    query: String,
    chunksVdb: BaseVectorStorage,
    queryParam: QueryParam,
    // Not used in naive, so omitting for now
): List<Map<String, Any?>> {
    try {
        val searchTopK = queryParam.chunkTopK.coerceAtMost(queryParam.topK)
        val cosineThreshold = chunksVdb.cosineBetterThanThreshold // Assuming this exists

        val results = chunksVdb.query(query, searchTopK)
        if (results.isEmpty()) {
            logger.info { "Naive query: 0 chunks (chunk_top_k:$searchTopK cosine:$cosineThreshold)" }
            return emptyList()
        }

        val validChunks = mutableListOf<Map<String, Any?>>()
        for (result in results) {
            if (result.containsKey("content")) {
                val chunkWithMetadata =
                    mapOf(
                        "content" to result["content"],
                        "created_at" to result["created_at"],
                        "file_path" to (result["file_path"] ?: "unknown_source"),
                        // Mark the source type so consumers know this came from the vector store
                        "source_type" to "vector",
                        "chunk_id" to (result["id"] ?: result["chunk_id"]),
                    )
                validChunks.add(chunkWithMetadata)
            }
        }

        logger.info { "Naive query: ${validChunks.size} chunks (chunk_top_k:$searchTopK cosine:$cosineThreshold)" }
        return validChunks
    } catch (e: IllegalStateException) {
        logger.error(e) { "Illegal state in _getVectorContext" }
        return emptyList()
    } catch (e: IllegalArgumentException) {
        logger.error(e) { "Invalid input in _getVectorContext" }
        return emptyList()
    }
}

// Equivalent to python's generate_reference_list_from_chunks

/**
 * This function assumes chunk_id and file_path are present in the chunk map.
 * It adds "reference_id" to each processed chunk.
 * @param chunks The chunks to process.
 * @return A pair of the list of references and the list of processed chunks.
 */
fun generateReferenceListFromChunks(chunks: List<Map<String, Any?>>): Pair<List<Map<String, String>>, List<Map<String, Any?>>> {
    val referenceList = mutableListOf<Map<String, String>>()
    val processedChunks = mutableListOf<Map<String, Any?>>()
    val seenFilePaths = mutableSetOf<String>()

    for ((index, chunk) in chunks.withIndex()) {
        val referenceId = (index + 1).toString()
        val filePath = chunk["file_path"] as? String ?: "unknown_source"

        if (!seenFilePaths.contains(filePath)) {
            referenceList.add(mapOf("reference_id" to referenceId, "file_path" to filePath))
            seenFilePaths.add(filePath)
        }

        processedChunks.add(chunk + ("reference_id" to referenceId))
    }
    return referenceList to processedChunks
}

// Simplified version of python's process_chunks_unified, focusing on truncation
// This will just apply token limit, no reranking for now.

/**
 * Processes a list of chunks to fit within a token limit.
 * @param query The query string.
 * @param uniqueChunks The list of unique chunks.
 * @param queryParam The query parameters.
 * @param globalConfig The global configuration.
 * @param sourceType The source type of the chunks.
 * @param chunkTokenLimit The token limit for the chunks.
 * @param tokenizer The tokenizer to use.
 * @param decoder The decoder to use.
 * @return A list of processed chunks.
 */
fun processChunksUnified(
    query: String,
    // Not directly used in this simplified version for reranking, but kept for signature
    uniqueChunks: List<Map<String, Any?>>,
    queryParam: QueryParam,
    globalConfig: Map<String, Any?>,
    sourceType: String,
    // e.g., "vector", "entity", "relation"
    chunkTokenLimit: Int,
    tokenizer: ((String) -> List<Int>),
    decoder: ((List<Int>) -> String),
): List<Map<String, Any?>> {
    val resultChunks = mutableListOf<Map<String, Any?>>()
    var currentTokens = 0

    // Ensure chunks are unique by content, preserving order
    val distinctChunks = uniqueChunks.distinctBy { it["content"] as? String }

    for (chunk in distinctChunks) {
        val content = chunk["content"] as? String ?: ""
        val chunkTokens = tokenizer(content).size

        if (currentTokens + chunkTokens <= chunkTokenLimit) {
            resultChunks.add(chunk)
            currentTokens += chunkTokens
        } else {
            // If adding the whole chunk exceeds the limit, try to truncate it
            val remainingTokens = chunkTokenLimit - currentTokens
            if (remainingTokens > 0) {
                val truncatedContent = truncateTextByTokenSize(content, remainingTokens, tokenizer, decoder)
                if (truncatedContent.isNotBlank()) {
                    resultChunks.add(chunk + ("content" to truncatedContent))
                    currentTokens += tokenizer(truncatedContent).size
                }
            }
            break // No more chunks can be added or partially added
        }
    }
    logger.debug { "Processed chunks: ${resultChunks.size} (total tokens: $currentTokens / $chunkTokenLimit)" }
    return resultChunks
}

/**
 * Truncates a text to a maximum number of tokens.
 * @param text The text to truncate.
 * @param maxTokenSize The maximum number of tokens.
 * @param tokenizer The tokenizer to use.
 * @param decoder The decoder to use.
 * @return The truncated text.
 */
fun truncateTextByTokenSize(
    text: String,
    maxTokenSize: Int,
    tokenizer: ((String) -> List<Int>),
    decoder: ((List<Int>) -> String),
): String {
    val tokens = tokenizer(text)
    if (tokens.size <= maxTokenSize) {
        return text
    }
    return decoder(tokens.subList(0, maxTokenSize))
}

// Function to convert map to JSON string, similar to Python's json.dumps
// This function needs to be properly implemented based on the Python version.
// For now, a placeholder that constructs a basic rawData map.

/**
 * Converts entities, relations, chunks, and references to a JSON format.
 * @param entities The list of entities.
 * @param relations The list of relations.
 * @param chunks The list of chunks.
 * @param references The list of references.
 * @param queryMode The query mode.
 * @param relationIdToOriginal A map of relation IDs to original relations.
 * @return A map representing the JSON format.
 */
fun convertToJsonFormat(
    entities: List<Map<String, Any?>>,
    relations: List<Map<String, Any?>>,
    chunks: List<Map<String, Any?>>,
    references: List<Map<String, String>>,
    queryMode: String,
): Map<String, Any?> {
    val dataMap =
        mutableMapOf<String, Any?>(
            "entities" to entities,
            "relationships" to relations,
            "chunks" to chunks,
            "references" to references,
        )
    val metadataMap =
        mutableMapOf<String, Any?>(
            "query_mode" to queryMode,
        )
    return mutableMapOf(
        "data" to dataMap,
        "metadata" to metadataMap,
    )
}

// Cache helpers moved to QueryProcessor
// For now, just re-implementing saveQueryCache here.
private suspend fun saveQueryCache(
    params: NaiveQueryParams,
    argsHash: String,
    content: String,
    userQuery: String,
    maxTotalTokens: Int,
) {
    if (params.hashingKv != null && (params.globalConfig["enable_llm_cache"] as? Boolean == true)) {
        val queryParamDict =
            QueryParamCache(
                mode = params.queryParam.mode,
                responseType = params.queryParam.responseType,
                topK = params.queryParam.topK,
                chunkTopK = params.queryParam.chunkTopK,
                maxEntityTokens = params.queryParam.maxEntityTokens,
                maxRelationTokens = params.queryParam.maxRelationTokens,
                maxTotalTokens = maxTotalTokens,
                hlKeywords = params.queryParam.hlKeywords.joinToString(", "),
                llKeywords = params.queryParam.llKeywords.joinToString(", "),
                userPrompt = params.queryParam.userPrompt ?: "",
                enableRerank = params.queryParam.enableRerank,
            )
        saveToCache(
            params.hashingKv,
            CacheData(
                argsHash = argsHash,
                content = content,
                prompt = userQuery,
                mode = params.queryParam.mode,
                cacheType = "query",
                queryParam = queryParamDict,
                historyMessages = params.queryParam.conversationHistory,
            ),
        )
    }
}

private suspend fun saveToCache(
    hashingKv: BaseKVStorage?,
    cacheData: CacheData,
) {
    if (hashingKv == null) return
    hashingKv.upsert(
        mapOf(
            cacheData.argsHash to
                mapOf(
                    "content" to cacheData.content,
                    "prompt" to cacheData.prompt,
                    "mode" to cacheData.mode,
                    "cache_type" to cacheData.cacheType,
                    "queryparam" to cacheData.queryParam,
                    "history_messages" to cacheData.historyMessages,
                    // Unix timestamp in seconds
                    "create_time" to System.currentTimeMillis() / 1000,
                ).filterValues { it != null }.mapValues { it.value!! },
        ),
    )
}

private suspend fun handleCache(
    hashingKv: BaseKVStorage?,
    argsHash: String,
    prompt: String,
    mode: String,
    cacheType: String,
): Pair<String, Long>? { // Returns content and timestamp
    if (hashingKv == null) return null

    val cached = hashingKv.getById(argsHash)
    if (cached != null) {
        val cachedContent = cached["content"] as? String
        val timestamp = (cached["create_time"] as? Number)?.toLong() ?: 0L
        if (cachedContent != null) {
            return cachedContent to timestamp
        }
    }
    return null
}

/**
 * Removes think tags from the given text.
 * @param text The text to remove the tags from.
 * @return The text without the tags.
 */
fun removeThinkTags(text: String): String {
    // This is a simplified placeholder. In Python, it would remove specific XML-like tags.
    // Assuming for now that the LLM is not generating these tags in Kotlin or they are removed differently.
    return text.replace("<THINK>", "").replace("</THINK>", "")
}

/**
 * Performs a naive query.
 * @param params The parameters for the query.
 * @return The query result.
 */
suspend fun naiveQuery(params: NaiveQueryParams): QueryResult? {
    if (params.query.isBlank()) return QueryResult(content = Prompts.FAIL_RESPONSE)

    val model = resolveNaiveModel(params) ?: return QueryResult(content = "Error: No LLM model configured.")
    val chunks = getVectorContext(params.query, params.chunksVdb, params.queryParam)
    if (chunks.isEmpty()) return null

    val promptContext = buildNaivePromptContext(params, chunks) ?: return QueryResult(content = Prompts.FAIL_RESPONSE)
    if (params.queryParam.onlyNeedContext) return QueryResult(content = promptContext.contextContent, rawData = promptContext.rawData)
    if (params.queryParam.onlyNeedPrompt) return QueryResult(content = promptContext.promptContent, rawData = promptContext.rawData)

    val cached = handleCache(params.hashingKv, promptContext.argsHash, promptContext.userQuery, params.queryParam.mode, "query")
    if (cached != null) return QueryResult(content = cached.first, rawData = promptContext.rawData)

    return if (params.queryParam.stream) {
        streamNaiveQuery(params, model, promptContext)
    } else {
        runNaiveQuery(params, model, promptContext)
    }
}

private fun resolveNaiveModel(params: NaiveQueryParams): ChatModel? {
    val model = params.chatModel ?: params.globalConfig["llm_model_func"] as? ChatModel
    if (model == null) {
        logger.error { "No ChatModel provided for naiveQuery" }
    }
    return model
}

private fun buildUserPromptStr(queryParam: QueryParam): String =
    if (!queryParam.userPrompt.isNullOrBlank()) {
        "\n\n${queryParam.userPrompt}"
    } else {
        "n/a"
    }

private fun maxTokens(params: NaiveQueryParams): Int =
    params.queryParam.maxTotalTokens.coerceAtMost(
        (params.globalConfig["max_total_tokens"] as? Int) ?: DEFAULT_MAX_TOTAL_TOKENS,
    )

private data class PromptContext(
    val rawData: Map<String, Any?>,
    val contextContent: String,
    val sysPrompt: String,
    val userQuery: String,
    val argsHash: String,
    val maxTotalTokens: Int,
    val promptContent: String,
)

private fun buildNaivePromptContext(
    params: NaiveQueryParams,
    chunks: List<Map<String, Any?>>,
): PromptContext? {
    val tokenizer = params.tokenizer
    val decoder = params.decoder
    val maxTotalTokens = maxTokens(params)
    val userPromptStr = buildUserPromptStr(params.queryParam)
    val responseType = params.queryParam.responseType ?: "Multiple Paragraphs"
    val sysPromptTemplate = params.systemPrompt ?: Prompts.NAIVE_RAG_RESPONSE

    val availableChunkTokens =
        calculateAvailableChunkTokens(
            sysPromptTemplate = sysPromptTemplate,
            responseType = responseType,
            userPromptStr = userPromptStr,
            maxTotalTokens = maxTotalTokens,
            tokenizer = tokenizer,
            userQuery = params.query,
        )

    val processedChunks =
        processChunksUnified(
            query = params.query,
            uniqueChunks = chunks,
            queryParam = params.queryParam,
            globalConfig = params.globalConfig,
            sourceType = "vector",
            chunkTokenLimit = availableChunkTokens,
            tokenizer = tokenizer,
            decoder = decoder,
        )
    val (referenceList, processedChunksWithRefIds) = generateReferenceListFromChunks(processedChunks)
    val rawData = buildRawData(chunks, processedChunksWithRefIds, referenceList)
    val contextContent = buildContextContent(processedChunksWithRefIds, referenceList)

    val sysPrompt =
        sysPromptTemplate
            .replace("{response_type}", responseType)
            .replace("{user_prompt}", userPromptStr)
            .replace("{content_data}", contextContent)

    val promptContent = listOf(sysPrompt, "---User Query---", params.query).joinToString("\n\n")
    val argsHash = computeArgsHash(params, maxTotalTokens)

    return PromptContext(
        rawData = rawData,
        contextContent = contextContent,
        sysPrompt = sysPrompt,
        userQuery = params.query,
        argsHash = argsHash,
        maxTotalTokens = maxTotalTokens,
        promptContent = promptContent,
    )
}

private fun calculateAvailableChunkTokens(
    sysPromptTemplate: String,
    responseType: String,
    userPromptStr: String,
    maxTotalTokens: Int,
    tokenizer: (String) -> List<Int>,
    userQuery: String,
): Int {
    val preSysPrompt =
        sysPromptTemplate
            .replace("{response_type}", responseType)
            .replace("{user_prompt}", userPromptStr)
            .replace("{content_data}", "")
    val sysPromptTokens = tokenizer(preSysPrompt).size
    val queryTokens = tokenizer(userQuery).size
    val bufferTokens = 200
    val availableChunkTokens = maxTotalTokens - (sysPromptTokens + queryTokens + bufferTokens)
    logger.debug {
        "Naive query token allocation - Total: $maxTotalTokens, " +
            "SysPrompt: $sysPromptTokens, Query: $queryTokens, " +
            "Buffer: $bufferTokens, Available for chunks: $availableChunkTokens"
    }
    return availableChunkTokens
}

private fun buildRawData(
    chunks: List<Map<String, Any?>>,
    processedChunksWithRefIds: List<Map<String, Any?>>,
    referenceList: List<Map<String, String>>,
): Map<String, Any?> {
    val rawData =
        convertToJsonFormat(
            emptyList(),
            emptyList(),
            processedChunksWithRefIds,
            referenceList,
            "naive",
        )
    val metadata = mutableMapOf<String, Any?>()
    metadata["keywords"] = mapOf("high_level" to emptyList<String>(), "low_level" to emptyList<String>())
    metadata["processing_info"] =
        mapOf(
            "total_chunks_found" to chunks.size,
            "final_chunks_count" to processedChunksWithRefIds.size,
        )
    (rawData as MutableMap)["metadata"] = metadata
    return rawData
}

private fun buildContextContent(
    processedChunksWithRefIds: List<Map<String, Any?>>,
    referenceList: List<Map<String, String>>,
): String {
    val chunksContext =
        processedChunksWithRefIds.map {
            ChunkContext(
                referenceId = it["reference_id"] as? String,
                content = it["content"] as? String,
            )
        }

    val textUnitsStr = Json.encodeToString(chunksContext)
    val referenceListStr =
        referenceList.joinToString("\n") { ref ->
            "[${ref["reference_id"]}] ${ref["file_path"]}"
        }

    val naiveContextTemplate = Prompts.NAIVE_QUERY_CONTEXT
    return naiveContextTemplate
        .replace("{text_chunks_str}", textUnitsStr)
        .replace("{reference_list_str}", referenceListStr)
}

private fun computeArgsHash(
    params: NaiveQueryParams,
    maxTotalTokens: Int,
): String =
    computeMd5(
        listOf(
            params.queryParam.mode,
            params.query,
            params.queryParam.responseType ?: "",
            params.queryParam.topK.toString(),
            params.queryParam.chunkTopK.toString(),
            params.queryParam.maxEntityTokens.toString(),
            params.queryParam.maxRelationTokens.toString(),
            maxTotalTokens.toString(),
            params.queryParam.userPrompt ?: "",
            params.queryParam.enableRerank.toString(),
        ).joinToString("|"),
    )

private suspend fun streamNaiveQuery(
    params: NaiveQueryParams,
    model: ChatModel,
    promptContext: PromptContext,
): QueryResult {
    val streamingModel = model as? StreamingChatModel
    if (streamingModel == null) {
        logger.error { "Streaming is requested but the model does not support it." }
        return QueryResult(content = "Error: Streaming not supported by model.")
    }
    logger.trace { "SysPrompt: ${promptContext.sysPrompt}" }
    logger.trace { "UserPrompt: ${promptContext.userQuery}" }
    val responseFlow =
        flow {
            val fullResponse = StringBuilder()
            val channel = Channel<String>(Channel.UNLIMITED)

            streamingModel.chat(
                listOf(SystemMessage(promptContext.sysPrompt), UserMessage(promptContext.userQuery)),
                object : StreamingChatResponseHandler {
                    override fun onPartialResponse(partialResponse: String) {
                        channel.trySend(partialResponse)
                        fullResponse.append(partialResponse)
                    }

                    override fun onCompleteResponse(response: ChatResponse) {
                        channel.close()
                    }

                    override fun onError(error: Throwable) {
                        channel.close(error)
                    }
                },
            )

            for (token in channel) {
                emit(token)
            }

            saveQueryCache(params, promptContext.argsHash, fullResponse.toString(), promptContext.userQuery, promptContext.maxTotalTokens)
        }
    return QueryResult(responseIterator = responseFlow, rawData = promptContext.rawData, isStreaming = true)
}

private suspend fun runNaiveQuery(
    params: NaiveQueryParams,
    model: ChatModel,
    promptContext: PromptContext,
): QueryResult {
    var responseContent =
        try {
            logger.trace { "SysPrompt: ${promptContext.sysPrompt}" }
            logger.trace { "UserPrompt: ${promptContext.userQuery}" }
            val chatResponse = model.chat(listOf(SystemMessage(promptContext.sysPrompt), UserMessage(promptContext.userQuery)))
            chatResponse.aiMessage()?.text() ?: ""
        } catch (e: IllegalStateException) {
            logger.error(e) { "Illegal state generating response in naiveQuery" }
            "Error generating response."
        } catch (e: IllegalArgumentException) {
            logger.error(e) { "Invalid input generating response in naiveQuery" }
            "Error generating response."
        }

    if (responseContent.length > promptContext.sysPrompt.length) {
        responseContent =
            responseContent
                .replace(promptContext.sysPrompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(promptContext.userQuery, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .trim()
    }

    saveQueryCache(params, promptContext.argsHash, responseContent, promptContext.userQuery, promptContext.maxTotalTokens)

    return QueryResult(content = responseContent, rawData = promptContext.rawData)
}
