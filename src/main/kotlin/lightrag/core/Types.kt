package lightrag.core

import kotlinx.coroutines.flow.Flow
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Represents the result of a query.
 * @property content The content of the response.
 * @property responseIterator A flow of response strings for streaming responses.
 * @property rawData The raw data of the response.
 * @property isStreaming Whether the response is streaming.
 */
data class QueryResult(
    val content: String? = null,
    val responseIterator: Flow<String>? = null,
    val rawData: Map<String, Any?>? = null,
    val isStreaming: Boolean = false,
) {
    /**
     * A list of references from the raw data.
     */
    val referenceList: List<Map<String, String>>
        get() =
            rawData
                ?.get("data")
                ?.let { it as? Map<*, *> }
                ?.get("references")
                ?.let { refs ->
                    (refs as? List<*>)?.mapNotNull { item ->
                        (item as? Map<*, *>)?.entries?.associate { (k, v) -> k.toString() to v.toString() }
                    }
                }
                ?: emptyList()

    /**
     * Metadata from the raw data.
     */
    val metadata: Map<String, Any?>
        get() =
            rawData
                ?.get("metadata")
                ?.let { it as? Map<*, *> }
                ?.entries
                ?.associate { (k, v) -> k.toString() to v }
                ?: emptyMap()
}

/**
 * Represents the context of a query result.
 * @property context The context string.
 * @property rawData The raw data of the context.
 */
data class QueryContextResult(
    val context: String,
    val rawData: Map<String, Any?>,
) {
    /**
     * A list of references from the raw data.
     */
    val referenceList: List<Map<String, String>>
        get() =
            rawData["data"]
                ?.let { it as? Map<*, *> }
                ?.get("references")
                ?.let { refs ->
                    (refs as? List<*>)?.mapNotNull { item ->
                        (item as? Map<*, *>)?.entries?.associate { (k, v) -> k.toString() to v.toString() }
                    }
                }
                ?: emptyList()
}

/**
 * Represents the schema of a text chunk.
 * @property tokens The number of tokens in the chunk.
 * @property content The content of the chunk.
 * @property fullDocId The ID of the full document.
 * @property chunkOrderIndex The order index of the chunk.
 */
@Serializable
data class TextChunkSchema(
    val tokens: Int,
    val content: String,
    @SerialName("full_doc_id")
    val fullDocId: String,
    @SerialName("chunk_order_index")
    val chunkOrderIndex: Int,
)

/**
 * Represents the cache of query parameters.
 * @property mode The query mode.
 * @property responseType The desired response type.
 * @property topK The number of top results to return.
 * @property chunkTopK The number of top chunks to return.
 * @property maxEntityTokens The maximum number of tokens for entities.
 * @property maxRelationTokens The maximum number of tokens for relations.
 * @property maxTotalTokens The maximum total number of tokens.
 * @property hlKeywords Keywords to highlight.
 * @property llKeywords Keywords to lowlight.
 * @property userPrompt The user prompt.
 * @property enableRerank Whether to enable reranking.
 */
@Serializable
data class QueryParamCache(
    val mode: String,
    val responseType: String?,
    val topK: Int,
    val chunkTopK: Int,
    val maxEntityTokens: Int,
    val maxRelationTokens: Int,
    val maxTotalTokens: Int,
    val hlKeywords: String,
    val llKeywords: String,
    val userPrompt: String,
    val enableRerank: Boolean,
)

/**
 * Represents the data stored in the cache.
 * @property argsHash The hash of the arguments.
 * @property content The content of the cache.
 * @property prompt The prompt of the cache.
 * @property mode The query mode.
 * @property cacheType The type of the cache.
 * @property queryParam The query parameters.
 * @property historyMessages The history messages.
 */
@Serializable
data class CacheData(
    val argsHash: String,
    val content: String,
    val prompt: String,
    val mode: String,
    val cacheType: String,
    val queryParam: QueryParamCache? = null,
    val historyMessages: List<Map<String, String>>? = null,
)
