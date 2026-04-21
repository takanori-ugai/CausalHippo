package lightrag.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager

private val logger =
    io.github.oshai.kotlinlogging.KotlinLogging
        .logger {}

/**
 * Parameters for a query.
 * @property mode The query mode.
 * @property onlyNeedContext Whether to return only the context.
 * @property onlyNeedPrompt Whether to return only the prompt.
 * @property responseType The desired response type.
 * @property stream Whether to stream the response.
 * @property topK The number of top results to return.
 * @property chunkTopK The number of top chunks to return.
 * @property maxEntityTokens The maximum number of tokens for entities.
 * @property maxRelationTokens The maximum number of tokens for relations.
 * @property maxTotalTokens The maximum total number of tokens.
 * @property hlKeywords Keywords to highlight.
 * @property llKeywords Keywords to lowlight.
 * @property conversationHistory The conversation history.
 * @property userPrompt The user prompt.
 * @property enableRerank Whether to enable reranking.
 * @property includeReferences Whether to include references.
 */
@Serializable
data class QueryParam(
    val mode: String = "global",
    @SerialName("only_need_context")
    val onlyNeedContext: Boolean = false,
    @SerialName("only_need_prompt")
    val onlyNeedPrompt: Boolean = false,
    @SerialName("response_type")
    val responseType: String? = "Multiple Paragraphs",
    val stream: Boolean = false,
    @SerialName("top_k")
    val topK: Int = 40,
    @SerialName("chunk_top_k")
    val chunkTopK: Int = 20,
    @SerialName("max_entity_tokens")
    val maxEntityTokens: Int = 6000,
    @SerialName("max_relation_tokens")
    val maxRelationTokens: Int = 8000,
    @SerialName("max_total_tokens")
    val maxTotalTokens: Int = 30000,
    @SerialName("hl_keywords")
    var hlKeywords: List<String> = emptyList(),
    @SerialName("ll_keywords")
    var llKeywords: List<String> = emptyList(),
    @SerialName("conversation_history")
    val conversationHistory: List<Map<String, String>> = emptyList(),
    @SerialName("user_prompt")
    val userPrompt: String? = null,
    @SerialName("enable_rerank")
    val enableRerank: Boolean = true,
    @SerialName("include_references")
    val includeReferences: Boolean = false,
)

class LightRAG(
    private val ingestionService: IngestionService,
    private val queryService: QueryService,
    val storageManager: StorageManager,
) {
    /**
     * Inserts a single document.
     * @param input The document to insert.
     * @param fileSource The source of the file.
     * @return A track ID for the insertion.
     */
    suspend fun insert(
        input: String,
        fileSource: String? = null,
    ): String = ingestionService.insert(input, fileSource)

    /**
     * Inserts multiple documents.
     * @param input The documents to insert.
     * @param fileSources The sources of the files.
     * @return A track ID for the insertion.
     */
    suspend fun insert(
        input: List<String>,
        fileSources: List<String>? = null,
    ): String = ingestionService.insert(input, fileSources)

    /**
     * Rebuilds the derived storage if it is empty.
     */
    suspend fun rebuildDerivedStorageIfEmpty() {
        ingestionService.rebuildDerivedStorageIfEmpty()
    }

    /**
     * Queries the LightRAG system.
     * @param query The query to execute.
     * @param param The query parameters.
     * @return The query result.
     */
    suspend fun query(
        query: String,
        param: QueryParam,
    ): QueryResult? = queryService.query(query, param)

    /**
     * Gets the processing status of the documents.
     * @return A map of the status counts.
     */
    suspend fun getProcessingStatus(): Map<String, Int> = ingestionService.getProcessingStatus()

    /**
     * Deletes a document by its ID.
     * @param docId The ID of the document to delete.
     * @return A map of the status.
     */
    suspend fun deleteByDocId(docId: String): Map<String, String> = ingestionService.deleteByDocId(docId)
}
