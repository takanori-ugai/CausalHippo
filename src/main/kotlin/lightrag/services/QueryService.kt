package lightrag.services

import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.StreamingChatModel
import io.github.oshai.kotlinlogging.KotlinLogging
import lightrag.core.QueryParam
import lightrag.core.QueryResult
import lightrag.core.types.BaseKVStorage
import lightrag.operate.NaiveQueryParams
import lightrag.operate.QueryProcessor
import lightrag.operate.naiveQuery // Added missing import

private val logger = KotlinLogging.logger {}

/**
 * Routes query requests to the appropriate query processor based on mode (KG-aware, naive, or bypass).
 */
class QueryService(
    private val storageManager: StorageManager,
    private val chatModel: ChatModel,
    private val hashingKv: BaseKVStorage?,
    private val globalConfig: Map<String, Any?>,
    private val tokenizer: (String) -> List<Int>,
    private val decoder: (List<Int>) -> String,
    private val streamingChatModel: StreamingChatModel? = null,
) {
    private val queryProcessor = processorFor(storageManager)

    private fun processorFor(stores: StorageManager) =
        QueryProcessor(
            knowledgeGraphInst = stores.chunkEntityRelationGraph,
            entitiesVdb = stores.entitiesVdb,
            relationshipsVdb = stores.relationshipsVdb,
            textChunksDb = stores.textChunks,
            chatModel = chatModel,
            streamingChatModel = streamingChatModel,
            hashingKv = hashingKv,
            globalConfig = globalConfig,
        )

    internal suspend fun queryPublishedGraph(
        stores: StorageManager,
        query: String,
        param: QueryParam,
    ): QueryResult? = processorFor(stores).kgQuery(query, param, chunksVdb = stores.chunksVdb)

    /**
     * Queries the LightRAG system.
     * @param query The query to execute.
     * @param param The query parameters.
     * @return The query result.
     */
    suspend fun query(
        query: String,
        param: QueryParam,
    ): QueryResult? =
        when (param.mode) {
            "local", "global", "hybrid", "mix" -> {
                queryProcessor.kgQuery(
                    query = query,
                    queryParam = param,
                    chunksVdb = storageManager.chunksVdb,
                )
            }

            "naive" -> {
                naiveQuery(
                    NaiveQueryParams(
                        query = query,
                        chunksVdb = storageManager.chunksVdb,
                        queryParam = param,
                        globalConfig = globalConfig,
                        chatModel = chatModel,
                        streamingChatModel = streamingChatModel,
                        hashingKv = hashingKv,
                        tokenizer = tokenizer,
                        decoder = decoder,
                    ),
                )
            }

            "bypass" -> {
                val response = chatModel.chat(query)
                QueryResult(content = response)
            }

            else -> {
                logger.error { "Unsupported query mode: ${param.mode}" }
                null
            }
        }
}
