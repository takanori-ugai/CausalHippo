package lightrag.services

import dev.langchain4j.model.chat.ChatModel
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
) {
    private val queryProcessor =
        QueryProcessor(
            knowledgeGraphInst = storageManager.chunkEntityRelationGraph,
            entitiesVdb = storageManager.entitiesVdb,
            relationshipsVdb = storageManager.relationshipsVdb,
            textChunksDb = storageManager.textChunks,
            chatModel = chatModel,
            hashingKv = hashingKv,
            globalConfig = globalConfig,
        )

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
