package lightrag.services

import io.github.oshai.kotlinlogging.KotlinLogging
import lightrag.core.types.DocProcessingStatus
import lightrag.core.types.DocStatus
import lightrag.operate.ChunkingResult
import lightrag.operate.chunkingByTokenSize
import lightrag.operate.extractEntities
import lightrag.operate.mergeNodesAndEdges
import lightrag.utils.computeMd5

private val logger = KotlinLogging.logger {}

/**
 * Orchestrates ingest-time processing steps: chunking documents, persisting chunks, and merging entities/relations.
 * Uses the injected tokenizer/decoder and storages provided by [StorageManager].
 */
class DocumentProcessor(
    private val storageManager: StorageManager,
    private val globalConfig: Map<String, Any?>,
    private val tokenizer: (String) -> List<Int>,
    private val decoder: (List<Int>) -> String,
) {
    companion object {
        private const val DEFAULT_CHUNK_TOKEN_SIZE = 1200
        private const val DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE = 100
    }

    /**
     * Processes all pending documents by chunking them, storing chunks, and updating the graph/vector stores.
     */
    suspend fun pipelineProcessEnqueueDocuments() {
        val pendingDocs = storageManager.docStatusStorage.getDocsByStatus(DocStatus.PENDING)
        if (pendingDocs.isEmpty()) return

        val chunkTokenSize = globalConfig["chunk_token_size"] as? Int ?: DEFAULT_CHUNK_TOKEN_SIZE
        val chunkOverlapTokenSize =
            globalConfig["chunk_overlap_token_size"] as? Int ?: DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE

        pendingDocs.forEach { (docId, status) ->
            processDocument(docId, status, chunkTokenSize, chunkOverlapTokenSize)
        }
    }

    private suspend fun processDocument(
        docId: String,
        status: DocProcessingStatus,
        chunkTokenSize: Int,
        chunkOverlapTokenSize: Int,
    ) {
        try {
            storageManager.docStatusStorage.upsert(mapOf(docId to mapOf("status" to DocStatus.PROCESSING.value)))
            val content = fetchContentOrFail(docId) ?: return
            val chunks = chunkDocument(content, chunkTokenSize, chunkOverlapTokenSize)
            val chunksData = buildChunksData(chunks, docId, status.filePath)
            persistChunks(chunksData, status)
            mergeExtractedEntities(chunksData)
            storageManager.docStatusStorage.upsert(mapOf(docId to mapOf("status" to DocStatus.PROCESSED.value)))
        } catch (e: IllegalStateException) {
            handleProcessingError(docId, e)
        } catch (e: IllegalArgumentException) {
            handleProcessingError(docId, e)
        }
    }

    private suspend fun fetchContentOrFail(docId: String): String? {
        val docData = storageManager.fullDocs.getById(docId)
        val content = docData?.get("content") as? String
        if (content != null) return content

        val msg = "Doc content missing for $docId"
        logger.warn { msg }
        storageManager.docStatusStorage.upsert(
            mapOf(
                docId to
                    mapOf(
                        "status" to DocStatus.FAILED.value,
                        "error_msg" to msg,
                    ),
            ),
        )
        // Clean up the orphaned status so we don't retry endlessly.
        storageManager.docStatusStorage.delete(listOf(docId))
        storageManager.fullDocs.delete(listOf(docId))
        return null
    }

    private fun chunkDocument(
        content: String,
        chunkTokenSize: Int,
        chunkOverlapTokenSize: Int,
    ) = chunkingByTokenSize(
        tokenizer = tokenizer,
        decoder = decoder,
        content = content,
        chunkTokenSize = chunkTokenSize,
        chunkOverlapTokenSize = chunkOverlapTokenSize,
    )

    private fun buildChunksData(
        chunks: List<ChunkingResult>,
        docId: String,
        filePath: String,
    ): MutableMap<String, Map<String, Any>> {
        val chunksData = mutableMapOf<String, Map<String, Any>>()
        chunks.forEach { chunk ->
            val chunkId = computeMd5(chunk.content)
            chunksData[chunkId] =
                mapOf(
                    "content" to chunk.content,
                    "full_doc_id" to docId,
                    "chunk_order_index" to chunk.chunkOrderIndex.toString(),
                    "tokens" to chunk.tokens.toString(),
                    "file_path" to (filePath.ifBlank { "unknown_source" }),
                )
        }
        return chunksData
    }

    private suspend fun persistChunks(
        chunksData: Map<String, Map<String, Any>>,
        status: DocProcessingStatus,
    ) {
        storageManager.textChunks.upsert(chunksData)
        val chunksVdbData =
            chunksData.mapValues { (_, v) ->
                mapOf(
                    "content" to v["content"]!!,
                    "full_doc_id" to v["full_doc_id"]!!,
                    "file_path" to status.filePath.ifBlank { "unknown_source" },
                )
            }
        storageManager.chunksVdb.upsert(chunksVdbData)
    }

    private suspend fun mergeExtractedEntities(chunksData: Map<String, Map<String, Any>>) {
        val (nodes, edges) = extractEntities(chunksData, globalConfig)

        mergeNodesAndEdges(
            nodes = nodes,
            edges = edges,
            knowledgeGraphInst = storageManager.chunkEntityRelationGraph,
            entitiesVdb = storageManager.entitiesVdb,
            relationshipsVdb = storageManager.relationshipsVdb,
            fullEntities = storageManager.fullEntities,
            fullRelations = storageManager.fullRelations,
        )
    }

    private suspend fun handleProcessingError(
        docId: String,
        e: Exception,
    ) {
        e.printStackTrace()
        storageManager.docStatusStorage.upsert(
            mapOf(
                docId to
                    mapOf(
                        "status" to DocStatus.FAILED.value,
                        "error_msg" to (e.message ?: "Unknown error"),
                    ),
            ),
        )
    }
}
