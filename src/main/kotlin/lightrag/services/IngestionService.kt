package lightrag.services

import io.github.oshai.kotlinlogging.KotlinLogging
import lightrag.core.types.DocStatus
import lightrag.utils.generateTrackId
import java.time.Instant

private val logger = KotlinLogging.logger {}

/**
 * Handles ingesting documents into LightRAG by tracking status and delegating processing to [DocumentProcessor].
 */
class IngestionService(
    private val storageManager: StorageManager,
    private val globalConfig: Map<String, Any?>,
    private val tokenizer: (String) -> List<Int>,
    private val decoder: (List<Int>) -> String,
) {
    private val documentProcessor =
        DocumentProcessor(
            storageManager = storageManager,
            globalConfig = globalConfig,
            tokenizer = tokenizer,
            decoder = decoder,
        )

    companion object {
        private const val SUMMARY_PREVIEW_LENGTH = 100
    }

    /**
     * Inserts a single document.
     * @param input The document to insert.
     * @param fileSource The source of the file.
     * @return A track ID for the insertion.
     */
    suspend fun insert(
        input: String,
        fileSource: String? = null,
    ): String {
        val fileSources = fileSource?.let { listOf(it) }
        return insert(listOf(input), fileSources)
    }

    /**
     * Inserts multiple documents.
     * @param input The documents to insert.
     * @param fileSources The sources of the files.
     * @return A track ID for the insertion.
     */
    suspend fun insert(
        input: List<String>,
        fileSources: List<String>? = null,
    ): String {
        val trackId = generateTrackId("insert")
        pipelineEnqueueDocuments(input, trackId, fileSources)
        documentProcessor.pipelineProcessEnqueueDocuments()
        return trackId
    }

    private suspend fun pipelineEnqueueDocuments(
        input: List<String>,
        trackId: String,
        filePaths: List<String>? = null,
    ): String {
        val uniqueContent = collectUniqueContent(input, filePaths)
        val uniqueNewDocIds = storageManager.docStatusStorage.filterKeys(uniqueContent.keys)

        updateExistingDocs(uniqueContent, uniqueNewDocIds, trackId, filePaths)

        val newDocs = buildNewDocs(uniqueContent, uniqueNewDocIds, trackId)
        if (newDocs.isEmpty()) return trackId

        saveNewDocuments(uniqueContent, uniqueNewDocIds, newDocs)

        return trackId
    }

    private fun collectUniqueContent(
        input: List<String>,
        filePaths: List<String>?,
    ): Map<String, Pair<String, String>> {
        val effectiveFilePaths =
            input.mapIndexed { index, _ ->
                filePaths?.getOrNull(index)
                    ?: filePaths?.lastOrNull()
                    ?: "unknown_source"
            }
        val uniqueContent = mutableMapOf<String, Pair<String, String>>()
        for (i in input.indices) {
            val content = input[i]
            val path = effectiveFilePaths[i]
            val md5 = lightrag.utils.computeMd5(content)
            uniqueContent[md5] = content to path
        }
        return uniqueContent
    }

    private suspend fun updateExistingDocs(
        uniqueContent: Map<String, Pair<String, String>>,
        uniqueNewDocIds: Set<String>,
        trackId: String,
        filePaths: List<String>?,
    ) {
        val existingDocIds = uniqueContent.keys - uniqueNewDocIds
        if (filePaths == null || existingDocIds.isEmpty()) return

        val updates = mutableMapOf<String, Map<String, Any>>()
        existingDocIds.forEach { docId ->
            val (_, path) = uniqueContent[docId] ?: return@forEach
            updates[docId] =
                mapOf(
                    "status" to DocStatus.PENDING.value,
                    "track_id" to trackId,
                    "file_path" to path,
                    "updated_at" to Instant.now().toString(),
                )
        }
        if (updates.isEmpty()) return

        storageManager.docStatusStorage.upsert(updates)
        storageManager.fullDocs.upsert(
            updates.mapValues { (docId, _) ->
                val (_, path) = uniqueContent[docId]!!
                mapOf("file_path" to path)
            },
        )
    }

    private fun buildNewDocs(
        uniqueContent: Map<String, Pair<String, String>>,
        uniqueNewDocIds: Set<String>,
        trackId: String,
    ): MutableMap<String, Map<String, Any>> {
        val newDocs = mutableMapOf<String, Map<String, Any>>()
        uniqueNewDocIds.forEach { docId ->
            val (content, path) = uniqueContent[docId]!!
            newDocs[docId] =
                mapOf(
                    "status" to DocStatus.PENDING.value,
                    "content_summary" to (content.take(SUMMARY_PREVIEW_LENGTH) + "..."),
                    "content_length" to content.length.toString(),
                    "created_at" to Instant.now().toString(),
                    "updated_at" to Instant.now().toString(),
                    "file_path" to path,
                    "track_id" to trackId,
                )
        }
        return newDocs
    }

    private suspend fun saveNewDocuments(
        uniqueContent: Map<String, Pair<String, String>>,
        uniqueNewDocIds: Set<String>,
        newDocs: Map<String, Map<String, Any>>,
    ) {
        val fullDocsData =
            uniqueNewDocIds.associateWith { docId ->
                val (content, path) = uniqueContent[docId]!!
                mapOf("content" to content, "file_path" to path)
            }
        storageManager.fullDocs.upsert(fullDocsData)

        storageManager.docStatusStorage.upsert(newDocs)
    }

    /**
     * Rebuilds the derived storage if it is empty.
     */
    suspend fun rebuildDerivedStorageIfEmpty() {
        val processedDocs = storageManager.docStatusStorage.getDocsByStatus(DocStatus.PROCESSED)
        val graphEmpty = storageManager.chunkEntityRelationGraph.getAllNodes().isEmpty()
        val chunksEmpty = storageManager.textChunks.isEmpty()

        if (processedDocs.isEmpty() || (!graphEmpty && !chunksEmpty)) {
            return
        }

        logger.warn { "Derived stores empty but processed docs exist. Rebuilding graph/vector/kvs from persisted full_docs." }

        // Clear derived stores
        storageManager.chunkEntityRelationGraph.drop()
        storageManager.chunksVdb.drop()
        storageManager.entitiesVdb.drop()
        storageManager.relationshipsVdb.drop()
        storageManager.fullEntities.drop()
        storageManager.fullRelations.drop()
        storageManager.textChunks.drop()

        // Mark processed docs back to pending and re-run pipeline
        val resetStatuses =
            processedDocs.keys.associateWith {
                mapOf(
                    "status" to DocStatus.PENDING.value,
                    "updated_at" to Instant.now().toString(),
                )
            }
        storageManager.docStatusStorage.upsert(resetStatuses)
        documentProcessor.pipelineProcessEnqueueDocuments()
    }

    /**
     * Gets the processing status of the documents.
     * @return A map of the status counts.
     */
    suspend fun getProcessingStatus(): Map<String, Int> = storageManager.docStatusStorage.getStatusCounts()

    /**
     * Deletes a document by its ID.
     * @param docId The ID of the document to delete.
     * @return A map of the status.
     */
    suspend fun deleteByDocId(docId: String): Map<String, String> {
        storageManager.docStatusStorage.delete(listOf(docId))
        return mapOf("status" to "success", "doc_id" to docId)
    }
}
