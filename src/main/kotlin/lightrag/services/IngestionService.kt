package lightrag.services

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
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
    private val indexMutex = Mutex()
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
    ): String =
        indexMutex.withLock {
            val trackId = generateTrackId("insert")
            pipelineEnqueueDocuments(input, trackId, fileSources)
            documentProcessor.pipelineProcessEnqueueDocuments()
            storageManager.persist()
            trackId
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
        for (docId in uniqueContent.keys - uniqueNewDocIds) {
            val (content, path) = uniqueContent.getValue(docId)
            val existing = storageManager.fullDocs.getById(docId).orEmpty()
            val status = storageManager.docStatusStorage.getById(docId)
            val currentState = status?.get("status")
            val needsRetry = currentState == DocStatus.FAILED.value || currentState == DocStatus.PROCESSING.value
            val effectivePath = if (filePaths == null) status?.get("file_path") as? String ?: path else path
            val pathChanged = filePaths != null && path != status?.get("file_path")
            if (!needsRetry && !pathChanged && existing["content"] == content) continue

            storageManager.fullDocs.upsert(mapOf(docId to (existing + mapOf("content" to content, "file_path" to effectivePath))))
            storageManager.docStatusStorage.upsert(
                mapOf(
                    docId to
                        mapOf(
                            "status" to DocStatus.PENDING.value,
                            "track_id" to trackId,
                            "file_path" to effectivePath,
                            "updated_at" to Instant.now().toString(),
                        ),
                ),
            )
        }
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
    suspend fun rebuildDerivedStorageIfEmpty() =
        indexMutex.withLock {
            val processedDocs = storageManager.docStatusStorage.getDocsByStatus(DocStatus.PROCESSED)
            val graphEmpty = storageManager.chunkEntityRelationGraph.getAllNodes().isEmpty()
            val chunksEmpty = storageManager.textChunks.isEmpty()

            if (processedDocs.isNotEmpty() && (graphEmpty || chunksEmpty)) {
                logger.warn { "Rebuilding derived stores from processed documents" }
                documentProcessor.rebuildDerivedStorage(processedDocs)
                storageManager.persist()
            }
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
    suspend fun deleteByDocId(docId: String): Map<String, String> =
        indexMutex.withLock {
            val doc = storageManager.fullDocs.getById(docId)
            val status = storageManager.docStatusStorage.getById(docId)
            if (doc == null && status == null) {
                return@withLock mapOf("status" to "not_found", "doc_id" to docId)
            }

            val retained = storageManager.docStatusStorage.getDocsByStatus(DocStatus.PROCESSED) - docId
            documentProcessor.rebuildDerivedStorage(retained)
            storageManager.persist()
            storageManager.fullDocs.delete(listOf(docId))
            storageManager.docStatusStorage.delete(listOf(docId))
            storageManager.persist()
            mapOf("status" to "success", "doc_id" to docId)
        }
}
