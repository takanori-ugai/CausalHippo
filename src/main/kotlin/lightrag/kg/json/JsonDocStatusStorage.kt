package lightrag.kg.json

import dev.langchain4j.model.embedding.EmbeddingModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.SerializationException
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import lightrag.core.types.DocProcessingStatus
import lightrag.core.types.DocStatus
import lightrag.core.types.DocStatusStorage
import java.io.File
import kotlin.math.min

private val logger = KotlinLogging.logger {}

/**
 * A JSON-backed storage for document processing statuses.
 * @property namespace The namespace of the storage.
 * @property workspace The workspace of the storage.
 * @property globalConfig The global configuration for the storage.
 * @property embeddingFunc The embedding model to use.
 */
class JsonDocStatusStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any?> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
) : DocStatusStorage {
    private val docs = mutableMapOf<String, DocProcessingStatus>()
    private val mutex = Mutex()
    private val workingDir = File(globalConfig["working_dir"] as? String ?: "./rag_storage")
    private val file = File(workingDir, "doc_status_$namespace.json")

    private val json =
        Json {
            ignoreUnknownKeys = true
            prettyPrint = true
        }

    /**
     * Initializes the storage by loading data from the JSON file.
     */
    override suspend fun initialize() {
        if (!workingDir.exists()) {
            workingDir.mkdirs()
        }
        if (file.exists()) {
            try {
                val content = file.readText()
                if (content.isNotBlank()) {
                    val loaded = json.decodeFromString<Map<String, DocProcessingStatus>>(content)
                    mutex.withLock {
                        docs.putAll(loaded)
                    }
                    logger.info { "Loaded ${loaded.size} docs status from ${file.absolutePath}" }
                }
            } catch (e: java.io.IOException) {
                logger.error(e) { "I/O error loading DocStatus storage from ${file.absolutePath}" }
            } catch (e: SerializationException) {
                logger.error(e) { "Serialization error loading DocStatus storage from ${file.absolutePath}" }
            }
        }
    }

    /**
     * Saves the current state of the storage to the JSON file.
     */
    override suspend fun indexDoneCallback() {
        mutex.withLock {
            try {
                val content = json.encodeToString(docs)
                file.writeText(content)
                logger.debug { "Saved ${docs.size} docs status to ${file.absolutePath}" }
            } catch (e: java.io.IOException) {
                logger.error(e) { "I/O error saving DocStatus storage to ${file.absolutePath}" }
            } catch (e: SerializationException) {
                logger.error(e) { "Serialization error saving DocStatus storage to ${file.absolutePath}" }
            }
        }
    }

    /**
     * Gets a document by its ID.
     * @param id The ID of the document to get.
     * @return A map representing the document.
     */
    override suspend fun getById(id: String): Map<String, Any>? =
        mutex.withLock {
            docs[id]?.let {
                mapOf(
                    "status" to it.status.value,
                    "content_summary" to it.contentSummary,
                    "content_length" to it.contentLength,
                    "created_at" to it.createdAt,
                    "updated_at" to it.updatedAt,
                    "file_path" to it.filePath,
                    "track_id" to (it.trackId ?: ""),
                    "chunks_count" to (it.chunksCount ?: 0),
                    "error_msg" to (it.errorMsg ?: ""),
                )
            }
        }

    /**
     * Gets documents by their IDs.
     * @param ids The IDs of the documents to get.
     * @return A list of maps representing the documents.
     */
    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> = ids.mapNotNull { getById(it) }

    /**
     * Filters keys from the storage.
     * @param keys The keys to filter.
     * @return A set of the filtered keys.
     */
    override suspend fun filterKeys(keys: Set<String>): Set<String> =
        mutex.withLock {
            keys.filter { !docs.containsKey(it) }.toSet()
        }

    /**
     * Upserts data into the storage.
     * @param data The data to upsert.
     */
    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        mutex.withLock {
            data.forEach { (id, map) ->
                val existing = docs[id]
                val statusStr = map["status"] as? String ?: existing?.status?.value ?: DocStatus.PENDING.value
                val status = DocStatus.values().find { it.value == statusStr } ?: DocStatus.PENDING

                val newDoc =
                    DocProcessingStatus(
                        status = status,
                        contentSummary = map["content_summary"] as? String ?: existing?.contentSummary ?: "",
                        contentLength =
                            (map["content_length"]?.toString()?.toIntOrNull())
                                ?: existing?.contentLength ?: 0,
                        createdAt = map["created_at"] as? String ?: existing?.createdAt ?: "",
                        updatedAt = map["updated_at"] as? String ?: existing?.updatedAt ?: "",
                        filePath = map["file_path"] as? String ?: existing?.filePath ?: "",
                        trackId = map["track_id"] as? String ?: existing?.trackId,
                        errorMsg = map["error_msg"] as? String ?: existing?.errorMsg,
                    )
                docs[id] = newDoc
            }
        }
    }

    /**
     * Deletes items by their IDs.
     * @param ids The IDs of the items to delete.
     */
    override suspend fun delete(ids: List<String>) {
        mutex.withLock {
            ids.forEach { docs.remove(it) }
        }
    }

    /**
     * Drops the storage.
     * @return A map with the status of the operation.
     */
    override suspend fun drop(): Map<String, String> {
        mutex.withLock {
            docs.clear()
            if (file.exists()) {
                file.delete()
            }
        }
        return mapOf("status" to "success", "message" to "data dropped")
    }

    /**
     * Checks if the storage is empty.
     * @return True if the storage is empty, false otherwise.
     */
    override suspend fun isEmpty(): Boolean =
        mutex.withLock {
            docs.isEmpty()
        }

    /**
     * Gets the status counts.
     * @return A map of status counts.
     */
    override suspend fun getStatusCounts(): Map<String, Int> =
        mutex.withLock {
            docs.values.groupingBy { it.status.value }.eachCount()
        }

    /**
     * Gets documents by their status.
     * @param status The status of the documents to get.
     * @return A map of document IDs to document processing statuses.
     */
    override suspend fun getDocsByStatus(status: DocStatus): Map<String, DocProcessingStatus> =
        mutex.withLock {
            docs.filterValues { it.status == status }
        }

    /**
     * Gets documents by their track ID.
     * @param trackId The track ID of the documents to get.
     * @return A map of document IDs to document processing statuses.
     */
    override suspend fun getDocsByTrackId(trackId: String): Map<String, DocProcessingStatus> =
        mutex.withLock {
            docs.filterValues { it.trackId == trackId }
        }

    /**
     * Gets documents with pagination.
     * @param statusFilter The status to filter by.
     * @param page The page number.
     * @param pageSize The size of the page.
     * @param sortField The field to sort by.
     * @param sortDirection The direction to sort by.
     * @return A pair of the list of documents and the total number of documents.
     */
    override suspend fun getDocsPaginated(
        statusFilter: DocStatus?,
        page: Int,
        pageSize: Int,
        sortField: String,
        sortDirection: String,
    ): Pair<List<Pair<String, DocProcessingStatus>>, Int> =
        mutex.withLock {
            var filtered =
                if (statusFilter != null) {
                    docs.filterValues { it.status == statusFilter }.toList()
                } else {
                    docs.toList()
                }

            val total = filtered.size

            // Sorting logic (simplified)
            filtered = filtered.sortedBy { it.second.updatedAt }
            if (sortDirection == "desc") {
                filtered = filtered.reversed()
            }

            val start = (page - 1) * pageSize
            val end = min(start + pageSize, total)

            if (start >= total) {
                return@withLock Pair(emptyList(), total)
            }

            Pair(filtered.subList(start, end), total)
        }

    /**
     * Gets all status counts.
     * @return A map of all status counts.
     */
    override suspend fun getAllStatusCounts(): Map<String, Int> = getStatusCounts()

    /**
     * Gets a document by its file path.
     * @param filePath The file path of the document to get.
     * @return A map representing the document.
     */
    override suspend fun getDocByFilePath(filePath: String): Map<String, Any>? =
        mutex.withLock {
            val entry = docs.entries.find { it.value.filePath == filePath }
            entry?.let { getById(it.key) }
        }
}
