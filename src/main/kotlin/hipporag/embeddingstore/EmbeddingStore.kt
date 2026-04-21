package hipporag.embeddingstore

import hipporag.embeddingmodel.BaseEmbeddingModel
import hipporag.utils.EmbeddingRow
import hipporag.utils.computeMdHashId
import hipporag.utils.jsonWithDefaults
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.Serializable
import java.io.File

/**
 * Simple JSON-backed embedding store keyed by hashed text IDs.
 *
 * @param embeddingModel model used to compute embeddings (required for inserts).
 * @param dbDirectory directory where the store file is persisted.
 * @param namespace prefix used to generate hash IDs.
 */
class EmbeddingStore(
    private val embeddingModel: BaseEmbeddingModel?,
    private val dbDirectory: String,
    private val namespace: String,
) {
    private val logger = KotlinLogging.logger {}

    private val filename: String

    private val hashIds = mutableListOf<String>()
    private val texts = mutableListOf<String>()
    private val embeddings = mutableListOf<DoubleArray>()

    private val hashIdToIdx = mutableMapOf<String, Int>()
    private val hashIdToRow = mutableMapOf<String, EmbeddingRow>()
    private val hashIdToText = mutableMapOf<String, String>()

    private var _textToHashId: Map<String, String> = emptyMap()

    /** Mapping from stored text content to hash IDs. */
    val textToHashId: Map<String, String> get() = _textToHashId

    init {
        val dir = File(dbDirectory)
        if (!dir.exists()) {
            logger.info { "Creating working directory: $dbDirectory" }
            dir.mkdirs()
        }

        filename = File(dbDirectory, "vdb_$namespace.json").path
        loadData()
    }

    private fun buildNodesDict(texts: List<String>): Map<String, EmbeddingRow> {
        val nodesDict = mutableMapOf<String, EmbeddingRow>()
        for (text in texts) {
            val hashId = computeMdHashId(text, prefix = "$namespace-")
            nodesDict[hashId] = EmbeddingRow(hashId = hashId, content = text)
        }
        return nodesDict
    }

    /**
     * Returns hash IDs for [texts] that are not yet present in the store.
     */
    fun getMissingStringHashIds(texts: List<String>): Map<String, EmbeddingRow> {
        val nodesDict = buildNodesDict(texts)

        if (nodesDict.isEmpty()) return emptyMap()

        val missingIds = nodesDict.keys.filter { it !in hashIdToRow.keys }
        return missingIds.associateWith { id -> nodesDict.getValue(id) }
    }

    /**
     * Inserts [texts] into the store, computing embeddings for missing entries.
     */
    fun insertStrings(texts: List<String>) {
        val cleanedTexts = texts.filter { it.isNotBlank() }
        if (cleanedTexts.size < texts.size) {
            logger.warn {
                "Skipping ${texts.size - cleanedTexts.size} blank texts for namespace '$namespace' during insert."
            }
        }
        val nodesDict = buildNodesDict(cleanedTexts)

        if (nodesDict.isEmpty()) return

        val missingIds = nodesDict.keys.filter { it !in hashIdToRow.keys }

        logger.info {
            "Inserting ${missingIds.size} new records, ${nodesDict.size - missingIds.size} records already exist."
        }

        if (missingIds.isEmpty()) return

        val textsToEncode = missingIds.map { nodesDict.getValue(it).content }
        val model = embeddingModel ?: error("Embedding model is required for insertStrings")
        val missingEmbeddings = model.batchEncode(textsToEncode)
        check(missingEmbeddings.size == textsToEncode.size) {
            "Embedding model returned ${missingEmbeddings.size} embeddings for ${textsToEncode.size} texts"
        }

        insertNew(missingIds, textsToEncode, missingEmbeddings)
    }

    /** Returns a copy of all stored rows keyed by hash ID. */
    fun getAllIdToRows(): Map<String, EmbeddingRow> = hashIdToRow.toMap()

    /** Returns all stored hash IDs. */
    fun getAllIds(): List<String> = hashIds.toList()

    /** Returns all stored text contents. */
    fun getAllTexts(): Set<String> = hashIdToRow.values.map { it.content }.toSet()

    /** Returns the stored row for [hashId]. */
    fun getRow(hashId: String): EmbeddingRow = hashIdToRow.getValue(hashId)

    /** Returns the hash ID for [text] or throws if missing. */
    fun getHashId(text: String): String = textToHashId[text] ?: error("Text not found in embedding store.")

    /** Returns rows for the provided [hashIds]. */
    fun getRows(hashIds: List<String>): Map<String, EmbeddingRow> {
        if (hashIds.isEmpty()) return emptyMap()
        return hashIds.associateWith { id -> hashIdToRow.getValue(id) }
    }

    /** Returns the embedding vector for [hashId]. */
    fun getEmbedding(hashId: String): DoubleArray {
        val idx = hashIdToIdx.getValue(hashId)
        return embeddings[idx]
    }

    /** Returns embedding vectors for the provided [hashIds]. */
    fun getEmbeddings(hashIds: List<String>): Array<DoubleArray> {
        if (hashIds.isEmpty()) return emptyArray()
        return hashIds
            .map { id ->
                val idx = hashIdToIdx.getValue(id)
                embeddings[idx]
            }.toTypedArray()
    }

    /**
     * Deletes embeddings by [hashIds] and persists the updated store.
     */
    fun delete(idsToDelete: Collection<String>) {
        val missingIds = idsToDelete.filter { it !in hashIdToIdx }
        if (missingIds.isNotEmpty()) {
            logger.warn { "Ignoring ${missingIds.size} unknown hash IDs during delete." }
        }
        val indices = idsToDelete.mapNotNull { hashIdToIdx[it] }.distinct().sortedDescending()
        for (idx in indices) {
            this.hashIds.removeAt(idx)
            this.texts.removeAt(idx)
            this.embeddings.removeAt(idx)
        }

        rebuildIndexes()
        logger.info { "Saving record after deletion." }
        saveData()
    }

    private fun insertNew(
        newHashIds: List<String>,
        newTexts: List<String>,
        newEmbeddings: Array<DoubleArray>,
    ) {
        require(newHashIds.size == newTexts.size && newTexts.size == newEmbeddings.size) {
            "Mismatched sizes: hashIds=${newHashIds.size}, texts=${newTexts.size}, embeddings=${newEmbeddings.size}"
        }
        val duplicateIds = newHashIds.filter { it in hashIdToIdx }
        require(duplicateIds.isEmpty()) {
            "Embedding store insertNew received existing hash IDs: ${duplicateIds.take(5)}" +
                if (duplicateIds.size > 5) " (and ${duplicateIds.size - 5} more)" else ""
        }
        this.hashIds.addAll(newHashIds)
        this.texts.addAll(newTexts)
        this.embeddings.addAll(newEmbeddings)

        rebuildIndexes()
        logger.info { "Saving new records." }
        saveData()
    }

    private fun loadData() {
        val file = File(filename)
        if (!file.exists()) {
            rebuildIndexes()
            return
        }

        val json = jsonWithDefaults { ignoreUnknownKeys = true }
        val data =
            runCatching {
                json.decodeFromString(EmbeddingStoreData.serializer(), file.readText())
            }.getOrElse { e ->
                logger.error(e) { "Failed to load embedding store from $filename; starting fresh." }
                rebuildIndexes()
                return
            }

        hashIds.clear()
        texts.clear()
        embeddings.clear()

        hashIds.addAll(data.hashIds)
        texts.addAll(data.texts)
        embeddings.addAll(data.embeddings.map { it.toDoubleArray() })

        rebuildIndexes()
        logger.info { "Loaded ${hashIds.size} records from $filename" }
    }

    /**
     * Saves the embedding store data to a JSON file.
     * NOTE: This approach serializes all data into a single JSON blob.
     * For very large embedding stores, this might become a performance bottleneck
     * and a more scalable persistence strategy (e.g., SQLite, append-only log)
     * should be considered.
     */
    private fun saveData() {
        val data =
            EmbeddingStoreData(
                hashIds = hashIds.toList(),
                texts = texts.toList(),
                embeddings = embeddings.map { it.toList() },
            )
        val json = jsonWithDefaults { prettyPrint = false }
        val target = File(filename)
        val tmp = File("$filename.tmp")
        tmp.writeText(json.encodeToString(EmbeddingStoreData.serializer(), data))
        try {
            java.nio.file.Files.move(
                tmp.toPath(),
                target.toPath(),
                java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                java.nio.file.StandardCopyOption.ATOMIC_MOVE,
            )
        } catch (_: java.nio.file.AtomicMoveNotSupportedException) {
            logger.warn { "Atomic move not supported; falling back to non-atomic replace." }
            java.nio.file.Files.move(
                tmp.toPath(),
                target.toPath(),
                java.nio.file.StandardCopyOption.REPLACE_EXISTING,
            )
        }
        logger.info { "Saved ${hashIds.size} records to $filename" }
    }

    private fun rebuildIndexes() {
        hashIdToIdx.clear()
        hashIdToRow.clear()
        hashIdToText.clear()

        hashIds.forEachIndexed { idx, hashId ->
            hashIdToIdx[hashId] = idx
            val text = texts[idx]
            hashIdToText[hashId] = text
            hashIdToRow[hashId] = EmbeddingRow(hashId = hashId, content = text)
        }
        val textToHash = mutableMapOf<String, String>()
        for ((hashId, text) in hashIdToText) {
            if (textToHash.containsKey(text)) {
                logger.warn { "Duplicate text key detected in embedding store; keeping last hashId." }
            }
            textToHash[text] = hashId
        }
        _textToHashId = textToHash
    }
}

@Serializable
private data class EmbeddingStoreData(
    val hashIds: List<String>,
    val texts: List<String>,
    val embeddings: List<List<Double>>,
)
