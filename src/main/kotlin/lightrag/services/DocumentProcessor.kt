package lightrag.services

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import lightrag.core.types.DocProcessingStatus
import lightrag.core.types.DocStatus
import lightrag.operate.ChunkingResult
import lightrag.operate.EntityExtractionResult
import lightrag.operate.RelationExtractionResult
import lightrag.operate.chunkingByTokenSize
import lightrag.operate.extractEntities
import lightrag.operate.mergeNodesAndEdges
import lightrag.utils.computeMd5
import shared.chunking.DEFAULT_TIKTOKEN_MODEL

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

    @Serializable
    private data class ExtractionSnapshot(
        val chunkIds: Set<String>,
        val nodes: Map<String, List<EntityExtractionResult>>,
        val edges: Map<String, List<RelationExtractionResult>>,
    )

    private data class PreparedDocument(
        val id: String,
        val source: Map<String, Any>,
        val chunks: Map<String, Map<String, Any>>,
        val extraction: ExtractionSnapshot,
    )

    suspend fun pipelineProcessEnqueueDocuments() {
        val pendingDocs = storageManager.docStatusStorage.getDocsByStatus(DocStatus.PENDING)
        for ((docId, status) in pendingDocs) {
            updateStatus(docId, DocStatus.PROCESSING)
            try {
                writeDocument(prepareDocument(docId, status))
                updateStatus(docId, DocStatus.PROCESSED)
            } catch (error: CancellationException) {
                throw error
            } catch (error: RuntimeException) {
                logger.error(error) { "LightRAG indexing failed for document $docId" }
                updateStatus(docId, DocStatus.FAILED, error.message ?: error.javaClass.simpleName)
            }
        }
    }

    /**
     * Rebuilds derived data from documents that will remain indexed.
     * Resolve all source contents and extractions before touching the current index.
     */
    suspend fun rebuildDerivedStorage(
        documents: Map<String, DocProcessingStatus>,
        afterPublish: suspend () -> Unit = {},
    ) {
        val prepared = documents.map { (id, status) -> prepareDocument(id, status) }
        // The staging stores are in memory and never initialized, persisted, or dropped on disk.
        val staging =
            StorageManager(
                workingDir = ".",
                embeddingModel = storageManager.chunksVdb.embeddingFunc,
                globalConfig = emptyMap(),
            )
        for (document in prepared) {
            writeDerivedDocument(document, staging)
        }
        val knownChunks = prepared.flatMap { it.chunks.keys }.toSet()
        val replacement = DerivedIndex.capture(staging, knownChunks)
        val previous = DerivedIndex.capture(storageManager, knownChunks + existingDocumentChunkIds())
        try {
            replacement.restore(storageManager)
            storageManager.persist()
            afterPublish()
        } catch (error: Exception) {
            withContext(NonCancellable) {
                runCatching {
                    previous.restore(storageManager)
                    storageManager.persist()
                }.exceptionOrNull()?.let(error::addSuppressed)
            }
            throw error
        }
    }

    private suspend fun existingDocumentChunkIds(): Set<String> {
        val chunkSize = globalConfig["chunk_token_size"] as? Int ?: DEFAULT_CHUNK_TOKEN_SIZE
        val overlap = globalConfig["chunk_overlap_token_size"] as? Int ?: DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE
        val documentIds = DocStatus.entries.flatMap { storageManager.docStatusStorage.getDocsByStatus(it).keys }.toSet()
        return documentIds
            .flatMap { id ->
                val content = storageManager.fullDocs.getById(id)?.get("content") as? String ?: return@flatMap emptyList()
                chunkDocument(content, chunkSize, overlap).map { computeMd5(it.content) }
            }.toSet()
    }

    private suspend fun prepareDocument(
        docId: String,
        status: DocProcessingStatus,
    ): PreparedDocument {
        val doc = checkNotNull(storageManager.fullDocs.getById(docId)) { "Document $docId is missing" }
        val content = checkNotNull(doc["content"] as? String) { "Document $docId has no content" }
        val chunkTokenSize = globalConfig["chunk_token_size"] as? Int ?: DEFAULT_CHUNK_TOKEN_SIZE
        val chunkOverlapTokenSize = globalConfig["chunk_overlap_token_size"] as? Int ?: DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE
        val chunks = buildChunksData(chunkDocument(content, chunkTokenSize, chunkOverlapTokenSize), docId, status.filePath)
        val cachedExtraction =
            (doc["graph_extraction"] as? String)
                ?.let { Json.decodeFromString<ExtractionSnapshot>(it) }
                ?.takeIf { it.chunkIds == chunks.keys }
        val extraction =
            cachedExtraction ?: run {
                val (nodes, edges) = extractEntities(chunks, globalConfig)
                ExtractionSnapshot(chunks.keys, nodes, edges)
            }
        return PreparedDocument(docId, doc, chunks, extraction)
    }

    private suspend fun writeDocument(document: PreparedDocument) {
        storageManager.fullDocs.upsert(
            mapOf(document.id to (document.source + ("graph_extraction" to Json.encodeToString(document.extraction)))),
        )
        writeDerivedDocument(document, storageManager)
    }

    private suspend fun writeDerivedDocument(
        document: PreparedDocument,
        stores: StorageManager,
    ) {
        mergeNodesAndEdges(
            nodes = document.extraction.nodes,
            edges = document.extraction.edges,
            knowledgeGraphInst = stores.chunkEntityRelationGraph,
            entitiesVdb = stores.entitiesVdb,
            relationshipsVdb = stores.relationshipsVdb,
            fullEntities = stores.fullEntities,
            fullRelations = stores.fullRelations,
        )
        stores.chunksVdb.upsert(document.chunks)
        stores.textChunks.upsert(document.chunks)
    }

    private suspend fun updateStatus(
        docId: String,
        status: DocStatus,
        error: String = "",
    ) {
        storageManager.docStatusStorage.upsert(
            mapOf(docId to mapOf("status" to status.value, "error_msg" to error)),
        )
    }

    private fun chunkDocument(
        content: String,
        chunkTokenSize: Int,
        chunkOverlapTokenSize: Int,
    ): List<ChunkingResult> {
        val tiktokenModel = (globalConfig["tiktoken_model"] as? String)?.ifBlank { DEFAULT_TIKTOKEN_MODEL } ?: DEFAULT_TIKTOKEN_MODEL
        return chunkingByTokenSize(
            tokenizer = tokenizer,
            decoder = decoder,
            content = content,
            chunkTokenSize = chunkTokenSize,
            chunkOverlapTokenSize = chunkOverlapTokenSize,
            tiktokenModel = tiktokenModel,
        )
    }

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
}
