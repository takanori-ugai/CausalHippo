package com.microsoft.graphrag

import com.microsoft.graphrag.index.GraphRagConfig
import com.microsoft.graphrag.index.NoopWorkflowCallbacks
import com.microsoft.graphrag.index.defaultEmbeddingModel
import com.microsoft.graphrag.index.defaultPipeline
import com.microsoft.graphrag.index.runPipeline
import com.microsoft.graphrag.query.BasicQueryEngine
import com.microsoft.graphrag.query.CollectingQueryCallbacks
import com.microsoft.graphrag.query.DriftSearchEngine
import com.microsoft.graphrag.query.GlobalSearchEngine
import com.microsoft.graphrag.query.LocalQueryEngine
import com.microsoft.graphrag.query.QueryCallbacks
import com.microsoft.graphrag.query.QueryIndexData
import com.microsoft.graphrag.query.QueryIndexLoader
import com.microsoft.graphrag.query.QueryResult
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import shared.rag.CommonRag
import shared.rag.CommonVectorStorage
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Comparator

private const val SNAPSHOT_INPUT = ".graphrag-input"
private const val QUERY_ONLY_MARKER = ".graphrag-query-only"
internal const val LEGACY_EMBEDDING_MODEL = "text-embedding-3-small"
internal const val EMBEDDING_MODEL_FILE = ".graphrag-embedding-model"

private fun queryApiKey(): String =
    System.getenv("OPENAI_API_KEY") ?: error("OPENAI_API_KEY environment variable is required for GraphRAG queries and vector operations.")

private fun defaultGraphRagWorkingDir(): Path =
    Path.of(
        "./GraphRAG_cache_" +
            LocalDateTime
                .now()
                .format(DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss")),
    )

data class QueryParam(
    val mode: String = "global",
    val responseType: String = "JSON response (response, score, follow_up_queries)",
    val streaming: Boolean = false,
    val topK: Int = 10,
    val maxContextTokens: Int = 12000,
    val topKEntities: Int = 5,
    val topKRelationships: Int = 10,
    val communityLevel: Int? = null,
    val driftQuery: String? = null,
    val conversationHistory: List<String> = emptyList(),
    val maxIterations: Int = 3,
    val chatModelName: String? = null,
    val embeddingModelName: String? = null,
    val includeAnswer: Boolean = true,
)

class GraphRAG internal constructor(
    private val rootDir: Path,
    private val inputDir: Path,
    private val outputDir: Path,
    private val updateOutputDir: Path,
    private val defaultChatModelName: String,
    private val defaultEmbeddingModelName: String,
    private val queryEmbeddingModelFactory: (String) -> EmbeddingModel,
    private val streamingModelFactory: (String) -> OpenAiStreamingChatModel,
) : CommonRag<QueryParam, QueryResult>,
    CommonVectorStorage<Any?>,
    AutoCloseable {
    constructor(
        rootDir: Path = defaultGraphRagWorkingDir(),
        inputDir: Path = rootDir.resolve("input"),
        outputDir: Path = rootDir.resolve("output"),
        updateOutputDir: Path = rootDir.resolve("update_output"),
        defaultChatModelName: String = "gpt-4o-mini",
        defaultEmbeddingModelName: String = LEGACY_EMBEDDING_MODEL,
    ) : this(
        rootDir,
        inputDir,
        outputDir,
        updateOutputDir,
        defaultChatModelName,
        defaultEmbeddingModelName,
        queryEmbeddingModelFactory = { modelName -> defaultEmbeddingModel(queryApiKey(), modelName) },
        streamingModelFactory = { modelName ->
            OpenAiStreamingChatModel
                .builder()
                .apiKey(queryApiKey())
                .modelName(modelName)
                .build()
        },
    )

    constructor() : this(rootDir = defaultGraphRagWorkingDir())

    private val logger = KotlinLogging.logger("GraphRAG")
    private val callbacks = NoopWorkflowCallbacks()
    private val indexMutex = Mutex()
    private val indexDataLock = Any()

    @Volatile
    private var cachedIndexData: QueryIndexData? = null
    private var vectorStorageBackend: GraphRAGVectorStorage? = null

    init {
        ensureDirectories()
    }

    override fun upsert(data: String) = runBlocking { aupsert(data) }

    override fun upsert(data: Collection<String>) = runBlocking { aupsert(data) }

    override suspend fun aupsert(data: String) = aupsert(listOf(data))

    override suspend fun aupsert(data: Collection<String>) {
        val documents =
            data
                .map { it.trim() }
                .filter { it.isNotBlank() }

        if (documents.isEmpty()) {
            logger.warn { "No valid documents provided for GraphRAG upsert." }
            return
        }

        indexMutex.withLock {
            check(!Files.exists(outputDir.resolve(QUERY_ONLY_MARKER))) {
                "This GraphRAG snapshot has no source documents. Restore a full checkpoint before upserting."
            }
            withStagedDirectories(listOf(inputDir, outputDir)) { (input, output) ->
                copyDirectory(inputDir, input.content)
                persistInputDocuments(documents, input.content)

                val errors = mutableListOf<String>()
                val config =
                    GraphRagConfig(
                        rootDir,
                        input.content,
                        output.content,
                        updateOutputDir,
                        embeddingModelName = defaultEmbeddingModelName,
                        sourceInputDir = inputDir,
                    )
                runPipeline(defaultPipeline(), config, callbacks).collect { errors.addAll(it.errors.orEmpty()) }
                check(errors.isEmpty()) { "GraphRAG indexing failed: ${errors.joinToString("; ")}" }
                validateIndex(output.content)
                Files.writeString(output.content.resolve(EMBEDDING_MODEL_FILE), config.embeddingModelName)
                publishDirectories(listOf(input, output))
                invalidateLoadedState()
            }
        }
    }

    override fun drop() = runBlocking { adrop() }

    override suspend fun adrop() =
        indexMutex.withLock {
            deleteDirectory(inputDir)
            deleteDirectory(outputDir)
            deleteDirectory(updateOutputDir)
            invalidateLoadedState()
            ensureDirectories()
        }

    override fun inspectGraph(): Map<String, Any?> = runBlocking { ainspectGraph() }

    override suspend fun ainspectGraph(): Map<String, Any?> = indexMutex.withLock { inspectIndex() }

    private fun inspectIndex(): Map<String, Any?> {
        val indexData =
            runCatching { loadIndexDataOrThrow("Failed to load GraphRAG index from '$outputDir'.") }.getOrElse { ex ->
                logger.warn(ex) { "GraphRAG inspection skipped: failed to load index from '$outputDir'." }
                return emptyGraphInspection()
            }

        val nodes =
            indexData.entities.map { entity ->
                mapOf(
                    "id" to entity.id,
                    "name" to entity.name,
                    "type" to entity.type,
                    "description" to entity.description,
                    "source_chunk_id" to entity.sourceChunkId,
                    "community_ids" to entity.communityIds,
                    "text_unit_ids" to entity.textUnitIds,
                )
            }

        val edges =
            indexData.relationships.map { relationship ->
                mapOf(
                    "id" to relationship.id,
                    "source" to relationship.sourceId,
                    "target" to relationship.targetId,
                    "type" to relationship.type,
                    "weight" to relationship.weight,
                    "description" to relationship.description,
                    "source_chunk_id" to relationship.sourceChunkId,
                    "text_unit_ids" to relationship.textUnitIds,
                )
            }

        return mapOf(
            "nodes" to nodes,
            "edges" to edges,
            "metadata" to
                mapOf(
                    "nodeCount" to nodes.size,
                    "edgeCount" to edges.size,
                    "communityCount" to
                        indexData.communities
                            .map { it.communityId }
                            .toSet()
                            .size,
                    "textUnitCount" to indexData.textUnits.size,
                ),
        )
    }

    override fun saveGraph(path: String) = runBlocking { asaveGraph(path) }

    override suspend fun asaveGraph(path: String) =
        indexMutex.withLock {
            require(path.isNotBlank()) { "Snapshot path must not be blank" }
            val target =
                Path
                    .of(path)
                    .toFile()
                    .canonicalFile
                    .toPath()
            checkSnapshotPath(target)
            withStagedDirectories(listOf(target)) { (snapshot) ->
                copyDirectory(outputDir, snapshot.content)
                validateIndex(snapshot.content)
                if (!Files.exists(outputDir.resolve(QUERY_ONLY_MARKER)) && hasSourceDocuments(inputDir)) {
                    copyDirectory(inputDir, snapshot.content.resolve(SNAPSHOT_INPUT))
                }
                publishDirectories(listOf(snapshot))
            }
        }

    override fun loadGraph(path: String) = runBlocking { aloadGraph(path) }

    override suspend fun aloadGraph(path: String) =
        indexMutex.withLock {
            require(path.isNotBlank()) { "Snapshot path must not be blank" }
            val source =
                Path
                    .of(path)
                    .toFile()
                    .canonicalFile
                    .toPath()
            require(Files.isDirectory(source)) { "Graph snapshot directory not found: $source" }
            checkSnapshotPath(source)
            withStagedDirectories(listOf(inputDir, outputDir)) { (input, output) ->
                copyDirectory(source, output.content)
                val snapshotInput = output.content.resolve(SNAPSHOT_INPUT)
                if (hasSourceDocuments(snapshotInput)) {
                    copyDirectory(snapshotInput, input.content)
                    Files.deleteIfExists(output.content.resolve(QUERY_ONLY_MARKER))
                } else {
                    Files.writeString(output.content.resolve(QUERY_ONLY_MARKER), "Source documents are absent.")
                }
                deleteDirectory(snapshotInput)
                validateIndex(output.content)
                QueryIndexLoader(output.content).embeddingModelIdentity()
                publishDirectories(listOf(input, output))
                invalidateLoadedState()
            }
        }

    private fun hasSourceDocuments(directory: Path): Boolean =
        Files.isDirectory(directory) &&
            Files.walk(directory).use { paths ->
                paths.anyMatch { Files.isRegularFile(it) && Files.readString(it).isNotBlank() }
            }

    private fun checkSnapshotPath(path: Path) {
        for (live in listOf(inputDir, outputDir, updateOutputDir)) {
            val canonical = live.toFile().canonicalFile.toPath()
            require(!path.startsWith(canonical) && !canonical.startsWith(path)) {
                "GraphRAG snapshot overlaps an active index directory: $path"
            }
        }
    }

    private fun validateIndex(directory: Path) {
        val index = QueryIndexLoader(directory).load()
        val chunkIds = index.textUnits.map { it.chunkId }.toSet()
        val payload = index.vectorStore.load()
        val vectors = payload?.textEmbeddings ?: index.textEmbeddings
        require(chunkIds.isNotEmpty() && vectors.map { it.chunkId }.toSet().containsAll(chunkIds)) {
            "GraphRAG index is missing text units or aligned embeddings: $directory"
        }
        val dimension = vectors.first().vector.size
        require(dimension > 0 && vectors.all { it.vector.size == dimension && it.vector.all(Double::isFinite) }) {
            "GraphRAG index contains invalid text embeddings: $directory"
        }
        val entities = payload?.entityEmbeddings ?: index.entityEmbeddings
        require(entities.all { it.vector.size == dimension && it.vector.all(Double::isFinite) }) {
            "GraphRAG index contains invalid entity embeddings: $directory"
        }
    }

    private data class StagedDirectory(
        val target: Path,
        val staging: Path,
    ) {
        val content: Path = staging.resolve("content")
        val backup: Path = staging.resolve("previous")
    }

    private suspend fun withStagedDirectories(
        targets: List<Path>,
        action: suspend (List<StagedDirectory>) -> Unit,
    ) {
        val canonicalTargets = targets.map { it.toFile().canonicalFile.toPath() }
        require(
            canonicalTargets.indices.all { i ->
                canonicalTargets.indices.none { j -> i != j && canonicalTargets[i].startsWith(canonicalTargets[j]) }
            },
        ) { "GraphRAG index directories must be disjoint" }

        val staged = mutableListOf<StagedDirectory>()
        try {
            for (target in targets) {
                val absolute = target.toAbsolutePath().normalize()
                Files.createDirectories(absolute.parent)
                val slot = StagedDirectory(absolute, Files.createTempDirectory(absolute.parent, ".graphrag-"))
                staged.add(slot)
                Files.createDirectory(slot.content)
            }
            action(staged)
        } finally {
            for (slot in staged) {
                if (!Files.exists(slot.backup)) {
                    runCatching { deleteDirectory(slot.staging) }
                        .onFailure { logger.warn(it) { "Could not clean up GraphRAG staging directory ${slot.staging}" } }
                }
            }
        }
    }

    private fun publishDirectories(staged: List<StagedDirectory>) {
        val published = mutableListOf<StagedDirectory>()
        try {
            for (slot in staged) {
                if (Files.exists(slot.target)) Files.move(slot.target, slot.backup)
                Files.move(slot.content, slot.target)
                published.add(slot)
            }
        } catch (failure: Exception) {
            for (slot in staged.asReversed()) {
                try {
                    if (slot in published) deleteDirectory(slot.target)
                    if (Files.exists(slot.backup)) Files.move(slot.backup, slot.target)
                } catch (rollback: Exception) {
                    failure.addSuppressed(rollback)
                    logger.error(rollback) { "GraphRAG backup retained at ${slot.backup}" }
                }
            }
            throw failure
        }
        staged.forEach { slot ->
            runCatching { deleteDirectory(slot.backup) }
                .onFailure { logger.warn(it) { "Could not remove GraphRAG backup ${slot.backup}" } }
        }
    }

    override fun query(
        query: String,
        param: QueryParam,
    ): QueryResult = runBlocking { aquery(query, param) }

    @Suppress("LongMethod")
    override suspend fun aquery(
        query: String,
        param: QueryParam,
    ): QueryResult {
        val mode = param.mode.lowercase()
        val (indexData, indexedEmbeddingModel) =
            indexMutex.withLock {
                loadIndexDataOrThrow("Failed to load GraphRAG index from '$outputDir'. Run upsert() first.") to
                    resolveEmbeddingModel(param.embeddingModelName)
            }
        val callbackCollector = CollectingQueryCallbacks()
        val callbackList: List<QueryCallbacks> = listOf(callbackCollector)
        val filteredReports =
            filterCommunityReports(
                reports = indexData.communityReports,
                hierarchy = indexData.communityHierarchy,
                level = param.communityLevel,
            )

        fun buildStreamingModel(modelName: String?): OpenAiStreamingChatModel = streamingModelFactory(modelName ?: defaultChatModelName)

        val queryEmbeddingModel = queryEmbeddingModelFactory(indexedEmbeddingModel)
        val localVectorStore = indexData.vectorStore

        fun createLocalEngine(callbacks: List<QueryCallbacks>): LocalQueryEngine =
            LocalQueryEngine(
                streamingModel = buildStreamingModel(param.chatModelName),
                embeddingModel = queryEmbeddingModel,
                vectorStore = localVectorStore,
                textUnits = indexData.textUnits,
                textEmbeddings = indexData.textEmbeddings,
                entities = indexData.entities,
                entitySummaries = indexData.entitySummaries,
                relationships = indexData.relationships,
                claims = indexData.claims,
                covariates = indexData.covariates,
                communities = indexData.communities,
                communityReports = filteredReports,
                topKEntities = param.topKEntities,
                topKRelationships = param.topKRelationships,
                maxContextTokens = param.maxContextTokens,
                callbacks = callbacks,
            )

        fun createGlobalEngine(callbacks: List<QueryCallbacks>): GlobalSearchEngine =
            GlobalSearchEngine(
                streamingModel = buildStreamingModel(param.chatModelName),
                communityReports = filteredReports,
                communityHierarchy = indexData.communityHierarchy,
                communityLevel = param.communityLevel,
                callbacks = callbacks,
                responseType = param.responseType,
                maxContextTokens = param.maxContextTokens,
                maxDataTokens = param.maxContextTokens,
            )

        return when (mode) {
            "basic" -> {
                val engine =
                    BasicQueryEngine(
                        streamingModel = buildStreamingModel(param.chatModelName),
                        embeddingModel = queryEmbeddingModel,
                        vectorStore = localVectorStore,
                        textUnits = indexData.textUnits,
                        textEmbeddings = indexData.textEmbeddings,
                        topK = param.topK,
                        maxContextTokens = param.maxContextTokens,
                        callbacks = callbackList,
                    )
                if (param.streaming && param.includeAnswer) {
                    val streamed = StringBuilder()
                    engine.streamAnswer(query, param.responseType).collect { partial -> streamed.append(partial) }
                    QueryResult(
                        answer = streamed.toString(),
                        context = emptyList(),
                        contextRecords = callbackCollector.contextRecords,
                    )
                } else {
                    engine.answer(query, param.responseType, includeAnswer = param.includeAnswer)
                }
            }

            "local" -> {
                val engine = createLocalEngine(callbackList)
                if (param.streaming && param.includeAnswer) {
                    val streamed = StringBuilder()
                    engine
                        .streamAnswer(
                            question = query,
                            responseType = param.responseType,
                            conversationHistory = param.conversationHistory,
                            driftQuery = param.driftQuery,
                        ).collect { partial -> streamed.append(partial) }
                    QueryResult(
                        answer = streamed.toString(),
                        context = emptyList(),
                        contextRecords = callbackCollector.contextRecords,
                        contextText = callbackCollector.reduceContext,
                    )
                } else {
                    engine.answer(
                        question = query,
                        responseType = param.responseType,
                        conversationHistory = param.conversationHistory,
                        driftQuery = param.driftQuery,
                        includeAnswer = param.includeAnswer,
                    )
                }
            }

            "global" -> {
                val engine = createGlobalEngine(callbackList)
                if (!param.includeAnswer) return engine.retrieveContext(query, param.conversationHistory)
                if (param.streaming) {
                    val streamed = StringBuilder()
                    engine
                        .streamSearch(
                            question = query,
                            conversationHistory = param.conversationHistory,
                        ).collect { partial -> streamed.append(partial) }
                    QueryResult(
                        answer = streamed.toString(),
                        context = emptyList(),
                        contextRecords = callbackCollector.contextRecords,
                        contextText = callbackCollector.reduceContext,
                    )
                } else {
                    val result =
                        engine.search(
                            question = query,
                            conversationHistory = param.conversationHistory,
                        )
                    QueryResult(
                        answer = result.answer,
                        context = emptyList(),
                        contextRecords = result.contextRecords,
                        contextText = result.reduceContextText,
                        followUpQueries = result.followUpQueries,
                        llmCalls = result.llmCalls,
                        promptTokens = result.promptTokens,
                        outputTokens = result.outputTokens,
                        llmCallsCategories = result.llmCallsCategories,
                        promptTokensCategories = result.promptTokensCategories,
                        outputTokensCategories = result.outputTokensCategories,
                    )
                }
            }

            "drift" -> {
                if (!param.includeAnswer) {
                    val reports = createGlobalEngine(emptyList()).retrieveContext(query, param.conversationHistory)
                    val local =
                        createLocalEngine(emptyList()).answer(
                            query,
                            param.responseType,
                            param.conversationHistory,
                            includeAnswer = false,
                        )
                    val groups = reports.contextRecords.keys + local.contextRecords.keys
                    return local.copy(
                        context = emptyList(),
                        contextText = listOf(reports.contextText, local.contextText).filter { it.isNotBlank() }.joinToString("\n\n"),
                        contextRecords = groups.associateWith { reports.contextRecords[it].orEmpty() + local.contextRecords[it].orEmpty() },
                    )
                }
                val driftCollector = CollectingQueryCallbacks()
                val driftCallbacks: List<QueryCallbacks> = listOf(driftCollector)
                val localEngine = createLocalEngine(driftCallbacks)
                val globalEngine = createGlobalEngine(driftCallbacks)
                val engine =
                    DriftSearchEngine(
                        streamingModel = buildStreamingModel(param.chatModelName),
                        communityReports = filteredReports,
                        globalSearchEngine = globalEngine,
                        localQueryEngine = localEngine,
                        callbacks = driftCallbacks,
                        responseType = param.responseType,
                        maxIterations = param.maxIterations,
                    )
                if (param.streaming) {
                    val streamed = StringBuilder()
                    engine
                        .streamSearch(
                            question = query,
                            followUpQueries = param.driftQuery?.let { listOf(it) } ?: emptyList(),
                        ).collect { partial -> streamed.append(partial) }
                    QueryResult(
                        answer = streamed.toString(),
                        context = emptyList(),
                        contextRecords = driftCollector.contextRecords,
                        contextText = driftCollector.reduceContext,
                    )
                } else {
                    val result =
                        engine.search(
                            question = query,
                            followUpQueries = param.driftQuery?.let { listOf(it) } ?: emptyList(),
                        )
                    QueryResult(
                        answer = result.answer,
                        context = emptyList(),
                        contextRecords = driftCollector.contextRecords,
                        contextText = driftCollector.reduceContext,
                        llmCalls = result.llmCalls,
                        promptTokens = result.promptTokens,
                        outputTokens = result.outputTokens,
                        llmCallsCategories = result.llmCallsCategories,
                        promptTokensCategories = result.promptTokensCategories,
                        outputTokensCategories = result.outputTokensCategories,
                    )
                }
            }

            else -> {
                throw IllegalArgumentException("Unsupported GraphRAG query mode '$mode'. Use one of: basic, local, global, drift.")
            }
        }
    }

    override fun close() {
        // no-op
    }

    override suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>> = indexMutex.withLock { ensureVectorStorageBackend().query(query, topK) }

    override suspend fun upsert(data: Map<String, Map<String, Any?>>) =
        indexMutex.withLock {
            ensureVectorStorageBackend().upsert(data)
            invalidateLoadedState()
        }

    override suspend fun deleteEntity(entityName: String) =
        indexMutex.withLock {
            ensureVectorStorageBackend().deleteEntity(entityName)
            invalidateLoadedState()
        }

    override suspend fun deleteEntityRelation(entityName: String) =
        indexMutex.withLock {
            ensureVectorStorageBackend().deleteEntityRelation(entityName)
            invalidateLoadedState()
        }

    private fun ensureDirectories() {
        Files.createDirectories(rootDir)
        Files.createDirectories(inputDir)
        Files.createDirectories(outputDir)
        Files.createDirectories(updateOutputDir)
    }

    private fun createVectorStorageBackend(
        indexData: QueryIndexData,
        embeddingModel: EmbeddingModel,
    ): GraphRAGVectorStorage =
        GraphRAGVectorStorage(
            localVectorStore = indexData.vectorStore,
            textUnits = indexData.textUnits,
            embeddingModel = embeddingModel,
        )

    private suspend fun ensureVectorStorageBackend(): GraphRAGVectorStorage {
        vectorStorageBackend?.let { return it }
        val indexData = loadIndexDataOrThrow("Failed to load GraphRAG vector index from '$outputDir'. Run upsert() first.")
        val embeddingModel = queryEmbeddingModelFactory(resolveEmbeddingModel(null))
        return createVectorStorageBackend(indexData, embeddingModel).also { vectorStorageBackend = it }
    }

    private fun resolveEmbeddingModel(requested: String?): String {
        val indexed = QueryIndexLoader(outputDir).embeddingModelIdentity() ?: LEGACY_EMBEDDING_MODEL
        require(requested == null || requested == indexed) {
            "GraphRAG index uses '$indexed'; cannot query it with '$requested'. Rebuild with the requested model."
        }
        return indexed
    }

    private fun loadIndexDataOrThrow(errorMessage: String): QueryIndexData {
        cachedIndexData?.let { return it }
        return synchronized(indexDataLock) {
            cachedIndexData
                ?: runCatching { QueryIndexLoader(outputDir).load() }
                    .getOrElse { ex ->
                        throw IllegalStateException(errorMessage, ex)
                    }.also { loaded ->
                        cachedIndexData = loaded
                    }
        }
    }

    private fun invalidateLoadedState() {
        synchronized(indexDataLock) {
            cachedIndexData = null
            vectorStorageBackend = null
        }
    }

    private fun persistInputDocuments(
        documents: List<String>,
        directory: Path,
    ) {
        documents.forEach { content ->
            val contentBytes = content.toByteArray(Charsets.UTF_8)
            val contentSize = contentBytes.size.toLong()
            val digest = md5(contentBytes)
            var target = directory.resolve("doc-$digest.txt")
            var suffix = 1
            while (Files.exists(target)) {
                val sameSize = runCatching { Files.size(target) == contentSize }.getOrDefault(false)
                if (sameSize) {
                    val existing = runCatching { Files.readString(target, Charsets.UTF_8) }.getOrNull()
                    if (existing == content) {
                        break
                    }
                }
                target = directory.resolve("doc-$digest-$suffix.txt")
                suffix += 1
            }
            if (!Files.exists(target)) {
                Files.write(target, contentBytes)
            }
        }
    }

    private fun md5(value: ByteArray): String {
        val bytes = MessageDigest.getInstance("MD5").digest(value)
        return bytes.joinToString("") { "%02x".format(it) }
    }

    private fun emptyGraphInspection(): Map<String, Any?> =
        mapOf(
            "nodes" to emptyList<Map<String, Any?>>(),
            "edges" to emptyList<Map<String, Any?>>(),
            "metadata" to
                mapOf(
                    "nodeCount" to 0,
                    "edgeCount" to 0,
                    "communityCount" to 0,
                    "textUnitCount" to 0,
                ),
        )

    private fun copyDirectory(
        source: Path,
        target: Path,
    ) {
        Files.walk(source).use { stream ->
            stream.forEach { from ->
                val relative = source.relativize(from)
                val to = target.resolve(relative.toString())
                if (Files.isDirectory(from)) {
                    Files.createDirectories(to)
                } else {
                    Files.createDirectories(to.parent)
                    Files.copy(
                        from,
                        to,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.COPY_ATTRIBUTES,
                    )
                }
            }
        }
    }

    private fun deleteDirectory(path: Path) {
        if (!Files.exists(path)) return
        val normalized = path.toAbsolutePath().normalize()
        require(normalized.nameCount > 1) { "Refusing to delete unsafe path: $normalized" }
        Files.walk(normalized).use { stream ->
            stream
                .sorted(Comparator.reverseOrder())
                .forEach { Files.deleteIfExists(it) }
        }
    }

    @Suppress("ReturnCount", "LoopWithTooManyJumpStatements")
    private fun filterCommunityReports(
        reports: List<com.microsoft.graphrag.index.CommunityReport>,
        hierarchy: Map<Int, Int>,
        level: Int?,
    ): List<com.microsoft.graphrag.index.CommunityReport> {
        if (level == null || level < 0) return reports
        if (hierarchy.isEmpty()) return reports
        val cache = mutableMapOf<Int, Int>()

        fun depth(id: Int): Int {
            cache[id]?.let { return it }
            var current = id
            var d = 0
            val seen = mutableSetOf<Int>()
            while (true) {
                if (!seen.add(current)) break
                val parent = hierarchy[current] ?: break
                if (parent < 0) break
                current = parent
                d += 1
            }
            cache[id] = d
            return d
        }

        return reports.filter { depth(it.communityId) == level }
    }
}
