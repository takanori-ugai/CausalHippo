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
import shared.rag.CommonRag
import shared.rag.CommonVectorStorage
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Comparator

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
)

class GraphRAG(
    private val rootDir: Path = defaultGraphRagWorkingDir(),
    private val inputDir: Path = rootDir.resolve("input"),
    private val outputDir: Path = rootDir.resolve("output"),
    private val updateOutputDir: Path = rootDir.resolve("update_output"),
    private val defaultChatModelName: String = "gpt-4o-mini",
    private val defaultEmbeddingModelName: String = "text-embedding-3-small",
) : CommonRag<QueryParam, QueryResult>,
    CommonVectorStorage<Any?>,
    AutoCloseable {
    private val logger = KotlinLogging.logger("GraphRAG")
    private val callbacks = NoopWorkflowCallbacks()
    private val indexDataLock = Any()

    @Volatile
    private var cachedIndexData: QueryIndexData? = null
    private var vectorStorageBackend: GraphRAGVectorStorage? = null

    init {
        ensureDirectories()
    }

    private fun config(): GraphRagConfig = GraphRagConfig(rootDir, inputDir, outputDir, updateOutputDir)

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

        invalidateLoadedState()
        ensureDirectories()
        persistInputDocuments(documents)

        val pipeline = defaultPipeline()
        val errors = mutableListOf<String>()
        runPipeline(
            pipeline = pipeline,
            config = config(),
            callbacks = callbacks,
        ).collect { result ->
            val stepErrors = result.errors.orEmpty()
            if (stepErrors.isNotEmpty()) {
                errors.addAll(stepErrors)
            }
        }
        if (errors.isNotEmpty()) {
            throw IllegalStateException("GraphRAG indexing failed: ${errors.joinToString("; ")}")
        }
    }

    override fun drop() = runBlocking { adrop() }

    override suspend fun adrop() {
        deleteDirectory(inputDir)
        deleteDirectory(outputDir)
        deleteDirectory(updateOutputDir)
        invalidateLoadedState()
        ensureDirectories()
    }

    override fun inspectGraph(): Map<String, Any?> = runBlocking { ainspectGraph() }

    override suspend fun ainspectGraph(): Map<String, Any?> {
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

    override suspend fun asaveGraph(path: String) {
        val source = outputDir.toAbsolutePath().normalize()
        require(Files.exists(source) && Files.isDirectory(source)) {
            "GraphRAG output directory does not exist: $source"
        }

        val target = Path.of(path).toAbsolutePath().normalize()
        deleteDirectory(target)
        copyDirectory(source, target)
        logger.info { "Saved GraphRAG graph artifacts from '$source' to '$target'." }
    }

    override fun loadGraph(path: String) = runBlocking { aloadGraph(path) }

    override suspend fun aloadGraph(path: String) {
        val source = Path.of(path).toAbsolutePath().normalize()
        require(Files.exists(source) && Files.isDirectory(source)) {
            "Graph snapshot directory not found: $source"
        }
        invalidateLoadedState()
        deleteDirectory(outputDir)
        copyDirectory(source, outputDir)
        logger.info { "Loaded GraphRAG graph artifacts from '$source' into '$outputDir'." }
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
        val indexData = loadIndexDataOrThrow("Failed to load GraphRAG index from '$outputDir'. Run upsert() first.")

        val apiKey = System.getenv("OPENAI_API_KEY") ?: error("OPENAI_API_KEY environment variable is required for GraphRAG query.")
        val callbackCollector = CollectingQueryCallbacks()
        val callbackList: List<QueryCallbacks> = listOf(callbackCollector)
        val filteredReports =
            filterCommunityReports(
                reports = indexData.communityReports,
                hierarchy = indexData.communityHierarchy,
                level = param.communityLevel,
            )

        fun buildStreamingModel(modelName: String?): OpenAiStreamingChatModel =
            OpenAiStreamingChatModel
                .builder()
                .apiKey(apiKey)
                .modelName(modelName ?: defaultChatModelName)
                .build()

        fun buildEmbeddingModel(modelName: String?): EmbeddingModel = defaultEmbeddingModel(apiKey, modelName ?: defaultEmbeddingModelName)

        val queryEmbeddingModel = buildEmbeddingModel(param.embeddingModelName)
        val backend = createVectorStorageBackend(indexData, queryEmbeddingModel)
        vectorStorageBackend = backend
        val localVectorStore = backend.asLocalVectorStore()

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
                if (param.streaming) {
                    val streamed = StringBuilder()
                    engine.streamAnswer(query, param.responseType).collect { partial -> streamed.append(partial) }
                    QueryResult(
                        answer = streamed.toString(),
                        context = emptyList(),
                        contextRecords = callbackCollector.contextRecords,
                    )
                } else {
                    engine.answer(query, param.responseType)
                }
            }

            "local" -> {
                val engine = createLocalEngine(callbackList)
                if (param.streaming) {
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
                    )
                }
            }

            "global" -> {
                val engine = createGlobalEngine(callbackList)
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
    ): List<Map<String, Any?>> = ensureVectorStorageBackend().query(query, topK)

    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {
        ensureVectorStorageBackend().upsert(data)
        invalidateLoadedState()
    }

    override suspend fun deleteEntity(entityName: String) {
        ensureVectorStorageBackend().deleteEntity(entityName)
        invalidateLoadedState()
    }

    override suspend fun deleteEntityRelation(entityName: String) {
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
        val apiKey =
            System.getenv("OPENAI_API_KEY") ?: error("OPENAI_API_KEY environment variable is required for GraphRAG vector operations.")
        val embeddingModel = defaultEmbeddingModel(apiKey, defaultEmbeddingModelName)
        return createVectorStorageBackend(indexData, embeddingModel).also { vectorStorageBackend = it }
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

    private fun persistInputDocuments(documents: List<String>) {
        documents.forEach { content ->
            val contentBytes = content.toByteArray(Charsets.UTF_8)
            val contentSize = contentBytes.size.toLong()
            val digest = md5(contentBytes)
            var target = inputDir.resolve("doc-$digest.txt")
            var suffix = 1
            while (Files.exists(target)) {
                val sameSize = runCatching { Files.size(target) == contentSize }.getOrDefault(false)
                if (sameSize) {
                    val existing = runCatching { Files.readString(target, Charsets.UTF_8) }.getOrNull()
                    if (existing == content) {
                        break
                    }
                }
                target = inputDir.resolve("doc-$digest-$suffix.txt")
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
