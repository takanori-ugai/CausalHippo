package causalrag

import causalrag.causalgraph.builder.CausalGraphBuilder
import causalrag.causalgraph.builder.CausalTriple
import causalrag.causalgraph.graph.DirectedGraph
import causalrag.causalgraph.retriever.CausalPathRetriever
import causalrag.generator.llm.LLMInterface
import causalrag.generator.promptbuilder.PromptBuilder
import causalrag.reranker.CausalPathReranker
import causalrag.retriever.Bm25Retriever
import causalrag.retriever.HybridRetriever
import causalrag.retriever.SemanticRetriever
import causalrag.retriever.VectorStoreRetriever
import causalrag.utils.cosineSimilarity
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import shared.chunking.DEFAULT_INGEST_CHUNK_OVERLAP_TOKEN_SIZE
import shared.chunking.DEFAULT_INGEST_CHUNK_TOKEN_SIZE
import shared.chunking.DEFAULT_PROMPT_CONTEXT_TOKEN_BUDGET
import shared.chunking.DEFAULT_TIKTOKEN_MODEL
import shared.chunking.chunkByTokenSizeWithOverlap
import shared.config.CaualRagConfig
import shared.config.CommonRagConfigLoader
import shared.rag.CommonRag
import shared.rag.CommonVectorStorage
import shared.rag.IndexingReport
import shared.rag.spi.persistence.GraphEdgeRecord
import shared.rag.spi.persistence.GraphNodeRecord
import shared.rag.spi.persistence.GraphSnapshot
import shared.rag.spi.persistence.KvSnapshot
import shared.rag.spi.persistence.PersistenceSession
import shared.rag.spi.persistence.VectorRecord
import shared.rag.spi.persistence.VectorSnapshot
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant

private val logger = KotlinLogging.logger("CausalRAG")
private val caualRagConfigJson = Json { ignoreUnknownKeys = true }
private val wordRegex = Regex("\\w+")

/**
 * Output produced by CausalRAG query execution.
 *
 * @property answer Generated answer text.
 * @property context Retrieved passages used as supporting context.
 * @property causalPaths Retrieved causal paths supplied to generation.
 */
data class CausalRagRunResult(
    val answer: String,
    val context: List<String>,
    val causalPaths: List<List<String>>,
)

data class QueryParam(
    val topK: Int = 5,
    val maxPaths: Int = 3,
    val onlyNeedContext: Boolean = false,
    val onlyNeedCausalPaths: Boolean = false,
)

class CausalRAG(
    modelName: String = "gpt-4",
    embeddingModel: String = "all-MiniLM-L6-v2",
    graphPath: String? = null,
    indexPath: String? = null,
    configPath: String? = null,
    templateStyle: String? = null,
    embeddingApiKey: String? = null,
    dynamicWeightingEnabled: Boolean = false,
    twoPassAdaptiveEnabled: Boolean = false,
    confidenceBasedSwitchEnabled: Boolean = false,
    persistenceSession: PersistenceSession? = null,
    useUnifiedSpiForRetrievalAndIndex: Boolean = false,
    persistenceNamespacePrefix: String = "causalrag",
) : CommonRag<QueryParam, CausalRagRunResult>,
    CommonVectorStorage<Any?> {
    private val config: CaualRagConfig? = configPath?.let { loadConfig(it) }
    private val effectiveModelName = config?.modelName ?: modelName
    private val effectiveEmbeddingModel = config?.embeddingModel ?: embeddingModel
    private val effectiveGraphPath = config?.graphPath ?: graphPath
    private val effectiveIndexPath = config?.indexPath ?: indexPath
    private val effectiveLlmProvider = config?.llmProvider ?: "openai"
    private val effectiveLlmApiKey = config?.llmApiKey ?: System.getenv("OPENAI_API_KEY")
    private val effectiveLlmBaseUrl = config?.llmBaseUrl
    private val effectiveEmbeddingApiKey = config?.embeddingApiKey ?: embeddingApiKey ?: System.getenv("OPENAI_API_KEY")
    private val effectiveTemplateStyle = templateStyle ?: config?.templateStyle ?: "detailed"
    private val effectiveMinCausalMatches = config?.minCausalMatches ?: 0
    private val effectiveIngestChunkTokenSize = config?.ingestChunkTokenSize ?: DEFAULT_INGEST_CHUNK_TOKEN_SIZE
    private val effectiveIngestChunkOverlapTokenSize =
        config?.ingestChunkOverlapTokenSize ?: DEFAULT_INGEST_CHUNK_OVERLAP_TOKEN_SIZE
    private val effectivePromptContextTokenBudget =
        config?.promptContextTokenBudget ?: DEFAULT_PROMPT_CONTEXT_TOKEN_BUDGET

    private val unifiedSpiSession = persistenceSession
    private val unifiedSpiEnabled = useUnifiedSpiForRetrievalAndIndex && unifiedSpiSession != null
    private val unifiedSpiNamespacePrefix = persistenceNamespacePrefix.trim().ifEmpty { "causalrag" }
    private val unifiedSpiGraphNamespace = "${unifiedSpiNamespacePrefix}_graph"
    private val unifiedSpiVectorNamespace = "${unifiedSpiNamespacePrefix}_vector"
    private val unifiedSpiMetadataNamespace = "${unifiedSpiNamespacePrefix}_metadata"

    private val llm: LLMInterface =
        LLMInterface(
            modelName = effectiveModelName,
            provider = effectiveLlmProvider,
            apiKey = effectiveLlmApiKey,
            baseUrl = effectiveLlmBaseUrl,
        )
    internal val evaluatorLlmInterface: LLMInterface
        get() = llm
    private val indexLock = Any()
    private val graphBuilder: CausalGraphBuilder =
        CausalGraphBuilder(
            modelName = effectiveEmbeddingModel,
            graphPath = effectiveGraphPath,
            embeddingApiKey = effectiveEmbeddingApiKey,
            extractorMethod = "hybrid",
            llmInterface = llm,
            ingestChunkTokenSize = effectiveIngestChunkTokenSize,
            ingestChunkOverlapTokenSize = effectiveIngestChunkOverlapTokenSize,
        )
    private val vectorRetriever: VectorStoreRetriever =
        VectorStoreRetriever(
            embeddingModel = effectiveEmbeddingModel,
            indexPath = effectiveIndexPath,
            embeddingApiKey = effectiveEmbeddingApiKey,
        )
    private val bm25Retriever: Bm25Retriever = Bm25Retriever()
    private val graphRetriever: CausalPathRetriever = CausalPathRetriever(graphBuilder)
    private val hybridRetriever: HybridRetriever =
        HybridRetriever(
            semanticRetriever =
                if (unifiedSpiEnabled) {
                    object : SemanticRetriever {
                        override fun searchWithScores(
                            query: String,
                            topK: Int,
                        ): List<Pair<String, Double>> = retrieveContextFromUnifiedSpi(query, topK)
                    }
                } else {
                    vectorRetriever
                },
            graphRetriever,
            semanticWeight = 0.4,
            causalWeight = 0.5,
            bm25Weight = 0.1,
            minCausalMatches = effectiveMinCausalMatches,
            bm25Retriever = bm25Retriever,
            dynamicWeightingEnabled = dynamicWeightingEnabled,
            twoPassAdaptiveEnabled = twoPassAdaptiveEnabled,
            confidenceBasedSwitchEnabled = confidenceBasedSwitchEnabled,
        )
    private val reranker: CausalPathReranker = CausalPathReranker(graphRetriever)
    private val vectorStorageBackend = CausalRAGVectorStorage(vectorRetriever)

    init {
        if (unifiedSpiEnabled) {
            ensureUnifiedSpiIndexInitialized()
        } else {
            refreshBm25()
        }
    }

    val lastIndexingReport: IndexingReport
        get() = synchronized(indexLock) { graphBuilder.lastIndexingReport }

    override fun upsert(data: String) = runBlocking { aupsert(data) }

    override fun upsert(data: Collection<String>) = runBlocking { aupsert(data) }

    override suspend fun aupsert(data: String) = aupsert(listOf(data))

    override suspend fun aupsert(data: Collection<String>) {
        val documents = normalizeDocuments(data)
        if (documents.isEmpty()) {
            logger.warn { "No valid documents provided for CausalRAG upsert." }
            return
        }
        updateIndex {
            indexDocuments(documents, replace = false)
            if (unifiedSpiEnabled) syncUnifiedSpiFromLocal()
        }
    }

    fun reindex(data: String) = runBlocking { areindex(data) }

    fun reindex(data: Collection<String>) = runBlocking { areindex(data) }

    suspend fun areindex(data: String) = areindex(listOf(data))

    suspend fun areindex(data: Collection<String>) {
        val documents = normalizeDocuments(data)
        updateIndex {
            graphBuilder.clear()
            vectorRetriever.clear()
            if (documents.isNotEmpty()) indexDocuments(documents, replace = true)
            if (unifiedSpiEnabled) syncUnifiedSpiFromLocal()
        }
    }

    override fun drop() = runBlocking { adrop() }

    override suspend fun adrop() {
        updateIndex {
            graphBuilder.clear()
            vectorRetriever.clear()
            if (unifiedSpiEnabled) clearUnifiedSpiState()
        }
    }

    override fun inspectGraph(): Map<String, Any?> = runBlocking { ainspectGraph() }

    override suspend fun ainspectGraph(): Map<String, Any?> =
        synchronized(indexLock) {
            if (unifiedSpiEnabled) inspectGraphFromUnifiedSpi() else inspectGraphFromLocal()
        }

    override fun saveGraph(path: String) = runBlocking { asaveGraph(path) }

    override suspend fun asaveGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        synchronized(indexLock) { check(save(path)) { "Failed to save CausalRAG graph data to '$path'." } }
    }

    override fun loadGraph(path: String) = runBlocking { aloadGraph(path) }

    override suspend fun aloadGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        updateIndex {
            check(load(path)) { "Failed to load CausalRAG graph data from '$path'." }
        }
    }

    override fun query(
        query: String,
        param: QueryParam,
    ): CausalRagRunResult = runBlocking { aquery(query, param) }

    override suspend fun aquery(
        query: String,
        param: QueryParam,
    ): CausalRagRunResult {
        val contextOnly = param.onlyNeedContext || param.onlyNeedCausalPaths
        if (param.onlyNeedCausalPaths && !param.onlyNeedContext) {
            return CausalRagRunResult("", emptyList(), retrieveCausalPaths(query, param.maxPaths))
        }
        val evidence = preparePrompt(query, param.topK, param.maxPaths, summarize = !contextOnly)
        val answer =
            if (contextOnly) "" else llm.generate(evidence.text, jsonMode = requiresJsonResponseFormat(effectiveTemplateStyle))
        val paths = if (contextOnly && !param.onlyNeedCausalPaths) emptyList() else evidence.paths
        return CausalRagRunResult(answer, evidence.passages, paths)
    }

    override suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>> = synchronized(indexLock) { vectorStorageBackend.search(query, topK) }

    override suspend fun upsert(data: Map<String, Map<String, Any?>>) =
        mutateVectors {
            vectorStorageBackend.upsertEntries(data)
        }

    override suspend fun deleteEntity(entityName: String) =
        mutateVectors {
            vectorStorageBackend.removeEntity(entityName)
        }

    override suspend fun deleteEntityRelation(entityName: String) =
        mutateVectors {
            vectorStorageBackend.removeEntityRelation(entityName)
        }

    private fun mutateVectors(mutate: () -> Unit) {
        updateIndex {
            mutate()
            if (unifiedSpiEnabled) syncUnifiedSpiFromLocal()
        }
    }

    fun run(
        query: String,
        topK: Int = 5,
    ): String = runBlocking { aquery(query, QueryParam(topK = topK)).answer }

    fun retrieveContext(
        query: String,
        topK: Int = 5,
    ): List<String> = preparePrompt(query, topK, maxPaths = 3, summarize = false).passages

    fun retrieveCausalPaths(
        query: String,
        maxPaths: Int = 3,
    ): List<List<String>> =
        synchronized(indexLock) {
            if (unifiedSpiEnabled) {
                retrieveCausalPathsFromUnifiedSpi(query, maxPaths.coerceAtLeast(1))
            } else {
                graphRetriever.retrievePaths(query, maxPaths = maxPaths.coerceAtLeast(1))
            }
        }

    private fun preparePrompt(
        query: String,
        topK: Int,
        maxPaths: Int,
        summarize: Boolean,
    ): PromptBuilder.BudgetedPrompt {
        val safeTopK = topK.coerceAtLeast(1)
        val (passages, paths) =
            synchronized(indexLock) {
                val candidates = hybridRetriever.retrieve(query, topK = safeTopK)
                val ranked = reranker.rerank(query, candidates).map { it.first }.take(safeTopK)
                ranked to retrieveCausalPaths(query, maxPaths)
            }
        return PromptBuilder(templateStyle = effectiveTemplateStyle, llmInterface = llm)
            .buildBudgetedPrompt(query, passages, paths, effectivePromptContextTokenBudget, summarize)
    }

    private fun loadConfig(configPath: String): CaualRagConfig {
        val path = Path.of(configPath)
        require(Files.exists(path)) { "Config file not found: $configPath" }
        return try {
            val content = Files.readString(path)
            CommonRagConfigLoader.parseOrNull(content)?.toCaualRagConfig()
                ?: caualRagConfigJson.decodeFromString(CaualRagConfig.serializer(), content)
        } catch (ex: IOException) {
            logger.error(ex) { "Failed to load config from $configPath" }
            throw ex
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Failed to load config from $configPath" }
            throw ex
        }
    }

    private fun indexDocuments(
        documents: List<String>,
        replace: Boolean,
    ) {
        val prepared = chunkDocumentsForIngest(documents)
        val newPassages = if (replace) prepared else prepared.distinct() - vectorRetriever.getPassages().toSet()
        if (newPassages.isEmpty()) return
        if (replace) vectorRetriever.indexCorpus(newPassages) else vectorRetriever.appendCorpus(newPassages)
        graphBuilder.indexDocuments(newPassages)
    }

    private fun refreshBm25() {
        bm25Retriever.clear()
        val passages = vectorRetriever.getPassages()
        if (passages.isNotEmpty()) bm25Retriever.indexDocuments(passages)
    }

    private fun updateIndex(update: () -> Unit) {
        synchronized(indexLock) {
            val graph = graphBuilder.snapshotState()
            val vectors = vectorRetriever.snapshotState()
            val session = unifiedSpiSession?.takeIf { unifiedSpiEnabled }
            val graphStore = session?.graph(unifiedSpiGraphNamespace)
            val vectorStore = session?.vector(unifiedSpiVectorNamespace)
            val kvStore = session?.kv(unifiedSpiMetadataNamespace)
            val persistedGraph = graphStore?.snapshot()
            val persistedVectors = vectorStore?.snapshot()
            val persistedMetadata = kvStore?.snapshot()
            try {
                update()
                refreshBm25()
                hybridRetriever.clearCache()
            } catch (failure: Exception) {
                graphBuilder.restoreState(graph)
                vectorRetriever.restoreState(vectors)
                for (rollback in listOf<() -> Unit>(
                    { refreshBm25() },
                    { if (persistedGraph != null) graphStore.restore(persistedGraph) },
                    { if (persistedVectors != null) vectorStore.restore(persistedVectors) },
                    { if (persistedMetadata != null) kvStore.restore(persistedMetadata) },
                )) {
                    try {
                        rollback()
                    } catch (rollbackFailure: Exception) {
                        if (rollbackFailure !== failure) failure.addSuppressed(rollbackFailure)
                    }
                }
                throw failure
            }
        }
    }

    private fun chunkDocumentsForIngest(documents: List<String>): List<String> =
        documents
            .asSequence()
            .filter { it.isNotBlank() }
            .flatMap { doc ->
                chunkByTokenSizeWithOverlap(
                    content = doc,
                    chunkTokenSize = effectiveIngestChunkTokenSize,
                    chunkOverlapTokenSize = effectiveIngestChunkOverlapTokenSize,
                    model = DEFAULT_TIKTOKEN_MODEL,
                ).asSequence()
            }.map { it.content }
            .filter { it.isNotBlank() }
            .toList()

    private fun save(dir: String): Boolean =
        if (unifiedSpiEnabled) {
            saveUnifiedSpiCheckpoint(dir)
        } else {
            saveLocalSnapshot(dir)
        }

    private fun saveLocalSnapshot(dir: String): Boolean =
        try {
            val path = Path.of(dir)
            Files.createDirectories(path)
            val graphSaved = graphBuilder.save(path.resolve("graph.json").toString())
            val vectorsSaved = vectorRetriever.saveIndex(path.resolve("vector_cache").toString())
            graphSaved && vectorsSaved
        } catch (ex: IOException) {
            logger.error(ex) { "Failed to save pipeline data to $dir" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Failed to save pipeline data to $dir" }
            false
        }

    private fun saveUnifiedSpiCheckpoint(dir: String): Boolean {
        val session = unifiedSpiSession ?: return false
        return try {
            runBlocking { session.checkpoint(dir) }
            true
        } catch (ex: IOException) {
            logger.error(ex) { "Failed to save unified SPI checkpoint to $dir" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Failed to save unified SPI checkpoint to $dir" }
            false
        }
    }

    private fun load(dir: String): Boolean =
        if (unifiedSpiEnabled) {
            loadUnifiedSpiCheckpoint(dir)
        } else {
            loadLocalSnapshot(dir)
        }

    private fun loadLocalSnapshot(dir: String): Boolean =
        try {
            val path = Path.of(dir)
            val graphLoaded = graphBuilder.load(path.resolve("graph.json").toString())
            val vectorsLoaded = vectorRetriever.loadIndex(path.resolve("vector_cache").toString())
            graphLoaded && vectorsLoaded
        } catch (ex: IOException) {
            logger.error(ex) { "Failed to load pipeline data from $dir" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Failed to load pipeline data from $dir" }
            false
        }

    private fun loadUnifiedSpiCheckpoint(dir: String): Boolean {
        val session = unifiedSpiSession ?: return false
        return try {
            runBlocking { session.restore(dir) }
            hydrateLocalIndexFromUnifiedSpi()
            true
        } catch (ex: IOException) {
            logger.error(ex) { "Failed to restore unified SPI checkpoint from $dir" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Failed to restore unified SPI checkpoint from $dir" }
            false
        }
    }

    private fun normalizeDocuments(data: Collection<String>): List<String> =
        data
            .map { it.trim() }
            .filter { it.isNotBlank() }

    private fun requiresJsonResponseFormat(style: String): Boolean = style.equals("experiments", ignoreCase = true)

    private fun inspectGraphFromLocal(): Map<String, Any?> {
        val graph = graphBuilder.getGraph()
        val nodeText = graphBuilder.nodeText
        val nodes =
            graph
                .nodes()
                .sorted()
                .map { nodeId ->
                    mapOf(
                        "id" to nodeId,
                        "text" to (nodeText[nodeId] ?: nodeId),
                        "in_degree" to graph.inDegree(nodeId),
                        "out_degree" to graph.outDegree(nodeId),
                    )
                }
        val edges =
            graph.edges().map { edge ->
                mapOf(
                    "source" to edge.from,
                    "target" to edge.to,
                    "weight" to edge.weight,
                )
            }
        return mapOf(
            "nodes" to nodes,
            "edges" to edges,
            "metadata" to
                mapOf(
                    "nodeCount" to nodes.size,
                    "edgeCount" to edges.size,
                    "hasCycle" to graph.hasCycle(),
                ),
        )
    }

    private fun inspectGraphFromUnifiedSpi(): Map<String, Any?> {
        val session = unifiedSpiSession ?: return inspectGraphFromLocal()
        val snapshot = session.graph(unifiedSpiGraphNamespace).snapshot()
        val (graph, nodeText) = graphFromSnapshot(snapshot)
        val nodeIds = (snapshot.nodes.map { it.id } + graph.nodes()).toSet().sorted()
        val nodes =
            nodeIds.map { nodeId ->
                mapOf(
                    "id" to nodeId,
                    "text" to (nodeText[nodeId] ?: nodeId),
                    "in_degree" to graph.inDegree(nodeId),
                    "out_degree" to graph.outDegree(nodeId),
                )
            }
        val edges =
            snapshot.edges.map { edge ->
                mapOf(
                    "source" to edge.source,
                    "target" to edge.target,
                    "weight" to ((edge.data["weight"] as? Number)?.toDouble() ?: 1.0),
                )
            }
        return mapOf(
            "nodes" to nodes,
            "edges" to edges,
            "metadata" to
                mapOf(
                    "nodeCount" to nodes.size,
                    "edgeCount" to edges.size,
                    "hasCycle" to graph.hasCycle(),
                    "persistenceBackend" to session.backendId,
                ),
        )
    }

    private fun ensureUnifiedSpiIndexInitialized() {
        val session = unifiedSpiSession ?: return
        if (!unifiedSpiEnabled) return
        val graphSnapshot = session.graph(unifiedSpiGraphNamespace).snapshot()
        val vectorSnapshot = session.vector(unifiedSpiVectorNamespace).snapshot()
        val hasSpiState =
            graphSnapshot.nodes.isNotEmpty() || graphSnapshot.edges.isNotEmpty() || vectorSnapshot.records.isNotEmpty()
        if (hasSpiState) {
            hydrateLocalIndexFromUnifiedSpi()
            return
        }

        val localHasState = graphBuilder.getGraph().numberOfEdges() > 0 || vectorRetriever.getPassages().isNotEmpty()
        if (localHasState) {
            syncUnifiedSpiFromLocal()
        }
    }

    private fun clearUnifiedSpiState() {
        val session = unifiedSpiSession ?: return
        session.graph(unifiedSpiGraphNamespace).restore(GraphSnapshot())
        session.vector(unifiedSpiVectorNamespace).restore(VectorSnapshot(metric = "cosine"))
        session.kv(unifiedSpiMetadataNamespace).restore(KvSnapshot())
    }

    private fun syncUnifiedSpiFromLocal() {
        val session = unifiedSpiSession ?: return
        val graph = graphBuilder.getGraph()
        val nodeText = graphBuilder.nodeText

        val graphSnapshot =
            GraphSnapshot(
                nodes =
                    graph
                        .nodes()
                        .sorted()
                        .map { nodeId ->
                            GraphNodeRecord(
                                id = nodeId,
                                data = mapOf("text" to (nodeText[nodeId] ?: nodeId)),
                            )
                        },
                edges =
                    graph.edges().map { edge ->
                        GraphEdgeRecord(
                            source = edge.from,
                            target = edge.to,
                            data = mapOf("weight" to edge.weight),
                        )
                    },
                metadata =
                    mapOf(
                        "nodeCount" to graph.numberOfNodes(),
                        "edgeCount" to graph.numberOfEdges(),
                    ),
            )
        session.graph(unifiedSpiGraphNamespace).restore(graphSnapshot)

        val passages = vectorRetriever.getPassages()
        val vectors = vectorRetriever.getVectors()
        val metadata = vectorRetriever.getMetadata()
        val records =
            passages.mapIndexedNotNull { index, passage ->
                val id =
                    metadata
                        .getOrNull(index)
                        ?.get("id")
                        ?.toString()
                        ?.takeIf { it.isNotBlank() } ?: index.toString()
                val vector = vectors.getOrNull(index)?.toList() ?: vectorRetriever.encodeForQuery(passage)?.toList()
                if (vector.isNullOrEmpty()) {
                    null
                } else {
                    val entryMetadata = metadata.getOrNull(index)?.toMutableMap() ?: mutableMapOf()
                    entryMetadata.putIfAbsent("id", id)
                    entryMetadata.putIfAbsent("passage", passage)
                    VectorRecord(
                        id = id,
                        vector = vector,
                        metadata = entryMetadata,
                    )
                }
            }

        val dimensions = records.firstOrNull()?.vector?.size
        session
            .vector(unifiedSpiVectorNamespace, dimensions = dimensions, metric = "cosine")
            .restore(
                VectorSnapshot(
                    dimensions = dimensions,
                    metric = "cosine",
                    records = records,
                    metadata = mapOf("recordCount" to records.size),
                ),
            )
        session.kv(unifiedSpiMetadataNamespace).putAll(
            mapOf(
                "updatedAt" to Instant.now().toString(),
                "recordCount" to records.size,
                "nodeCount" to graphSnapshot.nodes.size,
                "edgeCount" to graphSnapshot.edges.size,
                "embeddingIdentity" to vectorRetriever.embeddingIdentity,
            ),
        )
    }

    private fun hydrateLocalIndexFromUnifiedSpi() {
        val session = unifiedSpiSession ?: return
        val graphSnapshot = session.graph(unifiedSpiGraphNamespace).snapshot()
        graphBuilder.clear()
        if (graphSnapshot.edges.isNotEmpty()) {
            val nodeLabels = graphSnapshot.nodes.associate { node -> node.id to (node.data["text"]?.toString() ?: node.id) }
            val triples =
                graphSnapshot.edges.map { edge ->
                    CausalTriple(
                        cause = nodeLabels[edge.source] ?: edge.source,
                        effect = nodeLabels[edge.target] ?: edge.target,
                        confidence = (edge.data["weight"] as? Number)?.toDouble(),
                    )
                }
            graphBuilder.addTriples(triples)
        }

        val vectorSnapshot = session.vector(unifiedSpiVectorNamespace).snapshot()
        val records = vectorSnapshot.records.sortedBy { it.id }
        if (records.isEmpty()) {
            vectorRetriever.clear()
            bm25Retriever.clear()
            return
        }

        val passages =
            records.map { record ->
                record.metadata["passage"]?.toString()
                    ?: record.metadata["content"]?.toString()
                    ?: record.id
            }
        val metadata =
            records.map { record ->
                record.metadata.entries
                    .mapNotNull { (key, value) -> value?.let { key to it } }
                    .toMap()
            }
        val ids = records.map { it.id }

        val identity = session.kv(unifiedSpiMetadataNamespace).snapshot().entries["embeddingIdentity"]
        if (identity == null) {
            vectorRetriever.indexCorpus(passages, metadata.mapIndexed { index, row -> row + ("id" to ids[index]) }, ids)
            syncUnifiedSpiFromLocal()
        } else {
            check(identity == vectorRetriever.embeddingIdentity) { "Unified CausalRAG index uses a different embedding model or provider" }
            vectorRetriever.restoreState(
                VectorStoreRetriever.IndexState(
                    passages,
                    metadata.mapIndexed { index, row -> row + ("id" to ids[index]) },
                    records.map { it.vector.toDoubleArray() },
                ),
            )
        }
        refreshBm25()
        hybridRetriever.clearCache()
    }

    private fun retrieveContextFromUnifiedSpi(
        query: String,
        topK: Int,
    ): List<Pair<String, Double>> {
        val session = unifiedSpiSession ?: return emptyList()
        val queryEmbedding = vectorRetriever.encodeForQuery(query) ?: return emptyList()
        return try {
            session
                .vector(unifiedSpiVectorNamespace, dimensions = queryEmbedding.size, metric = "cosine")
                .query(queryEmbedding.toList(), topK = topK.coerceAtLeast(1))
                .map { record ->
                    val passage = record.metadata["passage"]?.toString() ?: record.metadata["content"]?.toString() ?: record.id
                    passage to cosineSimilarity(queryEmbedding, record.vector.toDoubleArray())
                }
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Unified SPI vector retrieval failed; returning empty context." }
            emptyList()
        }
    }

    private fun retrievePathNodesFromUnifiedSpi(
        query: String,
        topK: Int = 5,
        maxHops: Int = 2,
    ): List<String> {
        val session = unifiedSpiSession ?: return emptyList()
        val snapshot = session.graph(unifiedSpiGraphNamespace).snapshot()
        val (graph, nodeText) = graphFromSnapshot(snapshot)
        if (nodeText.isEmpty()) return emptyList()

        val queryEmbedding = vectorRetriever.encodeForQuery(query)
        val rankedNodes =
            nodeText
                .entries
                .map { (nodeId, text) ->
                    val score =
                        if (queryEmbedding != null) {
                            val nodeEmbedding = vectorRetriever.encodeForQuery(text)
                            if (nodeEmbedding != null && nodeEmbedding.size == queryEmbedding.size) {
                                cosineSimilarity(queryEmbedding, nodeEmbedding)
                            } else {
                                lexicalMatchScore(query, text)
                            }
                        } else {
                            lexicalMatchScore(query, text)
                        }
                    nodeId to score
                }.sortedByDescending { it.second }

        val seeds = rankedNodes.take(topK).map { it.first }
        val pathNodes = mutableSetOf<String>()
        for (nodeId in seeds) {
            pathNodes.add(nodeId)
            pathNodes.addAll(expandDescendants(graph, nodeId, maxHops))
            pathNodes.addAll(expandAncestors(graph, nodeId, maxHops))
        }
        return pathNodes.toList()
    }

    private fun retrieveCausalPathsFromUnifiedSpi(
        query: String,
        maxPaths: Int,
    ): List<List<String>> {
        val session = unifiedSpiSession ?: return emptyList()
        val snapshot = session.graph(unifiedSpiGraphNamespace).snapshot()
        val (graph, nodeText) = graphFromSnapshot(snapshot)
        val relevantNodes = retrievePathNodesFromUnifiedSpi(query, topK = 5, maxHops = 1)
        if (relevantNodes.size < 2) return emptyList()

        val paths = mutableListOf<Pair<List<String>, List<String>>>()
        val maxTotal = maxPaths.coerceAtLeast(1) * 3
        for (i in relevantNodes.indices) {
            for (j in i + 1 until relevantNodes.size) {
                if (paths.size >= maxTotal) break
                val src = relevantNodes[i]
                val tgt = relevantNodes[j]
                if (src == tgt) continue
                for ((start, end) in listOf(src to tgt, tgt to src)) {
                    if (paths.size >= maxTotal) break
                    val found = graph.findPaths(start, end, maxDepth = 4, limit = maxTotal - paths.size)
                    for (path in found) {
                        if (path.size >= 2) {
                            val labeledPath = path.map { nodeId -> nodeText[nodeId] ?: nodeId }
                            paths.add(path to labeledPath)
                            if (paths.size >= maxTotal) break
                        }
                    }
                }
            }
        }
        return paths.sortedBy { it.first.size }.take(maxPaths.coerceAtLeast(1)).map { it.second }
    }

    private fun graphFromSnapshot(snapshot: GraphSnapshot): Pair<DirectedGraph, Map<String, String>> {
        val graph = DirectedGraph()
        val nodeText = mutableMapOf<String, String>()
        snapshot.nodes.forEach { node ->
            nodeText[node.id] = node.data["text"]?.toString() ?: node.id
        }
        snapshot.edges.forEach { edge ->
            val weight = (edge.data["weight"] as? Number)?.toDouble() ?: 1.0
            graph.addEdge(edge.source, edge.target, weight)
            nodeText.putIfAbsent(edge.source, edge.source)
            nodeText.putIfAbsent(edge.target, edge.target)
        }
        return graph to nodeText
    }

    private fun expandDescendants(
        graph: DirectedGraph,
        node: String,
        maxHops: Int,
    ): Set<String> {
        val descendants = mutableSetOf<String>()
        var frontier = setOf(node)
        repeat(maxHops.coerceAtLeast(0)) {
            val next = mutableSetOf<String>()
            for (current in frontier) {
                next.addAll(graph.successors(current))
            }
            descendants.addAll(next)
            frontier = next
            if (frontier.isEmpty()) return@repeat
        }
        return descendants
    }

    private fun expandAncestors(
        graph: DirectedGraph,
        node: String,
        maxHops: Int,
    ): Set<String> {
        val ancestors = mutableSetOf<String>()
        var frontier = setOf(node)
        repeat(maxHops.coerceAtLeast(0)) {
            val next = mutableSetOf<String>()
            for (current in frontier) {
                next.addAll(graph.predecessors(current))
            }
            ancestors.addAll(next)
            frontier = next
            if (frontier.isEmpty()) return@repeat
        }
        return ancestors
    }

    private fun lexicalMatchScore(
        query: String,
        candidate: String,
    ): Double {
        val q = tokenizeForMatch(query)
        if (q.isEmpty()) return 0.0
        val c = tokenizeForMatch(candidate)
        if (c.isEmpty()) return 0.0
        return q.intersect(c).size.toDouble() / q.size.toDouble()
    }

    private fun tokenizeForMatch(text: String): Set<String> = wordRegex.findAll(text.lowercase()).map { it.value }.toSet()
}
