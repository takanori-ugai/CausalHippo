package causalrag

import causalrag.causalgraph.builder.CausalGraphBuilder
import causalrag.causalgraph.builder.CausalTriple
import causalrag.causalgraph.graph.DirectedGraph
import causalrag.causalgraph.retriever.CausalPathRetriever
import causalrag.generator.llm.LLMInterface
import causalrag.generator.promptbuilder.buildPrompt
import causalrag.reranker.CausalPathReranker
import causalrag.retriever.Bm25Retriever
import causalrag.retriever.HybridRetriever
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
import shared.chunking.hardTruncateStringsByTokenBudget
import shared.config.CaualRagConfig
import shared.config.CommonRagConfigLoader
import shared.rag.CommonRag
import shared.rag.CommonVectorStorage
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
            semanticRetriever = vectorRetriever,
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
        }
    }

    override fun upsert(data: String) = runBlocking { aupsert(data) }

    override fun upsert(data: Collection<String>) = runBlocking { aupsert(data) }

    override suspend fun aupsert(data: String) = aupsert(listOf(data))

    override suspend fun aupsert(data: Collection<String>) {
        val documents = normalizeDocuments(data)
        if (documents.isEmpty()) {
            logger.warn { "No valid documents provided for CausalRAG upsert." }
            return
        }
        indexDocuments(documents)
        hybridRetriever.clearCache()
    }

    fun reindex(data: String) = runBlocking { areindex(data) }

    fun reindex(data: Collection<String>) = runBlocking { areindex(data) }

    suspend fun areindex(data: String) = areindex(listOf(data))

    suspend fun areindex(data: Collection<String>) {
        val documents = normalizeDocuments(data)
        hybridRetriever.clearCache()
        graphBuilder.clear()
        vectorRetriever.clear()
        bm25Retriever.clear()
        if (documents.isEmpty()) {
            if (unifiedSpiEnabled) clearUnifiedSpiState()
            logger.warn { "No valid documents provided for CausalRAG reindex." }
            return
        }
        indexDocuments(documents)
        hybridRetriever.clearCache()
    }

    override fun drop() = runBlocking { adrop() }

    override suspend fun adrop() {
        hybridRetriever.clearCache()
        graphBuilder.clear()
        vectorRetriever.clear()
        bm25Retriever.clear()
        if (unifiedSpiEnabled) clearUnifiedSpiState()
    }

    override fun inspectGraph(): Map<String, Any?> = runBlocking { ainspectGraph() }

    override suspend fun ainspectGraph(): Map<String, Any?> =
        if (unifiedSpiEnabled) {
            inspectGraphFromUnifiedSpi()
        } else {
            inspectGraphFromLocal()
        }

    override fun saveGraph(path: String) = runBlocking { asaveGraph(path) }

    override suspend fun asaveGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        check(save(path)) { "Failed to save CausalRAG graph data to '$path'." }
    }

    override fun loadGraph(path: String) = runBlocking { aloadGraph(path) }

    override suspend fun aloadGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        check(load(path)) { "Failed to load CausalRAG graph data from '$path'." }
        hybridRetriever.clearCache()
    }

    override fun query(
        query: String,
        param: QueryParam,
    ): CausalRagRunResult = runBlocking { aquery(query, param) }

    override suspend fun aquery(
        query: String,
        param: QueryParam,
    ): CausalRagRunResult {
        val topK = param.topK.coerceAtLeast(1)
        val maxPaths = param.maxPaths.coerceAtLeast(1)

        if (param.onlyNeedContext || param.onlyNeedCausalPaths) {
            val context =
                if (param.onlyNeedContext) {
                    retrieveContext(query, topK = topK)
                } else {
                    emptyList()
                }
            val causalPaths =
                if (param.onlyNeedCausalPaths) {
                    retrieveCausalPaths(query, maxPaths = maxPaths)
                } else {
                    emptyList()
                }
            return CausalRagRunResult(answer = "", context = context, causalPaths = causalPaths)
        }

        val result = runWithContext(query, topK = topK)
        return if (maxPaths == 3) {
            result
        } else {
            result.copy(causalPaths = retrieveCausalPaths(query, maxPaths = maxPaths))
        }
    }

    override suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>> = vectorStorageBackend.query(query, topK)

    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {
        vectorStorageBackend.upsert(data)
        bm25Retriever.clear()
        val passages = vectorRetriever.getPassages()
        if (passages.isNotEmpty()) {
            bm25Retriever.indexDocuments(passages)
        }
        if (unifiedSpiEnabled) syncUnifiedSpiFromLocal()
        hybridRetriever.clearCache()
    }

    override suspend fun deleteEntity(entityName: String) {
        vectorStorageBackend.deleteEntity(entityName)
        bm25Retriever.clear()
        val passages = vectorRetriever.getPassages()
        if (passages.isNotEmpty()) {
            bm25Retriever.indexDocuments(passages)
        }
        if (unifiedSpiEnabled) syncUnifiedSpiFromLocal()
        hybridRetriever.clearCache()
    }

    override suspend fun deleteEntityRelation(entityName: String) {
        vectorStorageBackend.deleteEntityRelation(entityName)
        bm25Retriever.clear()
        val passages = vectorRetriever.getPassages()
        if (passages.isNotEmpty()) {
            bm25Retriever.indexDocuments(passages)
        }
        if (unifiedSpiEnabled) syncUnifiedSpiFromLocal()
        hybridRetriever.clearCache()
    }

    fun run(
        query: String,
        topK: Int = 5,
    ): String = runWithContext(query, topK = topK).answer

    fun retrieveContext(
        query: String,
        topK: Int = 5,
    ): List<String> =
        if (unifiedSpiEnabled) {
            retrieveContextFromUnifiedSpi(query, topK)
        } else {
            hybridRetriever.retrieve(query, topK = topK)
        }

    fun retrieveCausalPaths(
        query: String,
        maxPaths: Int = 3,
    ): List<List<String>> =
        if (unifiedSpiEnabled) {
            retrieveCausalPathsFromUnifiedSpi(query, maxPaths)
        } else {
            graphRetriever.retrievePaths(query, maxPaths = maxPaths)
        }

    private fun runWithContext(
        query: String,
        topK: Int = 5,
    ): CausalRagRunResult {
        val topPassages =
            if (unifiedSpiEnabled) {
                retrieveContextFromUnifiedSpi(query, topK = topK)
            } else {
                val candidates = hybridRetriever.retrieve(query, topK = topK)
                val reranked = reranker.rerank(query, candidates)
                reranked.map { it.first }.take(topK)
            }
        val rerankedPassages =
            hardTruncateStringsByTokenBudget(
                items = topPassages,
                maxTokenSize = effectivePromptContextTokenBudget,
                model = DEFAULT_TIKTOKEN_MODEL,
                includePartialLastItem = false,
            )

        val causalNodes = retrievePathNodes(query)
        val causalPaths = retrieveCausalPaths(query, maxPaths = 3)
        val prompt =
            buildPrompt(
                query,
                rerankedPassages,
                causalPaths = causalPaths,
                causalNodes = causalNodes,
                templateStyle = effectiveTemplateStyle,
                llmInterface = llm,
            )
        val answer = llm.generate(prompt, jsonMode = requiresJsonResponseFormat(effectiveTemplateStyle))
        return CausalRagRunResult(answer, rerankedPassages, causalPaths)
    }

    private fun retrievePathNodes(query: String): List<String> =
        if (unifiedSpiEnabled) {
            retrievePathNodesFromUnifiedSpi(query)
        } else {
            graphRetriever.retrievePathNodes(query)
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

    private fun indexDocuments(documents: List<String>) {
        val prepared = chunkDocumentsForIngest(documents)
        graphBuilder.indexDocuments(prepared)
        vectorRetriever.indexCorpus(prepared)
        bm25Retriever.indexDocuments(prepared)
        if (unifiedSpiEnabled) syncUnifiedSpiFromLocal()
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
            syncUnifiedSpiFromLocal()
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
            if (vectorsLoaded) {
                val passages = vectorRetriever.getPassages()
                if (passages.isNotEmpty()) {
                    bm25Retriever.indexDocuments(passages)
                } else {
                    logger.warn { "Vector index loaded but no passages found to rebuild BM25 index." }
                }
            }
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

        vectorRetriever.indexCorpus(
            texts = passages,
            metadata = metadata,
            ids = ids,
        )
        bm25Retriever.clear()
        if (passages.isNotEmpty()) {
            bm25Retriever.indexDocuments(passages)
        }
        hybridRetriever.clearCache()
    }

    private fun retrieveContextFromUnifiedSpi(
        query: String,
        topK: Int,
    ): List<String> {
        ensureUnifiedSpiIndexInitialized()
        val session = unifiedSpiSession ?: return emptyList()
        val queryEmbedding = vectorRetriever.encodeForQuery(query) ?: return emptyList()
        return try {
            session
                .vector(unifiedSpiVectorNamespace, dimensions = queryEmbedding.size, metric = "cosine")
                .query(queryEmbedding.toList(), topK = topK.coerceAtLeast(1))
                .map { record ->
                    record.metadata["passage"]?.toString()
                        ?: record.metadata["content"]?.toString()
                        ?: record.id
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
        ensureUnifiedSpiIndexInitialized()
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
        ensureUnifiedSpiIndexInitialized()
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
