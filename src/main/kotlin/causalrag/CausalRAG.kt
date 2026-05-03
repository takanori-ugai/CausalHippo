package causalrag

import causalrag.causalgraph.builder.CausalGraphBuilder
import causalrag.causalgraph.retriever.CausalPathRetriever
import causalrag.generator.llm.LLMInterface
import causalrag.generator.promptbuilder.buildPrompt
import causalrag.reranker.CausalPathReranker
import causalrag.retriever.Bm25Retriever
import causalrag.retriever.HybridRetriever
import causalrag.retriever.VectorStoreRetriever
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
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path

private val logger = KotlinLogging.logger("CausalRAG")
private val caualRagConfigJson = Json { ignoreUnknownKeys = true }

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
    dynamicWeightingEnabled: Boolean = false,
    twoPassAdaptiveEnabled: Boolean = false,
    confidenceBasedSwitchEnabled: Boolean = false,
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
    private val effectiveEmbeddingApiKey = config?.embeddingApiKey ?: System.getenv("OPENAI_API_KEY")
    private val effectiveTemplateStyle = templateStyle ?: config?.templateStyle ?: "detailed"
    private val effectiveMinCausalMatches = config?.minCausalMatches ?: 0
    private val effectiveIngestChunkTokenSize = config?.ingestChunkTokenSize ?: DEFAULT_INGEST_CHUNK_TOKEN_SIZE
    private val effectiveIngestChunkOverlapTokenSize =
        config?.ingestChunkOverlapTokenSize ?: DEFAULT_INGEST_CHUNK_OVERLAP_TOKEN_SIZE
    private val effectivePromptContextTokenBudget =
        config?.promptContextTokenBudget ?: DEFAULT_PROMPT_CONTEXT_TOKEN_BUDGET

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
    }

    override fun inspectGraph(): Map<String, Any?> = runBlocking { ainspectGraph() }

    override suspend fun ainspectGraph(): Map<String, Any?> {
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
        hybridRetriever.clearCache()
    }

    override suspend fun deleteEntity(entityName: String) {
        vectorStorageBackend.deleteEntity(entityName)
        bm25Retriever.clear()
        val passages = vectorRetriever.getPassages()
        if (passages.isNotEmpty()) {
            bm25Retriever.indexDocuments(passages)
        }
        hybridRetriever.clearCache()
    }

    override suspend fun deleteEntityRelation(entityName: String) {
        vectorStorageBackend.deleteEntityRelation(entityName)
        bm25Retriever.clear()
        val passages = vectorRetriever.getPassages()
        if (passages.isNotEmpty()) {
            bm25Retriever.indexDocuments(passages)
        }
        hybridRetriever.clearCache()
    }

    fun run(
        query: String,
        topK: Int = 5,
    ): String = runWithContext(query, topK = topK).answer

    fun retrieveContext(
        query: String,
        topK: Int = 5,
    ): List<String> = hybridRetriever.retrieve(query, topK = topK)

    fun retrieveCausalPaths(
        query: String,
        maxPaths: Int = 3,
    ): List<List<String>> = graphRetriever.retrievePaths(query, maxPaths = maxPaths)

    private fun runWithContext(
        query: String,
        topK: Int = 5,
    ): CausalRagRunResult {
        val candidates = hybridRetriever.retrieve(query, topK = topK)
        val reranked = reranker.rerank(query, candidates)
        val topPassages = reranked.map { it.first }.take(topK)
        val rerankedPassages =
            hardTruncateStringsByTokenBudget(
                items = topPassages,
                maxTokenSize = effectivePromptContextTokenBudget,
                model = DEFAULT_TIKTOKEN_MODEL,
                includePartialLastItem = false,
            )

        val causalNodes = graphRetriever.retrievePathNodes(query)
        val causalPaths = graphRetriever.retrievePaths(query, maxPaths = 3)
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

    private fun load(dir: String): Boolean =
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

    private fun normalizeDocuments(data: Collection<String>): List<String> =
        data
            .map { it.trim() }
            .filter { it.isNotBlank() }

    private fun requiresJsonResponseFormat(style: String): Boolean = style.equals("experiments", ignoreCase = true)
}
