package causalrag

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.runBlocking
import shared.rag.CommonRag
import shared.rag.CommonVectorStorage

data class QueryParam(
    val topK: Int = 5,
    val maxPaths: Int = 3,
    val onlyNeedContext: Boolean = false,
    val onlyNeedCausalPaths: Boolean = false,
)

class CausalRAG(
    private val pipeline: CausalRAGPipeline,
) : CommonRag<QueryParam, PipelineRunResult>,
    CommonVectorStorage<Any?> {
    constructor(
        modelName: String = "gpt-4",
        embeddingModel: String = "all-MiniLM-L6-v2",
        graphPath: String? = null,
        indexPath: String? = null,
        configPath: String? = null,
        templateStyle: String? = null,
        dynamicWeightingEnabled: Boolean = false,
        twoPassAdaptiveEnabled: Boolean = false,
        confidenceBasedSwitchEnabled: Boolean = false,
    ) : this(
        CausalRAGPipeline(
            modelName = modelName,
            embeddingModel = embeddingModel,
            graphPath = graphPath,
            indexPath = indexPath,
            configPath = configPath,
            templateStyle = templateStyle,
            dynamicWeightingEnabled = dynamicWeightingEnabled,
            twoPassAdaptiveEnabled = twoPassAdaptiveEnabled,
            confidenceBasedSwitchEnabled = confidenceBasedSwitchEnabled,
        ),
    )

    private val logger = KotlinLogging.logger("CausalRAG")
    private val vectorStorageBackend = CausalRAGVectorStorage(pipeline.vectorRetriever)

    override fun upsert(data: String) = runBlocking { aupsert(data) }

    override fun upsert(data: Collection<String>) = runBlocking { aupsert(data) }

    override suspend fun aupsert(data: String) = aupsert(listOf(data))

    override suspend fun aupsert(data: Collection<String>) {
        val documents = normalizeDocuments(data)
        if (documents.isEmpty()) {
            logger.warn { "No valid documents provided for CausalRAG upsert." }
            return
        }
        pipeline.index(documents)
        pipeline.hybridRetriever.clearCache()
    }

    override fun drop() = runBlocking { adrop() }

    override suspend fun adrop() {
        pipeline.hybridRetriever.clearCache()
        pipeline.graphBuilder.clear()
        pipeline.vectorRetriever.clear()
        pipeline.bm25Retriever.clear()
    }

    override fun inspectGraph(): Map<String, Any?> = runBlocking { ainspectGraph() }

    override suspend fun ainspectGraph(): Map<String, Any?> {
        val graph = pipeline.graphBuilder.getGraph()
        val nodeText = pipeline.graphBuilder.nodeText
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
        check(pipeline.save(path)) { "Failed to save CausalRAG graph data to '$path'." }
    }

    override fun loadGraph(path: String) = runBlocking { aloadGraph(path) }

    override suspend fun aloadGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        check(pipeline.load(path)) { "Failed to load CausalRAG graph data from '$path'." }
        pipeline.hybridRetriever.clearCache()
    }

    override fun query(
        query: String,
        param: QueryParam,
    ): PipelineRunResult = runBlocking { aquery(query, param) }

    override suspend fun aquery(
        query: String,
        param: QueryParam,
    ): PipelineRunResult {
        val topK = param.topK.coerceAtLeast(1)
        val maxPaths = param.maxPaths.coerceAtLeast(1)

        if (param.onlyNeedContext || param.onlyNeedCausalPaths) {
            val context =
                if (param.onlyNeedContext) {
                    pipeline.retrieveContext(query, topK = topK)
                } else {
                    emptyList()
                }
            val causalPaths =
                if (param.onlyNeedCausalPaths) {
                    pipeline.retrieveCausalPaths(query, maxPaths = maxPaths)
                } else {
                    emptyList()
                }
            return PipelineRunResult(answer = "", context = context, causalPaths = causalPaths)
        }

        val result = pipeline.runWithContext(query, topK = topK)
        return if (maxPaths == 3) {
            result
        } else {
            result.copy(causalPaths = pipeline.retrieveCausalPaths(query, maxPaths = maxPaths))
        }
    }

    override suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>> = vectorStorageBackend.query(query, topK)

    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {
        vectorStorageBackend.upsert(data)
        pipeline.bm25Retriever.clear()
        val passages = pipeline.vectorRetriever.getPassages()
        if (passages.isNotEmpty()) {
            pipeline.bm25Retriever.indexDocuments(passages)
        }
        pipeline.hybridRetriever.clearCache()
    }

    override suspend fun deleteEntity(entityName: String) {
        vectorStorageBackend.deleteEntity(entityName)
        pipeline.bm25Retriever.clear()
        val passages = pipeline.vectorRetriever.getPassages()
        if (passages.isNotEmpty()) {
            pipeline.bm25Retriever.indexDocuments(passages)
        }
        pipeline.hybridRetriever.clearCache()
    }

    override suspend fun deleteEntityRelation(entityName: String) {
        vectorStorageBackend.deleteEntityRelation(entityName)
        pipeline.bm25Retriever.clear()
        val passages = pipeline.vectorRetriever.getPassages()
        if (passages.isNotEmpty()) {
            pipeline.bm25Retriever.indexDocuments(passages)
        }
        pipeline.hybridRetriever.clearCache()
    }

    private fun normalizeDocuments(data: Collection<String>): List<String> =
        data
            .map { it.trim() }
            .filter { it.isNotBlank() }
}
