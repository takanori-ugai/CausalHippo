package causalhippo

import causalrag.CausalRagRunResult
import causalrag.retriever.HippoRagSemanticMode
import hipporag.config.BaseConfig
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.runBlocking
import shared.config.CommonRagConfigLoader
import shared.rag.CommonRag
import shared.rag.IndexingReport
import tools.jackson.core.type.TypeReference
import tools.jackson.module.kotlin.jacksonObjectMapper
import java.nio.file.Files
import java.nio.file.Path

data class CausalHippoQueryParam(
    val topK: Int = 5,
    val maxPaths: Int = 3,
    val onlyNeedContext: Boolean = false,
    val onlyNeedCausalPaths: Boolean = false,
)

class CausalHippoRAG(
    modelName: String = "gpt-4o-mini",
    embeddingModel: String = "text-embedding-3-small",
    configPath: String? = null,
    templateStyle: String? = null,
    hippoConfig: BaseConfig? = null,
    hippoSemanticMode: HippoRagSemanticMode = HippoRagSemanticMode.GRAPH,
    semanticWeight: Double = 0.4,
    causalWeight: Double = 0.5,
    bm25Weight: Double = 0.1,
    minCausalMatches: Int = 0,
    dynamicWeightingEnabled: Boolean = true,
    twoPassAdaptiveEnabled: Boolean = true,
    confidenceBasedSwitchEnabled: Boolean = true,
) : CommonRag<CausalHippoQueryParam, CausalRagRunResult> {
    private data class PipelineArgs(
        val modelName: String,
        val embeddingModel: String,
        val configPath: String?,
        val templateStyle: String?,
        val hippoConfig: BaseConfig?,
        val hippoSemanticMode: HippoRagSemanticMode,
        val semanticWeight: Double,
        val causalWeight: Double,
        val bm25Weight: Double,
        val minCausalMatches: Int,
        val dynamicWeightingEnabled: Boolean,
        val twoPassAdaptiveEnabled: Boolean,
        val confidenceBasedSwitchEnabled: Boolean,
    )

    private val logger = KotlinLogging.logger("CausalHippoRAG")
    private val objectMapper = jacksonObjectMapper()
    private val args =
        PipelineArgs(
            modelName = modelName,
            embeddingModel = embeddingModel,
            configPath = configPath,
            templateStyle = templateStyle,
            hippoConfig = hippoConfig?.copy(),
            hippoSemanticMode = hippoSemanticMode,
            semanticWeight = semanticWeight,
            causalWeight = causalWeight,
            bm25Weight = bm25Weight,
            minCausalMatches = minCausalMatches,
            dynamicWeightingEnabled = dynamicWeightingEnabled,
            twoPassAdaptiveEnabled = twoPassAdaptiveEnabled,
            confidenceBasedSwitchEnabled = confidenceBasedSwitchEnabled,
        )
    private val pipeline: CausalHippoPipeline = createPipeline()

    companion object {
        fun fromCommonConfig(configPath: String): CausalHippoRAG {
            val commonConfig = CommonRagConfigLoader.load(configPath)
            val causalConfig = commonConfig.toCaualRagConfig()
            val hippoConfig = commonConfig.toHippoBaseConfig()
            return CausalHippoRAG(
                modelName = causalConfig.modelName ?: "gpt-4o-mini",
                embeddingModel = causalConfig.embeddingModel ?: "text-embedding-3-small",
                configPath = configPath,
                templateStyle = causalConfig.templateStyle,
                hippoConfig = hippoConfig,
                minCausalMatches = causalConfig.minCausalMatches ?: 0,
            )
        }
    }

    val lastIndexingReport: IndexingReport
        get() = pipeline.lastIndexingReport

    override fun upsert(data: String) = runBlocking { aupsert(data) }

    override fun upsert(data: Collection<String>) = runBlocking { aupsert(data) }

    override suspend fun aupsert(data: String) = aupsert(listOf(data))

    override suspend fun aupsert(data: Collection<String>) {
        val documents = normalizeDocuments(data)
        if (documents.isEmpty()) {
            logger.warn { "No valid documents provided for CausalHippoRAG upsert." }
            return
        }
        pipeline.index(documents)
    }

    override fun drop() = runBlocking { adrop() }

    override suspend fun adrop() = pipeline.reindex(emptyList())

    override fun saveGraph(path: String) = runBlocking { asaveGraph(path) }

    override suspend fun asaveGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        pipeline.saveSnapshot(Path.of(path))
    }

    override fun loadGraph(path: String) = runBlocking { aloadGraph(path) }

    override suspend fun aloadGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        pipeline.loadSnapshot(Path.of(path))
    }

    override fun inspectGraph(): Map<String, Any?> = runBlocking { ainspectGraph() }

    override suspend fun ainspectGraph(): Map<String, Any?> = synchronized(pipeline) { inspect() }

    private fun inspect(): Map<String, Any?> {
        val causalSnapshot = inspectCausalGraph()
        val hippoSnapshot = inspectHippoGraph()
        val causalMeta = causalSnapshot["metadata"] as? Map<*, *> ?: emptyMap<Any?, Any?>()
        val hippoMeta = hippoSnapshot["metadata"] as? Map<*, *> ?: emptyMap<Any?, Any?>()

        return mapOf(
            "causal_graph" to causalSnapshot,
            "hippo_graph" to hippoSnapshot,
            "metadata" to
                mapOf(
                    "causalNodeCount" to (causalMeta["nodeCount"] as? Int ?: 0),
                    "causalEdgeCount" to (causalMeta["edgeCount"] as? Int ?: 0),
                    "hippoNodeCount" to (hippoMeta["nodeCount"] as? Int ?: 0),
                    "hippoEdgeCount" to (hippoMeta["edgeCount"] as? Int ?: 0),
                ),
        )
    }

    override fun query(
        query: String,
        param: CausalHippoQueryParam,
    ): CausalRagRunResult = runBlocking { aquery(query, param) }

    override suspend fun aquery(
        query: String,
        param: CausalHippoQueryParam,
    ): CausalRagRunResult {
        val topK = param.topK.coerceAtLeast(1)
        val maxPaths = param.maxPaths.coerceAtLeast(0)

        if (param.onlyNeedContext) {
            val evidence = pipeline.preparePrompt(query, topK, maxPaths, summarize = false)
            return CausalRagRunResult("", evidence.passages, if (param.onlyNeedCausalPaths) evidence.paths else emptyList())
        }
        if (param.onlyNeedCausalPaths) {
            return CausalRagRunResult("", emptyList(), pipeline.retrieveCausalPaths(query, maxPaths))
        }
        return pipeline.runWithContext(query, topK, maxPaths)
    }

    private fun createPipeline(): CausalHippoPipeline =
        CausalHippoPipeline(
            modelName = args.modelName,
            embeddingModel = args.embeddingModel,
            configPath = args.configPath,
            templateStyle = args.templateStyle,
            hippoConfig = args.hippoConfig?.copy(),
            hippoSemanticMode = args.hippoSemanticMode,
            semanticWeight = args.semanticWeight,
            causalWeight = args.causalWeight,
            bm25Weight = args.bm25Weight,
            minCausalMatches = args.minCausalMatches,
            dynamicWeightingEnabled = args.dynamicWeightingEnabled,
            twoPassAdaptiveEnabled = args.twoPassAdaptiveEnabled,
            confidenceBasedSwitchEnabled = args.confidenceBasedSwitchEnabled,
        )

    private fun normalizeDocuments(data: Collection<String>): List<String> =
        data
            .map { it.trim() }
            .filter { it.isNotBlank() }

    private fun hippoWorkingDir(config: BaseConfig): Path {
        val llmLabel = config.llmName.replace("/", "_")
        val embeddingLabel = config.embeddingModelName.replace("/", "_")
        return Path.of(config.saveDir, "${llmLabel}_$embeddingLabel").toAbsolutePath().normalize()
    }

    private fun inspectCausalGraph(): Map<String, Any?> {
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

    private fun inspectHippoGraph(): Map<String, Any?> {
        val graphPath = hippoWorkingDir(pipeline.hippoRag.globalConfig).resolve("graph.json")
        if (!Files.exists(graphPath) || !Files.isRegularFile(graphPath)) {
            return mapOf(
                "nodes" to emptyList<Map<String, Any?>>(),
                "edges" to emptyList<Map<String, Any?>>(),
                "metadata" to
                    mapOf(
                        "nodeCount" to 0,
                        "edgeCount" to 0,
                        "directed" to pipeline.hippoRag.globalConfig.isDirectedGraph,
                    ),
            )
        }
        val payloadType = object : TypeReference<Map<String, Any?>>() {}
        val payload = objectMapper.readValue(graphPath.toFile(), payloadType)
        val nodes = payload["vertices"] as? List<*> ?: emptyList<Any?>()
        val edges = payload["edges"] as? List<*> ?: emptyList<Any?>()
        return mapOf(
            "nodes" to nodes,
            "edges" to edges,
            "metadata" to
                mapOf(
                    "nodeCount" to nodes.size,
                    "edgeCount" to edges.size,
                    "directed" to (payload["directed"] as? Boolean ?: false),
                ),
        )
    }
}
