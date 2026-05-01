package causalrag

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import hipporag.config.BaseConfig
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import shared.config.CommonRagConfigLoader
import shared.rag.CommonRag
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.Comparator

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
    hippoSemanticMode: causalrag.retriever.HippoRagSemanticMode = causalrag.retriever.HippoRagSemanticMode.GRAPH,
    semanticWeight: Double = 0.4,
    causalWeight: Double = 0.5,
    bm25Weight: Double = 0.1,
    minCausalMatches: Int = 0,
    dynamicWeightingEnabled: Boolean = false,
    twoPassAdaptiveEnabled: Boolean = false,
    confidenceBasedSwitchEnabled: Boolean = false,
) : CommonRag<CausalHippoQueryParam, PipelineRunResult> {
    private data class PipelineArgs(
        val modelName: String,
        val embeddingModel: String,
        val configPath: String?,
        val templateStyle: String?,
        val hippoConfig: BaseConfig?,
        val hippoSemanticMode: causalrag.retriever.HippoRagSemanticMode,
        val semanticWeight: Double,
        val causalWeight: Double,
        val bm25Weight: Double,
        val minCausalMatches: Int,
        val dynamicWeightingEnabled: Boolean,
        val twoPassAdaptiveEnabled: Boolean,
        val confidenceBasedSwitchEnabled: Boolean,
    )

    private val logger = KotlinLogging.logger("CausalHippoRAG")
    private val json = Json { ignoreUnknownKeys = true }
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
    private var pipeline: HippoCausalRAGPipeline = createPipeline()

    companion object {
        fun fromCommonConfig(configPath: String): CausalHippoRAG {
            val commonConfig = CommonRagConfigLoader.load(configPath)
            val causalConfig = commonConfig.toPipelineConfig()
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
        pipeline.hybridRetriever.clearCache()
    }

    override fun drop() = runBlocking { adrop() }

    override suspend fun adrop() {
        pipeline.hybridRetriever.clearCache()
        pipeline.graphBuilder.clear()
        pipeline.bm25Retriever.clear()
        val hippoConfig = pipeline.hippoRag.globalConfig
        deleteDirectory(hippoWorkingDir(hippoConfig))
        Files.deleteIfExists(hippoOpenieFile(hippoConfig))
        recreatePipeline()
    }

    override fun saveGraph(path: String) = runBlocking { asaveGraph(path) }

    override suspend fun asaveGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        val target = Path.of(path).toAbsolutePath().normalize()
        deleteDirectory(target)
        Files.createDirectories(target)

        val hippoConfig = pipeline.hippoRag.globalConfig
        val hippoSourceDir = hippoWorkingDir(hippoConfig)
        require(Files.exists(hippoSourceDir) && Files.isDirectory(hippoSourceDir)) {
            "Hippo working directory does not exist: $hippoSourceDir"
        }
        copyDirectory(hippoSourceDir, target.resolve("hippo_working_dir"))

        val openieSource = hippoOpenieFile(hippoConfig)
        if (Files.exists(openieSource) && Files.isRegularFile(openieSource)) {
            Files.copy(
                openieSource,
                target.resolve(openieSource.fileName.toString()),
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.COPY_ATTRIBUTES,
            )
        }

        val causalGraphPath = target.resolve("causal_graph.json")
        check(pipeline.graphBuilder.save(causalGraphPath.toString())) {
            "Failed to save causal graph to '$causalGraphPath'."
        }
    }

    override fun loadGraph(path: String) = runBlocking { aloadGraph(path) }

    override suspend fun aloadGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        val sourceRoot = Path.of(path).toAbsolutePath().normalize()
        require(Files.exists(sourceRoot) && Files.isDirectory(sourceRoot)) {
            "CausalHippo snapshot directory not found: $sourceRoot"
        }

        val sourceHippoDir =
            when {
                Files.exists(sourceRoot.resolve("hippo_working_dir")) &&
                    Files.isDirectory(sourceRoot.resolve("hippo_working_dir")) -> sourceRoot.resolve("hippo_working_dir")

                Files.exists(sourceRoot.resolve("graph.json")) -> sourceRoot

                else -> error("Invalid CausalHippo snapshot: expected 'hippo_working_dir/' or Hippo graph.json in $sourceRoot")
            }

        val sourceCausalGraph = sourceRoot.resolve("causal_graph.json")
        require(Files.exists(sourceCausalGraph) && Files.isRegularFile(sourceCausalGraph)) {
            "Causal graph snapshot not found: $sourceCausalGraph"
        }

        val currentHippoConfig = pipeline.hippoRag.globalConfig
        val targetHippoDir = hippoWorkingDir(currentHippoConfig)
        deleteDirectory(targetHippoDir)
        copyDirectory(sourceHippoDir, targetHippoDir)

        val targetOpenie = hippoOpenieFile(currentHippoConfig)
        Files.createDirectories(targetOpenie.parent)
        Files.deleteIfExists(targetOpenie)
        val sourceOpenie = sourceRoot.resolve(targetOpenie.fileName.toString())
        if (Files.exists(sourceOpenie) && Files.isRegularFile(sourceOpenie)) {
            Files.copy(
                sourceOpenie,
                targetOpenie,
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.COPY_ATTRIBUTES,
            )
        }

        recreatePipeline()
        check(pipeline.graphBuilder.load(sourceCausalGraph.toString())) {
            "Failed to load causal graph from '$sourceCausalGraph'."
        }

        val bm25Docs = readHippoChunkTexts(pipeline.hippoRag.globalConfig)
        pipeline.bm25Retriever.clear()
        if (bm25Docs.isNotEmpty()) {
            pipeline.bm25Retriever.indexDocuments(bm25Docs)
        } else {
            logger.warn { "No Hippo chunk texts found after load; BM25 retriever remains empty." }
        }
        pipeline.hybridRetriever.clearCache()
    }

    override fun inspectGraph(): Map<String, Any?> = runBlocking { ainspectGraph() }

    override suspend fun ainspectGraph(): Map<String, Any?> {
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
    ): PipelineRunResult = runBlocking { aquery(query, param) }

    override suspend fun aquery(
        query: String,
        param: CausalHippoQueryParam,
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

    private fun createPipeline(): HippoCausalRAGPipeline =
        HippoCausalRAGPipeline(
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

    private fun recreatePipeline() {
        pipeline = createPipeline()
    }

    private fun normalizeDocuments(data: Collection<String>): List<String> =
        data
            .map { it.trim() }
            .filter { it.isNotBlank() }

    private fun hippoWorkingDir(config: BaseConfig): Path {
        val llmLabel = config.llmName.replace("/", "_")
        val embeddingLabel = config.embeddingModelName.replace("/", "_")
        return Path.of(config.saveDir, "${llmLabel}_$embeddingLabel").toAbsolutePath().normalize()
    }

    private fun hippoOpenieFile(config: BaseConfig): Path {
        val llmLabel = config.llmName.replace("/", "_")
        return Path.of(config.saveDir, "openie_results_ner_$llmLabel.json").toAbsolutePath().normalize()
    }

    private fun readHippoChunkTexts(config: BaseConfig): List<String> {
        val chunkStore = hippoWorkingDir(config).resolve("chunk_embeddings").resolve("vdb_chunk.json")
        if (!Files.exists(chunkStore) || !Files.isRegularFile(chunkStore)) return emptyList()
        return runCatching {
            val root = json.parseToJsonElement(Files.readString(chunkStore)) as JsonObject
            val rawTexts = root["texts"]
            if (rawTexts == null) {
                emptyList<String>()
            } else {
                rawTexts
                    .jsonArray
                    .mapNotNull { (it as? JsonPrimitive)?.contentOrNull?.trim() }
                    .filter { it.isNotBlank() }
            }
        }.getOrElse { ex ->
            logger.warn(ex) { "Failed to parse Hippo chunk store from '$chunkStore'." }
            emptyList()
        }
    }

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
