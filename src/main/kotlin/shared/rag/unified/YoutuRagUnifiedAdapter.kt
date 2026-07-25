package shared.rag.unified

import com.youtu.graphrag.server.api.GraphConstructionService
import com.youtu.graphrag.server.api.QuestionAnsweringService
import com.youtu.graphrag.server.api.contracts.QuestionResponse
import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.retriever.IrcotPromptSource
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import shared.rag.spi.persistence.GraphEdgeRecord
import shared.rag.spi.persistence.GraphNodeRecord
import shared.rag.spi.persistence.GraphSnapshot
import shared.rag.spi.persistence.KvSnapshot
import shared.rag.spi.persistence.PersistenceSession
import tools.jackson.databind.ObjectMapper
import tools.jackson.module.kotlin.jacksonObjectMapper
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.time.Instant
import java.util.concurrent.ConcurrentHashMap
import kotlin.io.path.createDirectories
import kotlin.io.path.deleteIfExists
import kotlin.io.path.exists
import kotlin.io.path.isDirectory

/**
 * Unified adapter for youtuRAG.
 */
class YoutuRagUnifiedAdapter(
    private val config: ConfigManager,
    private val datasetName: String = "demo",
    private val rootDir: Path = Path.of("."),
    private val ircotPromptSource: IrcotPromptSource = IrcotPromptSource.MAIN,
    private val persistenceSession: PersistenceSession? = null,
    private val useUnifiedSpiForRetrievalAndIndex: Boolean = false,
    private val persistenceNamespacePrefix: String = "youturag",
) : UnifiedRag {
    val capabilities: RagCapabilities = CAPABILITIES
    private val mapper = jacksonObjectMapper()
    private val graphService = GraphConstructionService(config = config, rootDir = rootDir)
    private val qaService =
        QuestionAnsweringService(
            config = config,
            rootDir = rootDir,
            ircotPromptSource = ircotPromptSource,
        )
    private val unifiedSpiEnabled = useUnifiedSpiForRetrievalAndIndex && persistenceSession != null
    private val hydratedDatasets = ConcurrentHashMap.newKeySet<String>()

    override fun upsert(data: String) = upsert(listOf(data))

    override fun upsert(data: Collection<String>) {
        val selectedDataset = datasetName
        val documents = data.mapIndexedNotNull { index, text -> text.takeIf { it.isNotBlank() }?.let { index to it } }
        if (documents.isEmpty()) return
        val corpusFile = corpusPath(selectedDataset)
        corpusFile.parent?.createDirectories()

        val payload =
            documents.map { (index, text) ->
                mapOf(
                    "title" to "document_$index",
                    "text" to text,
                )
            }
        mapper.writerWithDefaultPrettyPrinter().writeValue(corpusFile.toFile(), payload)
        graphService.constructGraph(selectedDataset)
        if (unifiedSpiEnabled) syncUnifiedSpiFromLocal(selectedDataset)
    }

    override suspend fun aupsert(data: String) = upsert(data)

    override suspend fun aupsert(data: Collection<String>) = upsert(data)

    override fun drop() {
        val selectedDataset = datasetName
        graphService.clearCacheFiles(selectedDataset)
        corpusPath(selectedDataset).deleteIfExists()
        if (unifiedSpiEnabled) clearUnifiedSpiState(selectedDataset)
    }

    override suspend fun adrop() = drop()

    override fun saveGraph(path: String) {
        val target = Path.of(path)
        val selectedDataset = datasetName
        ensureLocalStateHydratedFromSpi(selectedDataset)
        val graph = graphPath(selectedDataset)
        val chunks = chunksPath(selectedDataset)
        val corpus = corpusPath(selectedDataset)

        if (!target.exists()) target.createDirectories()
        require(target.isDirectory()) { "saveGraph path must be a directory: $target" }

        copyIfExists(graph, target.resolve(graph.fileName))
        copyIfExists(chunks, target.resolve(chunks.fileName))
        copyIfExists(corpus, target.resolve(corpus.fileName))
        if (unifiedSpiEnabled) {
            syncUnifiedSpiFromLocal(selectedDataset)
            runBlocking {
                val checkpointPath = "$path.unified_spi"
                persistenceSession?.checkpoint(checkpointPath)
                persistenceSession?.kv(metadataNamespace(selectedDataset))?.put("lastCheckpointPath", checkpointPath)
            }
        }
    }

    override suspend fun asaveGraph(path: String) = saveGraph(path)

    override fun loadGraph(path: String) {
        val source = Path.of(path)
        require(source.exists() && source.isDirectory()) { "loadGraph path must be an existing directory: $source" }

        val selectedDataset = datasetName
        copyIfExists(source.resolve("${selectedDataset}_new.json"), graphPath(selectedDataset))
        copyIfExists(source.resolve("$selectedDataset.txt"), chunksPath(selectedDataset))
        copyIfExists(source.resolve("corpus.json"), corpusPath(selectedDataset))
        if (unifiedSpiEnabled) {
            runBlocking {
                val checkpointPath = "$path.unified_spi"
                if (Files.exists(Path.of(checkpointPath).resolve("manifest.json"))) {
                    persistenceSession?.restore(checkpointPath)
                }
            }
            syncUnifiedSpiFromLocal(selectedDataset)
            ensureLocalStateHydratedFromSpi(selectedDataset, force = true)
        }
    }

    override suspend fun aloadGraph(path: String) = loadGraph(path)

    override fun inspectGraph(): Map<String, Any?> {
        val selectedDataset = datasetName
        ensureLocalStateHydratedFromSpi(selectedDataset)
        val normalized = normalizeInspectionFromVisualization(selectedDataset)
        if (unifiedSpiEnabled) syncUnifiedSpiFromLocal(selectedDataset)
        return normalized
    }

    override suspend fun ainspectGraph(): Map<String, Any?> = inspectGraph()

    override fun query(
        query: String,
        param: UnifiedQuery,
    ): UnifiedResponse = runBlocking { aquery(query, param) }

    override suspend fun aquery(
        query: String,
        param: UnifiedQuery,
    ): UnifiedResponse {
        if (!param.includeAnswer && !param.includeContext && !param.includeReferences && !param.includeFollowUps) {
            return UnifiedResponse(
                metadata =
                    buildMetadata(
                        ragId = RagId.YOUTU_RAG,
                        modeUsed = modeFor(param.mode),
                        unsupported = collectUnsupported(param, capabilities),
                        extra = mapOf("datasetName" to effectiveDatasetName(param)),
                    ),
            )
        }

        val effectiveMode = modeFor(param.mode)
        val selectedDataset = effectiveDatasetName(param)
        ensureLocalStateHydratedFromSpi(selectedDataset)
        applyQueryOverrides(effectiveMode, param.coercedTopK())

        val result = qaService.answerQuestion(datasetName = selectedDataset, question = query)
        if (unifiedSpiEnabled) syncUnifiedSpiFromLocal(selectedDataset)
        return result.toUnifiedResponse(param, effectiveMode, collectUnsupported(param, capabilities), selectedDataset)
    }

    private fun applyQueryOverrides(
        mode: String,
        topK: Int,
    ) {
        config.overrideConfig(
            mapOf(
                "triggers" to mapOf("mode" to mode),
                "construction" to mapOf("mode" to mode),
                "retrieval" to mapOf("topK" to topK, "topKFilter" to topK),
            ),
        )
    }

    private fun effectiveDatasetName(query: UnifiedQuery): String =
        query.stringExtra("datasetName") ?: query.stringExtra("dataset") ?: datasetName

    private fun QuestionResponse.toUnifiedResponse(
        query: UnifiedQuery,
        modeUsed: String,
        unsupported: List<String>,
        selectedDataset: String,
    ): UnifiedResponse {
        val context = if (query.includeContext) textContextItems(retrievedChunks) else emptyList()
        val references = if (query.includeReferences) textReferences(retrievedTriples) else emptyList()
        val followUps =
            if (query.includeFollowUps) {
                subQuestions
                    .mapNotNull { row -> row["sub-question"]?.trim() }
                    .filter { it.isNotBlank() }
            } else {
                emptyList()
            }

        return UnifiedResponse(
            answer = answer.takeIf { query.includeAnswer && it.isNotBlank() },
            context = context,
            references = references,
            followUpQueries = followUps,
            metadata =
                buildMetadata(
                    ragId = RagId.YOUTU_RAG,
                    modeUsed = modeUsed,
                    unsupported = unsupported,
                    extra =
                        mapOf(
                            "datasetName" to selectedDataset,
                            "subQuestionCount" to subQuestions.size,
                            "retrievedTriplesCount" to retrievedTriples.size,
                            "retrievedChunksCount" to retrievedChunks.size,
                            "reasoningStepCount" to reasoningSteps.size,
                        ),
                ),
            raw = this,
        )
    }

    private fun graphPath(selectedDataset: String): Path = resolvePath(config.output.graphsDir).resolve("${selectedDataset}_new.json")

    private fun chunksPath(selectedDataset: String): Path = resolvePath(config.output.chunksDir).resolve("$selectedDataset.txt")

    private fun corpusPath(selectedDataset: String): Path = rootDir.resolve("data/uploaded/$selectedDataset/corpus.json")

    private fun resolvePath(configuredPath: String): Path {
        val candidate = Path.of(configuredPath)
        return if (candidate.isAbsolute) candidate else rootDir.resolve(candidate)
    }

    private fun copyIfExists(
        source: Path,
        target: Path,
    ) {
        if (!source.exists()) return
        target.parent?.createDirectories()
        Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING)
    }

    private fun ensureLocalStateHydratedFromSpi(
        selectedDataset: String,
        force: Boolean = false,
    ) {
        if (!unifiedSpiEnabled) return
        if (!force && selectedDataset in hydratedDatasets) return
        val session = persistenceSession ?: return
        val files = session.kv(filesNamespace(selectedDataset)).snapshot().entries
        if (files.isEmpty()) return

        val graphPayload = files["graph_json"] as? String
        val chunksPayload = files["chunks_text"] as? String
        val corpusPayload = files["corpus_json"] as? String

        if (force || !graphPath(selectedDataset).exists()) {
            graphPayload?.let { writeText(graphPath(selectedDataset), it) }
        }
        if (force || !chunksPath(selectedDataset).exists()) {
            chunksPayload?.let { writeText(chunksPath(selectedDataset), it) }
        }
        if (force || !corpusPath(selectedDataset).exists()) {
            corpusPayload?.let { writeText(corpusPath(selectedDataset), it) }
        }
        hydratedDatasets += selectedDataset
    }

    private fun syncUnifiedSpiFromLocal(selectedDataset: String) {
        val session = persistenceSession ?: return

        val graphFile = graphPath(selectedDataset)
        val chunksFile = chunksPath(selectedDataset)
        val corpusFile = corpusPath(selectedDataset)

        val fileEntries = mutableMapOf<String, Any?>()
        if (graphFile.exists()) fileEntries["graph_json"] = Files.readString(graphFile)
        if (chunksFile.exists()) fileEntries["chunks_text"] = Files.readString(chunksFile)
        if (corpusFile.exists()) fileEntries["corpus_json"] = Files.readString(corpusFile)
        if (fileEntries.isNotEmpty()) {
            session.kv(filesNamespace(selectedDataset)).putAll(fileEntries)
        }

        val normalized = normalizeInspectionFromVisualization(selectedDataset)
        session.graph(graphNamespace(selectedDataset)).restore(toGraphSnapshot(normalized))
        session.kv(metadataNamespace(selectedDataset)).putAll(
            mapOf(
                "datasetName" to selectedDataset,
                "updatedAt" to Instant.now().toString(),
                "persistenceBackend" to session.backendId,
                "nodeCount" to ((normalized["metadata"] as? Map<*, *>)?.get("nodeCount") ?: 0),
                "edgeCount" to ((normalized["metadata"] as? Map<*, *>)?.get("edgeCount") ?: 0),
            ),
        )
        hydratedDatasets += selectedDataset
    }

    private fun clearUnifiedSpiState(selectedDataset: String) {
        val session = persistenceSession ?: return
        session.graph(graphNamespace(selectedDataset)).restore(GraphSnapshot())
        session.kv(filesNamespace(selectedDataset)).restore(KvSnapshot())
        session.kv(metadataNamespace(selectedDataset)).restore(KvSnapshot())
        hydratedDatasets.remove(selectedDataset)
    }

    private fun normalizeInspectionFromVisualization(selectedDataset: String): Map<String, Any?> {
        val visualization = graphService.loadGraphVisualization(selectedDataset)
        val raw = jsonObjectToMap(visualization)
        val nodes = (raw["nodes"] as? List<*>)?.map { asStringMap(it) } ?: emptyList()
        val links = (raw["links"] as? List<*>)?.map { asStringMap(it) } ?: emptyList()
        val edges =
            links.map { link ->
                mapOf(
                    "source" to link["source"]?.toString().orEmpty(),
                    "target" to link["target"]?.toString().orEmpty(),
                    "relation" to (link["name"]?.toString() ?: ""),
                    "weight" to link["value"],
                )
            }
        val stats = asStringMap(raw["stats"])
        val metadata =
            mutableMapOf<String, Any?>(
                "datasetName" to selectedDataset,
                "nodeCount" to (stats["total_nodes"] ?: nodes.size),
                "edgeCount" to (stats["total_edges"] ?: edges.size),
            )
        if (unifiedSpiEnabled) {
            metadata["persistenceBackend"] = persistenceSession?.backendId
            metadata["unifiedSpiEnabled"] = true
        }
        return normalizeGraphInspection(
            mapOf(
                "nodes" to nodes,
                "edges" to edges,
                "metadata" to metadata,
            ),
        )
    }

    private fun writeText(
        path: Path,
        content: String,
    ) {
        path.parent?.createDirectories()
        Files.writeString(path, content)
    }

    private fun graphNamespace(selectedDataset: String): String = "${namespaceBase(selectedDataset)}_graph"

    private fun filesNamespace(selectedDataset: String): String = "${namespaceBase(selectedDataset)}_files"

    private fun metadataNamespace(selectedDataset: String): String = "${namespaceBase(selectedDataset)}_metadata"

    private fun namespaceBase(selectedDataset: String): String {
        val safeDataset = selectedDataset.replace(Regex("[^A-Za-z0-9._-]"), "_")
        return "${persistenceNamespacePrefix}_$safeDataset"
    }

    private fun toGraphSnapshot(normalized: Map<String, Any?>): GraphSnapshot {
        val nodes =
            (normalized["nodes"] as? List<*>).orEmpty().mapNotNull { raw ->
                val row = asStringMap(raw)
                val id = row["id"]?.toString()?.trim().orEmpty()
                if (id.isBlank()) null else GraphNodeRecord(id = id, data = row - "id")
            }
        val edges =
            (normalized["edges"] as? List<*>).orEmpty().mapNotNull { raw ->
                val row = asStringMap(raw)
                val source = row["source"]?.toString()?.trim().orEmpty()
                val target = row["target"]?.toString()?.trim().orEmpty()
                if (source.isBlank() || target.isBlank()) {
                    null
                } else {
                    GraphEdgeRecord(source = source, target = target, data = row - setOf("source", "target"))
                }
            }
        return GraphSnapshot(
            nodes = nodes,
            edges = edges,
            metadata = asStringMap(normalized["metadata"]),
        )
    }

    @Suppress("UNCHECKED_CAST")
    private fun jsonObjectToMap(value: JsonObject?): Map<String, Any?> {
        if (value == null) return emptyMap()
        return jsonElementToAny(value) as? Map<String, Any?> ?: emptyMap()
    }

    private fun jsonElementToAny(element: JsonElement): Any? =
        when (element) {
            JsonNull -> {
                null
            }

            is JsonObject -> {
                element.entries.associate { (k, v) -> k to jsonElementToAny(v) }
            }

            is JsonArray -> {
                element.map { jsonElementToAny(it) }
            }

            is JsonPrimitive -> {
                when {
                    element.isString -> element.content
                    element.content == "true" -> true
                    element.content == "false" -> false
                    element.content.toLongOrNull() != null -> element.content.toLong()
                    element.content.toDoubleOrNull() != null -> element.content.toDouble()
                    else -> element.content
                }
            }
        }

    private fun modeFor(mode: UnifiedMode): String =
        when (mode) {
            UnifiedMode.BASIC,
            UnifiedMode.LOCAL,
            UnifiedMode.NAIVE,
            UnifiedMode.BYPASS,
            -> "noagent"

            else -> "agent"
        }

    companion object {
        val CAPABILITIES =
            RagCapabilities(
                supportedModes = UnifiedMode.entries.toSet(),
                supportsStreaming = false,
                supportsGraphPaths = false,
                supportsFollowUpQueries = true,
                supportsReferences = true,
            )
    }
}
