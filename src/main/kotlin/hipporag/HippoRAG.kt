package hipporag

import hipporag.config.BaseConfig
import hipporag.embeddingstore.EmbeddingStore
import hipporag.graph.SimpleGraph
import hipporag.utils.OpenieResults
import hipporag.utils.QuerySolution
import hipporag.utils.jsonWithDefaults
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import shared.config.CommonRagConfigLoader
import shared.rag.CommonRag
import tools.jackson.core.type.TypeReference
import tools.jackson.module.kotlin.jacksonObjectMapper
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.Comparator

data class QueryParam(
    val mode: String = "graph",
    val topK: Int = 5,
    val includeAnswer: Boolean = true,
)

class HippoRAG(
    private var hippoRag: HippoRag,
) : CommonRag<QueryParam, QuerySolution> {
    private var config: BaseConfig = hippoRag.globalConfig.copy()
    private val indexMutex = Mutex()
    private val objectMapper = jacksonObjectMapper()

    constructor(
        config: BaseConfig = BaseConfig(),
        saveDir: String? = null,
        llmModelName: String? = null,
        llmBaseUrl: String? = null,
        embeddingModelName: String? = null,
        embeddingBaseUrl: String? = null,
    ) : this(
        HippoRag(
            initialConfig =
                config.copy().apply {
                    saveDir?.let { this.saveDir = it }
                    llmModelName?.let { this.llmName = it }
                    llmBaseUrl?.let { this.llmBaseUrl = it }
                    embeddingModelName?.let { this.embeddingModelName = it }
                    embeddingBaseUrl?.let { this.embeddingBaseUrl = it }
                },
        ),
    )

    companion object {
        fun fromCommonConfig(configPath: String): HippoRAG {
            val commonConfig = CommonRagConfigLoader.load(configPath)
            return HippoRAG(commonConfig.toHippoBaseConfig())
        }
    }

    override fun upsert(data: String) = runBlocking { aupsert(data) }

    override fun upsert(data: Collection<String>) = runBlocking { aupsert(data) }

    override suspend fun aupsert(data: String) = aupsert(listOf(data))

    override suspend fun aupsert(data: Collection<String>) =
        indexMutex.withLock {
            val documents = normalizeDocuments(data)
            if (documents.isNotEmpty()) hippoRag.index(documents)
        }

    override fun drop() = runBlocking { adrop() }

    override suspend fun adrop() =
        indexMutex.withLock {
            deleteDirectory(workingDir())
            Files.deleteIfExists(openIeResultsPath())
            reinitialize()
        }

    override fun saveGraph(path: String) = runBlocking { asaveGraph(path) }

    override suspend fun asaveGraph(path: String) =
        indexMutex.withLock {
            require(path.isNotBlank()) { "path must not be blank" }
            val target =
                Path
                    .of(path)
                    .toFile()
                    .canonicalFile
                    .toPath()
            val working = workingDir()
            val openie = openIeResultsPath()
            requireDisjoint(target, working)
            requireDisjoint(target, openie)

            Files.createDirectories(target.parent)
            withSnapshotStaging(target.parent) { staging ->
                val snapshot = Files.createDirectory(staging.resolve("snapshot"))
                copyDirectory(working, snapshot.resolve("working_dir"))
                validateSnapshot(snapshot.resolve("working_dir"))
                if (Files.isRegularFile(openie)) Files.copy(openie, snapshot.resolve(openie.fileName))
                publishSnapshot(staging, mapOf(target to snapshot))
            }
        }

    override fun loadGraph(path: String) = runBlocking { aloadGraph(path) }

    override suspend fun aloadGraph(path: String) =
        indexMutex.withLock {
            require(path.isNotBlank()) { "path must not be blank" }
            val source =
                Path
                    .of(path)
                    .toFile()
                    .canonicalFile
                    .toPath()
            val working = workingDir()
            val openie = openIeResultsPath()
            requireDisjoint(source, working)
            requireDisjoint(source, openie)
            val sourceWorking = source.resolve("working_dir").takeIf(Files::isDirectory) ?: source
            require(Files.isDirectory(sourceWorking)) { "HippoRAG snapshot directory not found: $sourceWorking" }
            requireDisjoint(sourceWorking, working)

            withSnapshotStaging(working.parent) { staging ->
                val stagedWorking = staging.resolve("working_dir")
                copyDirectory(sourceWorking, stagedWorking)
                validateSnapshot(stagedWorking)
                val stagedOpenie =
                    source.resolve(openie.fileName).takeIf(Files::isRegularFile)?.let { sourceOpenie ->
                        val json = jsonWithDefaults { ignoreUnknownKeys = true }
                        json.decodeFromString(OpenieResults.serializer(), Files.readString(sourceOpenie))
                        Files.copy(sourceOpenie, staging.resolve(openie.fileName))
                    }
                publishSnapshot(staging, mapOf(working to stagedWorking, openie to stagedOpenie)) {
                    reinitialize(forceIndexFromScratch = false)
                }
            }
        }

    private fun requireDisjoint(
        first: Path,
        second: Path,
    ) {
        val a = first.toFile().canonicalFile.toPath()
        val b = second.toFile().canonicalFile.toPath()
        require(!a.startsWith(b) && !b.startsWith(a)) { "HippoRAG snapshot overlaps the live index: $first" }
    }

    private fun validateSnapshot(path: Path) {
        val graph = SimpleGraph.load(path.resolve("graph.json").toFile(), strict = true)
        val chunks = EmbeddingStore.validateFile(path.resolve("chunk_embeddings/vdb_chunk.json").toFile())
        val entities = EmbeddingStore.validateFile(path.resolve("entity_embeddings/vdb_entity.json").toFile())
        val facts = EmbeddingStore.validateFile(path.resolve("fact_embeddings/vdb_fact.json").toFile())
        require(graph.vertexNames().toSet() == chunks + entities && (entities.isEmpty() || facts.isNotEmpty())) {
            "HippoRAG snapshot graph and embedding stores are inconsistent: $path"
        }
    }

    private fun withSnapshotStaging(
        directory: Path,
        action: (Path) -> Unit,
    ) {
        val staging = Files.createTempDirectory(directory, ".hipporag-snapshot-")
        try {
            action(staging)
        } finally {
            // If rollback itself failed, keep the backup so the prior index can be recovered.
            val hasBackup = Files.list(staging).use { paths -> paths.anyMatch { it.fileName.toString().startsWith("previous-") } }
            if (!hasBackup) deleteDirectory(staging)
        }
    }

    private fun publishSnapshot(
        staging: Path,
        replacements: Map<Path, Path?>,
        initialize: () -> Unit = {},
    ) {
        val previous = mutableMapOf<Path, Path>()
        val published = mutableListOf<Path>()
        try {
            replacements.entries.forEachIndexed { index, (target, replacement) ->
                if (Files.exists(target)) {
                    val backup = staging.resolve("previous-$index")
                    Files.move(target, backup)
                    previous[target] = backup
                }
                if (replacement != null) {
                    Files.move(replacement, target)
                    published.add(target)
                }
            }
            initialize()
        } catch (failure: Exception) {
            for (target in replacements.keys.reversed()) {
                try {
                    if (target in published) deleteDirectory(target)
                    previous[target]?.let { Files.move(it, target) }
                } catch (rollbackFailure: Exception) {
                    failure.addSuppressed(rollbackFailure)
                }
            }
            throw failure
        }
        previous.values.forEach(::deleteDirectory)
    }

    override fun inspectGraph(): Map<String, Any?> = runBlocking { ainspectGraph() }

    override suspend fun ainspectGraph(): Map<String, Any?> = indexMutex.withLock { inspectGraphFromDisk() }

    private fun inspectGraphFromDisk(): Map<String, Any?> {
        val graphPath = workingDir().resolve("graph.json")
        if (!Files.exists(graphPath) || !Files.isRegularFile(graphPath)) {
            return emptyGraphInspection()
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

    override fun query(
        query: String,
        param: QueryParam,
    ): QuerySolution = runBlocking { aquery(query, param) }

    override suspend fun aquery(
        query: String,
        param: QueryParam,
    ): QuerySolution =
        indexMutex.withLock {
            val topK = param.topK.coerceAtLeast(1)
            val mode = param.mode.lowercase()

            val retrievalSolutions =
                when (mode) {
                    "graph", "hippo" -> hippoRag.retrieve(listOf(query), numToRetrieve = topK).first
                    "dpr" -> hippoRag.retrieveDpr(listOf(query), numToRetrieve = topK).first
                    else -> throw IllegalArgumentException("Unsupported HippoRAG query mode '$mode'. Use one of: graph, dpr.")
                }
            var solution = retrievalSolutions.firstOrNull() ?: QuerySolution(question = query, docs = emptyList())

            if (param.includeAnswer) {
                val qaResult =
                    when (mode) {
                        "graph", "hippo" -> hippoRag.ragQaWithSolutions(listOf(solution))
                        "dpr" -> hippoRag.ragQaDprWithSolutions(listOf(solution))
                        else -> error("Unreachable mode branch.")
                    }
                solution = qaResult.solutions.firstOrNull() ?: solution
            }

            solution
        }

    private fun reinitialize(forceIndexFromScratch: Boolean = config.forceIndexFromScratch) {
        val replacement = HippoRag(initialConfig = config.copy(forceIndexFromScratch = forceIndexFromScratch))
        hippoRag = replacement
        config = replacement.globalConfig.copy()
    }

    private fun normalizeDocuments(data: Collection<String>): List<String> =
        data
            .map { it.trim() }
            .filter { it.isNotBlank() }

    private fun workingDir(): Path {
        val llmLabel = config.llmName.replace("/", "_")
        val embeddingLabel = config.embeddingModelName.replace("/", "_")
        return Path.of(config.saveDir, "${llmLabel}_$embeddingLabel").toAbsolutePath().normalize()
    }

    private fun openIeResultsPath(): Path {
        val llmLabel = config.llmName.replace("/", "_")
        return Path.of(config.saveDir, "openie_results_ner_$llmLabel.json").toAbsolutePath().normalize()
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

    private fun emptyGraphInspection(): Map<String, Any?> =
        mapOf(
            "nodes" to emptyList<Map<String, Any?>>(),
            "edges" to emptyList<Map<String, Any?>>(),
            "metadata" to
                mapOf(
                    "nodeCount" to 0,
                    "edgeCount" to 0,
                    "directed" to config.isDirectedGraph,
                ),
        )
}
