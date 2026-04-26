package hipporag

import hipporag.config.BaseConfig
import hipporag.utils.QuerySolution
import kotlinx.coroutines.runBlocking
import shared.config.CommonRagConfigLoader
import shared.rag.CommonRag
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

    override suspend fun aupsert(data: Collection<String>) {
        val documents = normalizeDocuments(data)
        if (documents.isEmpty()) return
        hippoRag.index(documents)
    }

    override fun drop() = runBlocking { adrop() }

    override suspend fun adrop() {
        deleteDirectory(workingDir())
        Files.deleteIfExists(openIeResultsPath())
        reinitialize()
    }

    override fun saveGraph(path: String) = runBlocking { asaveGraph(path) }

    override suspend fun asaveGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        val target = Path.of(path).toAbsolutePath().normalize()
        deleteDirectory(target)
        Files.createDirectories(target)

        val sourceWorkingDir = workingDir()
        require(Files.exists(sourceWorkingDir) && Files.isDirectory(sourceWorkingDir)) {
            "HippoRAG working directory does not exist: $sourceWorkingDir"
        }
        copyDirectory(sourceWorkingDir, target.resolve("working_dir"))

        val sourceOpenie = openIeResultsPath()
        if (Files.exists(sourceOpenie) && Files.isRegularFile(sourceOpenie)) {
            Files.copy(
                sourceOpenie,
                target.resolve(sourceOpenie.fileName.toString()),
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.COPY_ATTRIBUTES,
            )
        }
    }

    override fun loadGraph(path: String) = runBlocking { aloadGraph(path) }

    override suspend fun aloadGraph(path: String) {
        require(path.isNotBlank()) { "path must not be blank" }
        val sourceRoot = Path.of(path).toAbsolutePath().normalize()
        require(Files.exists(sourceRoot) && Files.isDirectory(sourceRoot)) {
            "HippoRAG snapshot directory not found: $sourceRoot"
        }

        val sourceWorkingDirCandidate = sourceRoot.resolve("working_dir")
        val sourceWorkingDir =
            when {
                Files.exists(sourceWorkingDirCandidate) && Files.isDirectory(sourceWorkingDirCandidate) -> sourceWorkingDirCandidate
                Files.exists(sourceRoot.resolve("graph.json")) -> sourceRoot
                else -> error("Invalid HippoRAG snapshot: expected 'working_dir/' or 'graph.json' in $sourceRoot")
            }

        val targetWorkingDir = workingDir()
        deleteDirectory(targetWorkingDir)
        copyDirectory(sourceWorkingDir, targetWorkingDir)

        val targetOpenie = openIeResultsPath()
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

        reinitialize()
    }

    override fun query(
        query: String,
        param: QueryParam,
    ): QuerySolution = runBlocking { aquery(query, param) }

    override suspend fun aquery(
        query: String,
        param: QueryParam,
    ): QuerySolution {
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

        return solution
    }

    private fun reinitialize() {
        hippoRag = HippoRag(initialConfig = config.copy())
        config = hippoRag.globalConfig.copy()
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
        Files
            .walk(normalized)
            .sorted(Comparator.reverseOrder())
            .forEach { Files.deleteIfExists(it) }
    }
}
