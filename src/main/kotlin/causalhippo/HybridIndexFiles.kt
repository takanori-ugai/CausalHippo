package causalhippo

import causalrag.causalgraph.builder.CausalGraphBuilder
import hipporag.config.BaseConfig
import hipporag.graph.SimpleGraph
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.Comparator

internal class HybridIndexFiles(
    val config: BaseConfig,
) {
    private val root = Path.of(config.saveDir).toAbsolutePath().normalize()
    private val llmLabel = config.llmName.replace('/', '_')
    val workingDir: Path = root.resolve("${llmLabel}_${config.embeddingModelName.replace('/', '_')}")
    private val openie = root.resolve("openie_results_ner_$llmLabel.json")
    val causalGraph: Path = workingDir.resolve("causal_graph.json")

    fun <T> withStaging(action: (HybridIndexFiles) -> T): T =
        withTemporaryDirectory(root) { temp ->
            action(HybridIndexFiles(config.copy(saveDir = temp.toString(), forceIndexFromScratch = false)))
        }

    fun copyTo(destination: HybridIndexFiles) {
        copyTree(workingDir, destination.workingDir)
        copyOptional(openie, destination.openie)
    }

    fun install(
        staged: HybridIndexFiles,
        initialize: () -> Unit,
    ) {
        replaceFiles(staged.root, mapOf(workingDir to staged.workingDir, openie to staged.openie), initialize)
    }

    fun saveSnapshot(
        rawTarget: Path,
        graph: CausalGraphBuilder,
    ) {
        val target = checkedPath(rawTarget)
        withTemporaryDirectory(target.parent) { temp ->
            val snapshot = Files.createDirectory(temp.resolve("snapshot"))
            val hippo = snapshot.resolve("hippo_working_dir")
            copyTree(workingDir, hippo)
            if (!Files.exists(hippo.resolve("graph.json"))) {
                SimpleGraph(config.isDirectedGraph).save(hippo.resolve("graph.json").toFile())
            }
            validateHippo(hippo)
            copyOptional(openie, snapshot.resolve(openie.fileName))
            check(graph.save(snapshot.resolve("causal_graph.json").toString())) { "Could not save causal graph" }
            replaceFiles(temp, mapOf(target to snapshot))
        }
    }

    fun readSnapshot(
        rawSource: Path,
        staged: HybridIndexFiles,
    ) {
        val source = checkedPath(rawSource)
        val nested = source.resolve("hippo_working_dir")
        val hippo = if (Files.isDirectory(nested)) nested else source
        require(Files.isDirectory(hippo)) { "Snapshot not found: $source" }
        copyTree(hippo, staged.workingDir)
        validateHippo(staged.workingDir)
        Files.copy(source.resolve("causal_graph.json"), staged.causalGraph, StandardCopyOption.REPLACE_EXISTING)
        copyOptional(source.resolve(openie.fileName), staged.openie)
        if (Files.exists(staged.openie)) {
            Json
                .parseToJsonElement(Files.readString(staged.openie))
                .jsonObject
                .getValue("docs")
                .jsonArray
        }
    }

    private fun checkedPath(path: Path): Path {
        val canonical = path.toFile().canonicalFile.toPath()
        for (active in listOf(workingDir, openie)) {
            val live = active.toFile().canonicalFile.toPath()
            require(!canonical.startsWith(live) && !live.startsWith(canonical)) {
                "Snapshot overlaps live CausalHippo artifacts: $canonical"
            }
        }
        return canonical
    }

    private fun validateHippo(dir: Path) {
        val graph = SimpleGraph.load(dir.resolve("graph.json").toFile())
        val chunks = vectorIds(dir.resolve("chunk_embeddings/vdb_chunk.json"))
        val entities = vectorIds(dir.resolve("entity_embeddings/vdb_entity.json"))
        val facts = vectorIds(dir.resolve("fact_embeddings/vdb_fact.json"))
        require(graph.vertexNames().toSet() == chunks + entities && (entities.isEmpty() || facts.isNotEmpty())) {
            "HippoRAG graph and embedding artifacts are inconsistent: $dir"
        }
    }

    private fun vectorIds(path: Path): Set<String> {
        if (!Files.exists(path)) return emptySet()
        val store = Json.parseToJsonElement(Files.readString(path)).jsonObject
        val ids = store.getValue("hashIds").jsonArray.map { it.jsonPrimitive.content }
        val texts = store.getValue("texts").jsonArray
        val vectors = store.getValue("embeddings").jsonArray
        val dimension = vectors.firstOrNull()?.jsonArray?.size ?: 0
        require(
            ids.distinct().size == ids.size && ids.size == texts.size && ids.size == vectors.size &&
                vectors.all { row ->
                    dimension > 0 && row.jsonArray.size == dimension && row.jsonArray.all { it.jsonPrimitive.double.isFinite() }
                },
        ) { "Invalid embedding artifacts: $path" }
        return ids.toSet()
    }

    private fun <T> withTemporaryDirectory(
        parent: Path,
        action: (Path) -> T,
    ): T {
        Files.createDirectories(parent)
        val directory = Files.createTempDirectory(parent, ".causalhippo-")
        try {
            return action(directory)
        } finally {
            // Retain previous files if rollback itself could not restore them.
            val hasBackups =
                Files.list(directory).use { entries ->
                    entries.anyMatch { it.fileName.toString().startsWith(".previous-") }
                }
            if (!hasBackups) deleteTree(directory)
        }
    }

    private fun replaceFiles(
        staging: Path,
        replacements: Map<Path, Path>,
        initialize: () -> Unit = {},
    ) {
        val backups = mutableMapOf<Path, Path>()
        val published = mutableListOf<Path>()
        try {
            replacements.entries.forEachIndexed { index, (target, staged) ->
                if (Files.exists(target)) {
                    val backup = staging.resolve(".previous-$index")
                    Files.move(target, backup)
                    backups[target] = backup
                }
                if (Files.exists(staged)) {
                    Files.move(staged, target)
                    published.add(target)
                }
            }
            initialize()
        } catch (failure: Exception) {
            replacements.keys.reversed().forEach { target ->
                try {
                    if (target in published) deleteTree(target)
                    backups[target]?.let { Files.move(it, target) }
                } catch (rollback: Exception) {
                    failure.addSuppressed(rollback)
                }
            }
            throw failure
        }
        backups.values.forEach { runCatching { deleteTree(it) } }
    }

    private fun copyTree(
        source: Path,
        target: Path,
    ) {
        Files.createDirectories(target)
        if (!Files.exists(source)) return
        Files.walk(source).use { entries ->
            entries.forEach { entry ->
                val destination = target.resolve(source.relativize(entry))
                if (Files.isDirectory(entry)) {
                    Files.createDirectories(destination)
                } else {
                    Files.copy(entry, destination, StandardCopyOption.REPLACE_EXISTING)
                }
            }
        }
    }

    private fun copyOptional(
        source: Path,
        target: Path,
    ) {
        if (Files.isRegularFile(source)) Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING)
    }

    private fun deleteTree(path: Path) {
        if (!Files.exists(path)) return
        Files.walk(path).use { files -> files.sorted(Comparator.reverseOrder()).forEach { Files.deleteIfExists(it) } }
    }
}
