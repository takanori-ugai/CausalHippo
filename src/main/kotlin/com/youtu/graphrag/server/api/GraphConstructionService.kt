package com.youtu.graphrag.server.api

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.constructor.KTBuilder
import com.youtu.graphrag.shared.graph.GraphRelationship
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import java.nio.file.Path
import kotlin.io.path.createDirectories
import kotlin.io.path.deleteIfExists
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.io.path.listDirectoryEntries
import kotlin.io.path.name

class DatasetNotFoundException(
    message: String,
) : RuntimeException(message)

data class GraphConstructionResult(
    val success: Boolean,
    val message: String,
    val graphData: JsonObject,
)

class GraphConstructionService(
    private val config: ConfigManager,
    private val rootDir: Path = Path.of("."),
) {
    private val logger = KotlinLogging.logger {}
    private val mapper = ObjectMapper().registerKotlinModule()
    private val json = Json { ignoreUnknownKeys = true }

    private val graphsDir = resolveConfiguredPath(config.output.graphsDir)
    private val chunksDir = resolveConfiguredPath(config.output.chunksDir)
    private val logsDir = resolveConfiguredPath(config.output.logsDir)
    private val cacheDir = resolveConfiguredPath(config.retrieval.cacheDir)
    private val schemasDir = rootDir.resolve("schemas")

    private fun requireSafeDatasetName(datasetName: String): String {
        require(Regex("^[A-Za-z0-9_-]+$").matches(datasetName)) {
            "Invalid dataset name"
        }
        return datasetName
    }

    fun constructGraph(datasetName: String): GraphConstructionResult {
        val safeDatasetName = requireSafeDatasetName(datasetName)
        clearCacheFiles(safeDatasetName)

        val corpusPath =
            resolveCorpusPathWithDemoFallback(safeDatasetName)
                ?: throw DatasetNotFoundException("Dataset not found")
        val schemaPath = resolveSchemaPath(safeDatasetName)

        val builder =
            KTBuilder(
                datasetName = safeDatasetName,
                schemaPath = schemaPath.toString(),
                mode = config.construction.mode,
                config = config,
                rootDir = rootDir,
            )

        builder.buildKnowledgeGraph(corpusPath.toString())
        val visualization = loadGraphVisualization(safeDatasetName) ?: defaultGraphVisualization()

        return GraphConstructionResult(
            success = true,
            message = "Graph construction completed!",
            graphData = visualization,
        )
    }

    fun reconstructGraph(datasetName: String): GraphConstructionResult {
        val safeDatasetName = requireSafeDatasetName(datasetName)
        val corpusPath =
            resolveCorpusPathForReconstruct(safeDatasetName)
                ?: throw DatasetNotFoundException("Dataset not found")

        clearCacheFiles(safeDatasetName)
        val schemaPath = resolveSchemaPath(safeDatasetName)

        val builder =
            KTBuilder(
                datasetName = safeDatasetName,
                schemaPath = schemaPath.toString(),
                mode = config.construction.mode,
                config = config,
                rootDir = rootDir,
            )

        builder.buildKnowledgeGraph(corpusPath.toString())
        val visualization = loadGraphVisualization(safeDatasetName) ?: defaultGraphVisualization()

        return GraphConstructionResult(
            success = true,
            message = "Dataset reconstructed successfully",
            graphData = visualization,
        )
    }

    fun loadGraphVisualization(datasetName: String): JsonObject? {
        val safeDatasetName = requireSafeDatasetName(datasetName)
        val graphFile = graphsDir.resolve("${safeDatasetName}_new.json")
        if (!graphFile.exists()) {
            return null
        }

        val relationships =
            runCatching {
                mapper.readValue(graphFile.toFile(), object : TypeReference<List<GraphRelationship>>() {})
            }.getOrElse { error ->
                logger.warn(error) { "Failed to parse graph file: $graphFile" }
                emptyList()
            }

        val payload = buildVisualization(relationships)
        return json.parseToJsonElement(mapper.writeValueAsString(payload)).jsonObject
    }

    fun clearCacheFiles(datasetName: String) {
        val safeDatasetName = requireSafeDatasetName(datasetName)
        deleteRecursivelyIfExists(cacheDir.resolve(safeDatasetName))
        deleteRecursivelyIfExists(chunksDir.resolve("$safeDatasetName.txt"))
        deleteRecursivelyIfExists(graphsDir.resolve("${safeDatasetName}_new.json"))

        deleteDatasetPrefixEntries(logsDir, safeDatasetName)
        deleteDatasetPrefixEntries(chunksDir, safeDatasetName)
        deleteDatasetPrefixEntries(graphsDir, safeDatasetName)
    }

    private fun resolveCorpusPathWithDemoFallback(datasetName: String): Path? {
        val uploadedCorpus = rootDir.resolve("data/uploaded/$datasetName/corpus.json")
        if (uploadedCorpus.exists()) {
            return uploadedCorpus
        }

        val demoCorpus = rootDir.resolve("data/demo/demo_corpus.json")
        if (demoCorpus.exists()) {
            return demoCorpus
        }

        return null
    }

    private fun resolveCorpusPathForReconstruct(datasetName: String): Path? {
        if (datasetName == "demo") {
            val demoCorpus = rootDir.resolve("data/demo/demo_corpus.json")
            return if (demoCorpus.exists()) demoCorpus else null
        }

        val uploadedCorpus = rootDir.resolve("data/uploaded/$datasetName/corpus.json")
        return if (uploadedCorpus.exists()) uploadedCorpus else null
    }

    private fun resolveSchemaPath(datasetName: String): Path {
        val datasetSchema = schemasDir.resolve("$datasetName.json")
        if (datasetName != "demo" && datasetSchema.exists()) {
            return datasetSchema
        }

        return ensureDemoSchemaExists()
    }

    private fun ensureDemoSchemaExists(): Path {
        val demoSchemaPath = schemasDir.resolve("demo.json")
        if (demoSchemaPath.exists()) {
            return demoSchemaPath
        }

        schemasDir.createDirectories()
        mapper.writerWithDefaultPrettyPrinter().writeValue(demoSchemaPath.toFile(), DEMO_SCHEMA)
        return demoSchemaPath
    }

    private fun buildVisualization(relationships: List<GraphRelationship>): Map<String, Any> {
        val nodes = mutableListOf<Map<String, Any>>()
        val links = mutableListOf<Map<String, Any>>()
        val categories = linkedSetOf<String>()
        val nodeSeen = linkedSetOf<String>()

        relationships.forEach { relationship ->
            val startName =
                relationship.startNode.properties["name"]
                    ?.toString()
                    .orEmpty()
            val endName =
                relationship.endNode.properties["name"]
                    ?.toString()
                    .orEmpty()

            addNodeIfNeeded(nodeSeen, nodes, categories, relationship.startNode.label, startName)
            addNodeIfNeeded(nodeSeen, nodes, categories, relationship.endNode.label, endName)

            links.add(
                mapOf(
                    "source" to startName,
                    "target" to endName,
                    "name" to relationship.relation,
                    "value" to 1,
                ),
            )
        }

        return mapOf(
            "nodes" to nodes.take(500),
            "links" to links.take(1000),
            "categories" to categories.map { category -> mapOf("name" to category) },
            "stats" to
                mapOf(
                    "total_nodes" to nodes.size,
                    "total_edges" to links.size,
                    "displayed_nodes" to minOf(nodes.size, 500),
                    "displayed_edges" to minOf(links.size, 1000),
                ),
        )
    }

    private fun addNodeIfNeeded(
        nodeSeen: MutableSet<String>,
        nodes: MutableList<Map<String, Any>>,
        categories: MutableSet<String>,
        label: String,
        name: String,
    ) {
        if (name.isBlank()) {
            return
        }
        if (!nodeSeen.add(name)) {
            return
        }

        categories.add(label)
        nodes.add(
            mapOf(
                "id" to name,
                "name" to name.take(30),
                "category" to label,
                "value" to 1,
                "symbolSize" to
                    when (label) {
                        "entity" -> 30
                        "attribute" -> 20
                        "keyword" -> 22
                        else -> 18
                    },
            ),
        )
    }

    private fun deleteDatasetPrefixEntries(
        directory: Path,
        datasetName: String,
    ) {
        if (!directory.exists() || !directory.isDirectory()) {
            return
        }

        val prefix = "${datasetName}_"
        directory.listDirectoryEntries().forEach { entry ->
            if (entry.name.startsWith(prefix)) {
                deleteRecursivelyIfExists(entry)
            }
        }
    }

    private fun deleteRecursivelyIfExists(path: Path) {
        if (!path.exists()) {
            return
        }

        if (path.isDirectory()) {
            path.listDirectoryEntries().forEach { child ->
                deleteRecursivelyIfExists(child)
            }
        }

        path.deleteIfExists()
    }

    private fun resolveConfiguredPath(configuredPath: String): Path {
        val raw = Path.of(configuredPath)
        return if (raw.isAbsolute) raw else rootDir.resolve(raw)
    }

    private fun defaultGraphVisualization(): JsonObject =
        json.parseToJsonElement(mapper.writeValueAsString(DEFAULT_GRAPH_VISUALIZATION)).jsonObject

    companion object {
        private val DEMO_SCHEMA: Map<String, Any> =
            mapOf(
                "Nodes" to
                    listOf(
                        "person",
                        "location",
                        "organization",
                        "event",
                        "object",
                        "concept",
                        "time_period",
                        "creative_work",
                        "biological_entity",
                        "natural_phenomenon",
                    ),
                "Relations" to
                    listOf(
                        "is_a",
                        "part_of",
                        "located_in",
                        "created_by",
                        "used_by",
                        "participates_in",
                        "related_to",
                        "belongs_to",
                        "influences",
                        "precedes",
                        "arrives_in",
                        "comparable_to",
                    ),
                "Attributes" to
                    listOf(
                        "name",
                        "date",
                        "size",
                        "type",
                        "description",
                        "status",
                        "quantity",
                        "value",
                        "position",
                        "duration",
                        "time",
                    ),
            )

        private val DEFAULT_GRAPH_VISUALIZATION: Map<String, Any> =
            mapOf(
                "nodes" to
                    listOf(
                        mapOf(
                            "id" to "node1",
                            "name" to "Example Entity 1",
                            "category" to "person",
                            "value" to 5,
                            "symbolSize" to 25,
                        ),
                        mapOf(
                            "id" to "node2",
                            "name" to "Example Entity 2",
                            "category" to "location",
                            "value" to 3,
                            "symbolSize" to 20,
                        ),
                    ),
                "links" to
                    listOf(
                        mapOf(
                            "source" to "node1",
                            "target" to "node2",
                            "name" to "located_in",
                            "value" to 1,
                        ),
                    ),
                "categories" to
                    listOf(
                        mapOf("name" to "person"),
                        mapOf("name" to "location"),
                    ),
                "stats" to
                    mapOf(
                        "total_nodes" to 2,
                        "total_edges" to 1,
                        "displayed_nodes" to 2,
                        "displayed_edges" to 1,
                    ),
            )
    }
}
