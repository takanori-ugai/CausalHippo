package lightrag.operate

import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.service.AiServices
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.CancellationException
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import lightrag.core.Constants
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.llm.ENTITY_EXTRACTION_INSTRUCTIONS
import lightrag.llm.EntityExtractor
import lightrag.utils.computeMd5
import shared.chunking.DEFAULT_TIKTOKEN_MODEL
import shared.chunking.chunkByTokenSizeWithOverlap
import shared.chunking.countTokensWithJTokKit

private val logger = KotlinLogging.logger {}

/**
 * The result of a chunking operation.
 * @property tokens The number of tokens in the chunk.
 * @property content The content of the chunk.
 * @property chunkOrderIndex The order index of the chunk.
 */
data class ChunkingResult(
    val tokens: Int,
    val content: String,
    val chunkOrderIndex: Int,
)

/**
 * The result of an entity extraction operation.
 * @property entityName The name of the entity.
 * @property entityType The type of the entity.
 * @property description A description of the entity.
 * @property sourceId The ID of the source document.
 */
@Serializable
data class EntityExtractionResult(
    val entityName: String,
    val entityType: String,
    val description: String,
    val sourceId: String,
)

/**
 * The result of a relation extraction operation.
 * @property srcId The ID of the source entity.
 * @property tgtId The ID of the target entity.
 * @property description A description of the relation.
 * @property keywords Keywords associated with the relation.
 * @property weight The weight of the relation.
 * @property sourceId The ID of the source document.
 */
@Serializable
data class RelationExtractionResult(
    val srcId: String,
    val tgtId: String,
    val description: String,
    val keywords: String,
    val weight: Double,
    val sourceId: String,
)

/**
 * Chunks a string by token size.
 * @param tokenizer The tokenizer to use.
 * @param decoder The decoder to use.
 * @param content The content to chunk.
 * @param splitByCharacter The character to split by.
 * @param splitByCharacterOnly Whether to split only by the character.
 * @param chunkOverlapTokenSize The size of the chunk overlap in tokens.
 * @param chunkTokenSize The size of the chunk in tokens.
 * @return A list of [ChunkingResult]s.
 */
fun chunkingByTokenSize(
    // Assuming tokenizer returns list of tokens (Int)
    tokenizer: (String) -> List<Int>,
    decoder: (List<Int>) -> String,
    content: String,
    splitByCharacter: String? = null,
    splitByCharacterOnly: Boolean = false,
    chunkOverlapTokenSize: Int = 100,
    chunkTokenSize: Int = 1200,
    tiktokenModel: String = DEFAULT_TIKTOKEN_MODEL,
): List<ChunkingResult> {
    validateChunkSizes(chunkTokenSize, chunkOverlapTokenSize)

    val processedChunks =
        if (splitByCharacter != null) {
            val rawChunks = content.split(splitByCharacter)
            splitByCharacterChunks(
                rawChunks = rawChunks,
                splitByCharacterOnly = splitByCharacterOnly,
                chunkTokenSize = chunkTokenSize,
                chunkOverlapTokenSize = chunkOverlapTokenSize,
                tokenizer = tokenizer,
                decoder = decoder,
            )
        } else {
            chunkByTokenSizeWithOverlap(
                content = content,
                chunkTokenSize = chunkTokenSize,
                chunkOverlapTokenSize = chunkOverlapTokenSize,
                model = tiktokenModel,
            ).map { it.tokens to it.content }
        }

    return processedChunks.mapIndexed { index, (length, chunk) ->
        ChunkingResult(length, chunk.trim(), index)
    }
}

private fun splitByCharacterChunks(
    rawChunks: List<String>,
    splitByCharacterOnly: Boolean,
    chunkTokenSize: Int,
    chunkOverlapTokenSize: Int,
    tokenizer: (String) -> List<Int>,
    decoder: (List<Int>) -> String,
): List<Pair<Int, String>> {
    val newChunks = mutableListOf<Pair<Int, String>>()
    val overlapStep = chunkTokenSize - chunkOverlapTokenSize

    rawChunks.forEach { chunk ->
        val chunkTokens = tokenizer(chunk)
        when {
            chunkTokens.size <= chunkTokenSize -> {
                newChunks.add(chunkTokens.size to chunk)
            }

            splitByCharacterOnly -> {
                logger.warn {
                    "Chunk split_by_character exceeds token limit: len=${chunkTokens.size} limit=$chunkTokenSize"
                }
                throw IllegalArgumentException("Chunk token limit exceeded: ${chunkTokens.size} > $chunkTokenSize")
            }

            else -> {
                newChunks.addAll(
                    splitTokensWithOverlap(chunkTokens, chunkTokenSize, overlapStep, decoder),
                )
            }
        }
    }
    return newChunks
}

private fun splitTokensWithOverlap(
    tokens: List<Int>,
    chunkTokenSize: Int,
    overlapStep: Int,
    decoder: (List<Int>) -> String,
): List<Pair<Int, String>> {
    val newChunks = mutableListOf<Pair<Int, String>>()
    var start = 0
    while (start < tokens.size) {
        val end = minOf(start + chunkTokenSize, tokens.size)
        val slice = tokens.subList(start, end)
        newChunks.add(slice.size to decoder(slice))
        start += overlapStep
    }
    return newChunks
}

private fun validateChunkSizes(
    chunkTokenSize: Int,
    chunkOverlapTokenSize: Int,
) {
    require(chunkTokenSize > 0) { "chunkTokenSize must be positive" }
    require(chunkOverlapTokenSize >= 0) { "chunkOverlapTokenSize must be non-negative" }
    require(chunkTokenSize > chunkOverlapTokenSize) {
        "chunkTokenSize ($chunkTokenSize) must be greater than chunkOverlapTokenSize ($chunkOverlapTokenSize)"
    }
}

/**
 * Extracts entities and relationships from a map of chunks.
 * @param chunks A map of chunks.
 * @param globalConfig The global configuration.
 * @return A pair of maps containing the extracted entities and relationships.
 */
suspend fun extractEntities(
    chunks: Map<String, Map<String, Any>>,
    globalConfig: Map<String, Any?>,
): Pair<Map<String, List<EntityExtractionResult>>, Map<String, List<RelationExtractionResult>>> {
    if (chunks.isEmpty()) return emptyMap<String, List<EntityExtractionResult>>() to emptyMap()
    val model = checkNotNull(globalConfig["llm_model_func"] as? ChatModel) { "No ChatModel provided for entity extraction" }

    val nodes = mutableMapOf<String, MutableList<EntityExtractionResult>>()
    val edges = mutableMapOf<String, MutableList<RelationExtractionResult>>()

    val entityTypes =
        (globalConfig["entity_types"] as? List<*>)?.joinToString(", ") ?: ""
    val language = globalConfig["language"] as? String ?: "English"

    val glean = (globalConfig["entity_extract_max_gleaning"] as? Number)?.toInt() != 0
    val maxInput = (globalConfig["max_extract_input_tokens"] as? Number)?.toInt() ?: 20480
    val entityExtractor = AiServices.create(EntityExtractor::class.java, model)

    chunks.forEach { (chunkKey, chunkData) ->
        val content = chunkData["content"] as? String ?: return@forEach

        try {
            val extractionResult = entityExtractor.extract(content, entityTypes, language)
            val (initialNodes, initialEdges) = processExtractionResult(extractionResult, chunkKey)
            val chunkNodes = initialNodes.toMutableMap()
            val chunkEdges = initialEdges.toMutableMap()
            if (glean) {
                val request =
                    "Review the previous extraction against the original text. Return only missing or corrected records. " +
                        "Use empty arrays when there are no corrections.\nOriginal text:\n$content\nPrevious extraction:\n" +
                        Json.encodeToString(extractionResult)
                val system = ENTITY_EXTRACTION_INSTRUCTIONS.replace("{{entity_types}}", entityTypes).replace("{{language}}", language)
                if (maxInput <= 0 || countTokensWithJTokKit(system + "\n" + request) <= maxInput) {
                    val corrections = entityExtractor.correct(request, entityTypes, language)
                    val (moreNodes, moreEdges) = processExtractionResult(corrections, chunkKey)
                    mergeCorrections(chunkNodes, moreNodes) { it.description }
                    mergeCorrections(chunkEdges, moreEdges) { it.description }
                }
            }

            chunkNodes.forEach { (name, list) ->
                nodes.computeIfAbsent(name) { mutableListOf() }.addAll(list)
            }
            chunkEdges.forEach { (key, list) ->
                edges.computeIfAbsent(key) { mutableListOf() }.addAll(list)
            }
        } catch (error: CancellationException) {
            throw error
        } catch (error: RuntimeException) {
            throw IllegalStateException("LightRAG extraction failed for chunk $chunkKey", error)
        }
    }

    return nodes to edges
}

private fun <T> mergeCorrections(
    initial: MutableMap<String, List<T>>,
    corrections: Map<String, List<T>>,
    description: (T) -> String,
) {
    for ((key, rows) in corrections) {
        val previous = initial[key]
        if (previous == null || description(rows.first()).length > description(previous.first()).length) initial[key] = rows
    }
}

private fun normalizeName(value: String): String = value.trim().removeSurrounding("\"").trim()

/**
 * Processes the result of an extraction operation.
 * @param result The extraction result.
 * @param chunkKey The key of the chunk.
 * @return A pair of maps containing the processed entities and relationships.
 */
fun processExtractionResult(
    result: ExtractionResult,
    chunkKey: String,
): Pair<Map<String, List<EntityExtractionResult>>, Map<String, List<RelationExtractionResult>>> {
    val nodes = mutableMapOf<String, MutableList<EntityExtractionResult>>()
    val edges = mutableMapOf<String, MutableList<RelationExtractionResult>>()

    result.entities.forEach { entity ->
        val name = normalizeName(entity.name)
        if (name.isEmpty() || entity.type.isBlank() || entity.description.isBlank()) return@forEach
        nodes
            .computeIfAbsent(name) { mutableListOf() }
            .add(EntityExtractionResult(name, entity.type.trim(), entity.description.trim(), chunkKey))
    }

    result.relations.forEach { relation ->
        val from = normalizeName(relation.source)
        val to = normalizeName(relation.target)
        if (from.isEmpty() || to.isEmpty() || relation.description.isBlank()) return@forEach
        val source = minOf(from, to)
        val target = maxOf(from, to)
        if (source == target) return@forEach
        edges
            .computeIfAbsent(relationKey(source, target)) { mutableListOf() }
            .add(
                RelationExtractionResult(
                    source,
                    target,
                    relation.description,
                    relation.keywords,
                    1.0,
                    chunkKey,
                ),
            )
    }

    return nodes to edges
}

/**
 * Merges nodes and edges into the knowledge graph.
 * @param nodes A map of nodes.
 * @param edges A map of edges.
 * @param knowledgeGraphInst The knowledge graph instance.
 * @param entitiesVdb The vector storage for entities.
 * @param relationshipsVdb The vector storage for relationships.
 * @param fullEntities The key-value storage for full entities.
 * @param fullRelations The key-value storage for full relations.
 */
suspend fun mergeNodesAndEdges(
    nodes: Map<String, List<EntityExtractionResult>>,
    edges: Map<String, List<RelationExtractionResult>>,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    fullEntities: BaseKVStorage? = null,
    fullRelations: BaseKVStorage? = null,
    chunkFilePaths: Map<String, String> = emptyMap(),
    globalConfig: Map<String, Any?> = emptyMap(),
) {
    val descriptions = DescriptionMerger(globalConfig)
    // 1. Process Nodes
    for ((name, entityList) in nodes) {
        upsertNodeAndVectors(name, entityList, knowledgeGraphInst, entitiesVdb, fullEntities, chunkFilePaths, descriptions)
    }

    // 2. Process Edges
    for ((key, edgeList) in edges) {
        upsertEdgeAndVectors(
            key,
            edgeList,
            knowledgeGraphInst,
            entitiesVdb,
            relationshipsVdb,
            fullEntities,
            fullRelations,
            chunkFilePaths,
            descriptions,
        )
    }
}

internal fun relationKey(
    source: String,
    target: String,
): String =
    if ('#' in source || '#' in target) {
        "${source.length}#$source#$target"
    } else {
        "$source#$target"
    }

private fun mergeGraphField(
    existing: String?,
    values: List<String>,
): String =
    (existing.orEmpty().split(Constants.GRAPH_FIELD_SEP) + values)
        .filter { it.isNotBlank() }
        .distinct()
        .joinToString(Constants.GRAPH_FIELD_SEP)

private suspend fun upsertNodeAndVectors(
    name: String,
    entityList: List<EntityExtractionResult>,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    fullEntities: BaseKVStorage?,
    chunkFilePaths: Map<String, String>,
    descriptions: DescriptionMerger,
) {
    val existing = knowledgeGraphInst.getNode(name)
    val description = descriptions.merge("Entity", name, existing, entityList.map { it.description })
    val types =
        (
            entityList.map {
                it.entityType
            } + listOfNotNull(existing?.get("entity_type"))
        ).filterNot { it.equals("UNKNOWN", ignoreCase = true) }
    val entityType =
        types
            .groupingBy { it }
            .eachCount()
            .maxByOrNull { it.value }
            ?.key
            ?: existing?.get("entity_type") ?: "UNKNOWN"
    val sourceIds = mergeGraphField(existing?.get("source_id"), entityList.map { it.sourceId })

    val nodeData =
        mapOf(
            "entity_id" to name,
            "entity_type" to entityType,
            "source_id" to sourceIds,
            "file_path" to mergeGraphField(existing?.get("file_path"), entityList.mapNotNull { chunkFilePaths[it.sourceId] }),
        ) + description

    writeEntity(name, nodeData, knowledgeGraphInst, entitiesVdb, fullEntities)
}

private suspend fun writeEntity(
    name: String,
    nodeData: Map<String, String>,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    fullEntities: BaseKVStorage?,
) {
    knowledgeGraphInst.upsertNode(name, nodeData)
    fullEntities?.upsert(mapOf(name to nodeData))

    // Update VDB
    val entityContent = "$name\n${nodeData["description"].orEmpty()}"
    val vdbData =
        mapOf(
            computeMd5(name) to
                mapOf(
                    "content" to entityContent,
                    "entity_name" to name,
                    "source_id" to nodeData.getValue("source_id"),
                    "entity_type" to (nodeData["entity_type"] ?: "UNKNOWN"),
                    "file_path" to nodeData["file_path"].orEmpty(),
                ),
        )
    entitiesVdb.upsert(vdbData)
}

private suspend fun upsertEdgeAndVectors(
    key: String,
    edgeList: List<RelationExtractionResult>,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    fullEntities: BaseKVStorage?,
    fullRelations: BaseKVStorage?,
    chunkFilePaths: Map<String, String>,
    descriptions: DescriptionMerger,
) {
    val first = edgeList.firstOrNull() ?: return
    val src = first.srcId
    val tgt = first.tgtId
    val existing = knowledgeGraphInst.getEdge(src, tgt)
    val oldSources = existing?.get("source_id")
    val oldSourceIds = oldSources.orEmpty().split(Constants.GRAPH_FIELD_SEP).toSet()
    val description = descriptions.merge("Relationship", "$src — $tgt", existing, edgeList.map { it.description })
    val allKeywords =
        (existing?.get("keywords").orEmpty().split(Constants.GRAPH_FIELD_SEP) + edgeList.map { it.keywords })
            .flatMap { it.split(',', '，') }
            .map(String::trim)
            .filter(String::isNotEmpty)
            .toSortedSet()
            .joinToString(",")
    val weight =
        (existing?.get("weight")?.toDoubleOrNull() ?: 0.0) +
            edgeList.filterNot { it.sourceId in oldSourceIds }.distinctBy { it.sourceId }.sumOf { it.weight }
    val sourceIds = mergeGraphField(oldSources, edgeList.map { it.sourceId })

    val filePath = mergeGraphField(existing?.get("file_path"), edgeList.mapNotNull { chunkFilePaths[it.sourceId] })
    val edgeData =
        mapOf(
            "weight" to weight.toString(),
            "file_path" to filePath,
            "keywords" to allKeywords,
            "source_id" to sourceIds,
            "src_id" to src,
            "tgt_id" to tgt,
        ) + description

    for (endpoint in listOf(src, tgt)) {
        val node = knowledgeGraphInst.getNode(endpoint)
        if (node == null) {
            val placeholder =
                mapOf(
                    "entity_id" to endpoint,
                    "entity_type" to "UNKNOWN",
                    "description" to description.getValue("description"),
                    "source_id" to sourceIds,
                    "file_path" to filePath,
                )
            writeEntity(endpoint, placeholder, knowledgeGraphInst, entitiesVdb, fullEntities)
        } else {
            val mergedSources = mergeGraphField(node["source_id"], sourceIds.split(Constants.GRAPH_FIELD_SEP))
            val mergedPaths = mergeGraphField(node["file_path"], filePath.split(Constants.GRAPH_FIELD_SEP))
            if (mergedSources != node["source_id"] || mergedPaths != node["file_path"].orEmpty()) {
                writeEntity(
                    endpoint,
                    node + mapOf("source_id" to mergedSources, "file_path" to mergedPaths),
                    knowledgeGraphInst,
                    entitiesVdb,
                    fullEntities,
                )
            }
        }
    }
    knowledgeGraphInst.upsertEdge(src, tgt, edgeData)
    fullRelations?.upsert(mapOf(key to edgeData))

    // Update VDB
    val relContent = "$allKeywords\t$src\n$tgt\n${description.getValue("description")}"
    val vdbData =
        mapOf(
            computeMd5(key) to
                mapOf(
                    "content" to relContent,
                    "source_id" to sourceIds,
                    "file_path" to filePath,
                    "src_id" to src,
                    "tgt_id" to tgt,
                ),
        )
    relationshipsVdb.upsert(vdbData)
    val reverseKey = relationKey(tgt, src)
    if (reverseKey != key) {
        relationshipsVdb.delete(listOf(computeMd5(reverseKey)))
        fullRelations?.delete(listOf(reverseKey))
    }
}
