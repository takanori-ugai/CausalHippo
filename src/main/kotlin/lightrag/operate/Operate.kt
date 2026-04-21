package lightrag.operate

import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.service.AiServices
import io.github.oshai.kotlinlogging.KotlinLogging
import lightrag.core.Constants
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.llm.EntityExtractor
import lightrag.utils.computeMd5

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
            sequentialChunks(
                tokens = tokenizer(content),
                chunkTokenSize = chunkTokenSize,
                chunkOverlapTokenSize = chunkOverlapTokenSize,
                decoder = decoder,
            )
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

private fun sequentialChunks(
    tokens: List<Int>,
    chunkTokenSize: Int,
    chunkOverlapTokenSize: Int,
    decoder: (List<Int>) -> String,
): List<Pair<Int, String>> {
    val overlapStep = chunkTokenSize - chunkOverlapTokenSize
    return splitTokensWithOverlap(tokens, chunkTokenSize, overlapStep, decoder)
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
    val model = globalConfig["llm_model_func"] as? ChatModel
    if (model == null) {
        logger.error { "No ChatModel provided for entity extraction" }
        return emptyMap<String, List<EntityExtractionResult>>() to emptyMap()
    }

    val nodes = mutableMapOf<String, MutableList<EntityExtractionResult>>()
    val edges = mutableMapOf<String, MutableList<RelationExtractionResult>>()

    val entityTypes =
        (globalConfig["entity_types"] as? List<*>)?.joinToString(", ") ?: ""
    val language = globalConfig["language"] as? String ?: "English"

    val entityExtractor = AiServices.create(EntityExtractor::class.java, model)

    chunks.forEach { (chunkKey, chunkData) ->
        val content = chunkData["content"] as? String ?: return@forEach

        try {
            val extractionResult = entityExtractor.extract(content, entityTypes, language)
            val (chunkNodes, chunkEdges) =
                processExtractionResult(extractionResult, chunkKey)

            chunkNodes.forEach { (name, list) ->
                nodes.computeIfAbsent(name) { mutableListOf() }.addAll(list)
            }
            chunkEdges.forEach { (key, list) ->
                edges.computeIfAbsent(key) { mutableListOf() }.addAll(list)
            }
        } catch (e: IllegalStateException) {
            logger.error(e) { "Illegal state while extracting entities for chunk $chunkKey" }
        } catch (e: IllegalArgumentException) {
            logger.error(e) { "Invalid input while extracting entities for chunk $chunkKey" }
        } catch (e: Exception) {
            logger.error(e) { "Unexpected error while extracting entities for chunk $chunkKey" }
        }
    }

    return nodes to edges
}

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
        nodes
            .computeIfAbsent(entity.name) { mutableListOf() }
            .add(EntityExtractionResult(entity.name, entity.type, entity.description, chunkKey))
    }

    result.relations.forEach { relation ->
        edges
            .computeIfAbsent(relation.key()) { mutableListOf() }
            .add(
                RelationExtractionResult(
                    relation.source,
                    relation.target,
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
) {
    // 1. Process Nodes
    for ((name, entityList) in nodes) {
        upsertNodeAndVectors(name, entityList, knowledgeGraphInst, entitiesVdb, fullEntities)
    }

    // 2. Process Edges
    for ((key, edgeList) in edges) {
        upsertEdgeAndVectors(key, edgeList, knowledgeGraphInst, relationshipsVdb, fullRelations)
    }
}

private fun ExtractedRelation.key(): String = "$source#$target"

private suspend fun upsertNodeAndVectors(
    name: String,
    entityList: List<EntityExtractionResult>,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    fullEntities: BaseKVStorage?,
) {
    // Simple merge: take the longest description and majority type.
    val longestDesc = entityList.maxByOrNull { it.description.length }?.description.orEmpty()
    val majorityType =
        entityList
            .groupingBy { it.entityType }
            .eachCount()
            .maxByOrNull { it.value }
            ?.key ?: "Unknown"
    val sourceIds = entityList.joinToString(Constants.GRAPH_FIELD_SEP) { it.sourceId }

    val nodeData =
        mapOf(
            "entity_id" to name,
            "entity_type" to majorityType,
            "description" to longestDesc,
            "source_id" to sourceIds,
        )

    knowledgeGraphInst.upsertNode(name, nodeData)
    fullEntities?.upsert(mapOf(name to nodeData))

    // Update VDB
    val entityContent = "$name\n$longestDesc"
    val vdbData =
        mapOf(
            computeMd5(name) to
                mapOf(
                    "content" to entityContent,
                    "entity_name" to name,
                ),
        )
    entitiesVdb.upsert(vdbData)
}

private suspend fun upsertEdgeAndVectors(
    key: String,
    edgeList: List<RelationExtractionResult>,
    knowledgeGraphInst: BaseGraphStorage,
    relationshipsVdb: BaseVectorStorage,
    fullRelations: BaseKVStorage?,
) {
    val first = edgeList.firstOrNull() ?: return
    val src = first.srcId
    val tgt = first.tgtId
    val longestDesc = edgeList.maxByOrNull { it.description.length }?.description.orEmpty()
    val allKeywords = edgeList.joinToString(", ") { it.keywords }
    val weight = edgeList.sumOf { it.weight }
    val sourceIds = edgeList.joinToString(Constants.GRAPH_FIELD_SEP) { it.sourceId }

    val edgeData =
        mapOf(
            "weight" to weight.toString(),
            "description" to longestDesc,
            "keywords" to allKeywords,
            "source_id" to sourceIds,
            "src_id" to src,
            "tgt_id" to tgt,
        )

    knowledgeGraphInst.upsertEdge(src, tgt, edgeData)
    fullRelations?.upsert(mapOf(key to edgeData))

    // Update VDB
    val relContent = "$allKeywords\t$src\n$tgt\n$longestDesc"
    val vdbData =
        mapOf(
            computeMd5(key) to
                mapOf(
                    "content" to relContent,
                    "src_id" to src,
                    "tgt_id" to tgt,
                ),
        )
    relationshipsVdb.upsert(vdbData)
}
