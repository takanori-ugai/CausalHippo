package com.youtu.graphrag.shared.constructor

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList
import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.graph.GraphNode
import com.youtu.graphrag.shared.graph.GraphRelationship
import com.youtu.graphrag.shared.llm.LlmClient
import com.youtu.graphrag.shared.llm.LlmClientFactory
import com.youtu.graphrag.shared.llm.LlmOutputParser
import com.youtu.graphrag.shared.treecomm.FastTreeComm
import com.youtu.graphrag.shared.treecomm.TreeCommOptions
import io.github.oshai.kotlinlogging.KotlinLogging
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import java.util.Base64
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.math.max
import kotlin.math.min

private data class ChunkRecord(
    val id: String,
    val title: String,
    val text: String,
)

private data class ParsedTriple(
    val subject: String,
    val relation: String,
    val target: String,
)

private data class ExtractionPayload(
    val attributes: Map<String, List<String>>,
    val triples: List<ParsedTriple>,
    val entityTypes: Map<String, String>,
    val newSchemaTypes: Map<String, List<String>>,
)

private data class EntityTripleKey(
    val source: String,
    val relation: String,
    val target: String,
)

class KTBuilder(
    private val datasetName: String,
    private val schemaPath: String,
    private val mode: String,
    private val config: ConfigManager,
    private val rootDir: Path = Path.of("."),
    private val llmClient: LlmClient = LlmClientFactory.fromEnvironment(),
) {
    private val logger = KotlinLogging.logger {}
    private val mapper = ObjectMapper().registerKotlinModule()
    private val schemaFile: Path = resolvePath(schemaPath)
    private var schema: MutableMap<String, Any?> = loadSchema(schemaFile)
    private var llmExtractionDisabled = false
    private val tokenizer: Encoding = TOKENIZER_REGISTRY.getEncoding(EncodingType.CL100K_BASE)

    fun buildKnowledgeGraph(corpusPath: String): List<GraphRelationship> {
        require(datasetName.isNotBlank()) { "datasetName must not be blank" }
        require(schemaPath.isNotBlank()) { "schemaPath must not be blank" }
        require(corpusPath.isNotBlank()) { "corpusPath must not be blank" }
        require(mode.isNotBlank()) { "mode must not be blank" }

        val corpusFile = resolvePath(corpusPath)
        require(corpusFile.exists()) { "Corpus file not found: $corpusFile" }

        val documents = loadDocuments(corpusFile)
        val chunks = createChunks(documents)

        writeChunksFile(chunks)
        val relationships = buildRelationships(chunks)
        writeGraphFile(relationships)

        logger.info {
            "Knowledge graph built for '$datasetName' with ${relationships.size} relation(s) and ${chunks.size} chunk(s)"
        }

        return relationships
    }

    private fun loadSchema(schemaFile: Path): MutableMap<String, Any?> {
        if (!schemaFile.exists()) {
            return mutableMapOf()
        }

        return runCatching {
            mapper.readValue(schemaFile.toFile(), object : TypeReference<MutableMap<String, Any?>>() {})
        }.getOrElse { error ->
            logger.warn(error) { "Failed to read schema file '$schemaFile'. Using empty schema." }
            mutableMapOf()
        }
    }

    private fun loadDocuments(corpusFile: Path): List<Map<String, Any?>> {
        val root = mapper.readTree(corpusFile.toFile())

        return when {
            root.isArray -> {
                root.mapIndexedNotNull { index, node -> toDocumentMap(index, node) }
            }

            root.isObject -> {
                listOf(mapper.convertValue(root, object : TypeReference<Map<String, Any?>>() {}))
            }

            root.isTextual -> {
                listOf(
                    mapOf(
                        "title" to "document_0",
                        "text" to root.asText(),
                    ),
                )
            }

            else -> {
                emptyList()
            }
        }
    }

    private fun toDocumentMap(
        index: Int,
        node: JsonNode,
    ): Map<String, Any?>? =
        when {
            node.isObject -> {
                mapper.convertValue(node, object : TypeReference<Map<String, Any?>>() {})
            }

            node.isTextual -> {
                mapOf(
                    "title" to "document_$index",
                    "text" to node.asText(),
                )
            }

            else -> {
                null
            }
        }

    private fun createChunks(documents: List<Map<String, Any?>>): List<ChunkRecord> {
        val chunkRecords = mutableListOf<ChunkRecord>()
        val skipChunking = datasetName in config.construction.datasetsNoChunk

        documents.forEachIndexed { documentIndex, rawDocument ->
            val originalTitle =
                rawDocument["title"]
                    ?.toString()
                    ?.trim()
                    .orEmpty()
            val title =
                originalTitle
                    .ifBlank { "document_$documentIndex" }
            val text = rawDocument["text"]?.toString()?.trim().orEmpty()
            if (text.isBlank()) {
                return@forEachIndexed
            }

            val chunkInput =
                listOfNotNull(
                    originalTitle.takeIf { it.isNotBlank() },
                    text,
                ).joinToString(" ")
            val chunkTexts =
                if (skipChunking) {
                    listOf(chunkInput)
                } else {
                    splitTextWithOverlap(
                        text = chunkInput,
                        chunkSize = config.construction.chunkSize,
                        overlap = config.construction.overlap,
                        minTailTokens = config.construction.minTailTokens,
                    )
                }

            chunkTexts.forEachIndexed { chunkIndex, chunkText ->
                val chunkId = buildChunkId(documentIndex, chunkIndex, chunkText)
                chunkRecords.add(
                    ChunkRecord(
                        id = chunkId,
                        title = title,
                        text = chunkText,
                    ),
                )
            }
        }

        return chunkRecords
    }

    private fun splitTextWithOverlap(
        text: String,
        chunkSize: Int,
        overlap: Int,
        minTailTokens: Int,
    ): List<String> {
        if (text.isBlank()) {
            return emptyList()
        }

        val normalizedChunkSize = max(1, chunkSize)
        return runCatching {
            val encodedTokens = tokenizer.encode(text)
            if (encodedTokens.size() <= normalizedChunkSize) {
                return@runCatching listOf(text)
            }

            val normalizedOverlap = min(max(0, overlap), normalizedChunkSize - 1)
            val step = (normalizedChunkSize - normalizedOverlap).coerceAtLeast(1)
            val windows = mutableListOf<Pair<Int, Int>>()

            var start = 0
            val totalTokens = encodedTokens.size()
            while (start < totalTokens) {
                val end = min(start + normalizedChunkSize, totalTokens)
                windows.add(start to end)
                start += step
            }

            if (windows.size >= 2) {
                var index = 0
                while (index < windows.size) {
                    val current = windows[index]
                    val currentLength = current.second - current.first
                    if (currentLength < minTailTokens) {
                        when {
                            index > 0 -> {
                                val previous = windows[index - 1]
                                windows[index - 1] = previous.first to current.second
                                windows.removeAt(index)
                                continue
                            }

                            index + 1 < windows.size -> {
                                val next = windows[index + 1]
                                windows[index + 1] = current.first to next.second
                                windows.removeAt(index)
                                continue
                            }
                        }
                    }
                    index += 1
                }
            }

            val tokenArray = encodedTokens.toArray()
            windows.mapNotNull { (windowStart, windowEnd) ->
                val tokenSpanLength = windowEnd - windowStart
                val tokenSpan = IntArrayList(tokenSpanLength)
                for (tokenIndex in windowStart until windowEnd) {
                    tokenSpan.add(tokenArray[tokenIndex])
                }

                val chunk = tokenizer.decode(tokenSpan).trim()
                if (tokenSpanLength < 5 || chunk.length < 5) {
                    null
                } else {
                    chunk
                }
            }
        }.getOrElse { error ->
            logger.warn(error) { "jtokkit token chunking failed for '$datasetName'; falling back to character chunking." }
            splitTextWithCharOverlap(
                text = text,
                chunkSize = normalizedChunkSize,
                overlap = overlap,
                minTailTokens = minTailTokens,
            )
        }
    }

    private fun splitTextWithCharOverlap(
        text: String,
        chunkSize: Int,
        overlap: Int,
        minTailTokens: Int,
    ): List<String> {
        if (text.length <= chunkSize) {
            return listOf(text)
        }

        val normalizedOverlap = min(max(0, overlap), chunkSize - 1)
        val step = (chunkSize - normalizedOverlap).coerceAtLeast(1)
        val windows = mutableListOf<Pair<Int, Int>>()

        var start = 0
        while (start < text.length) {
            val end = min(start + chunkSize, text.length)
            windows.add(start to end)
            start += step
        }

        if (windows.size >= 2) {
            var index = 0
            while (index < windows.size) {
                val current = windows[index]
                val currentLength = current.second - current.first
                if (currentLength < minTailTokens) {
                    when {
                        index > 0 -> {
                            val previous = windows[index - 1]
                            windows[index - 1] = previous.first to current.second
                            windows.removeAt(index)
                            continue
                        }

                        index + 1 < windows.size -> {
                            val next = windows[index + 1]
                            windows[index + 1] = current.first to next.second
                            windows.removeAt(index)
                            continue
                        }
                    }
                }
                index += 1
            }
        }

        return windows.mapNotNull { (windowStart, windowEnd) ->
            val chunk = text.substring(windowStart, windowEnd).trim()
            if (chunk.length < 5) {
                null
            } else {
                chunk
            }
        }
    }

    private fun buildChunkId(
        documentIndex: Int,
        chunkIndex: Int,
        text: String,
    ): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val input = "$datasetName|$documentIndex|$chunkIndex|$text"
        val encoded = Base64.getUrlEncoder().withoutPadding().encodeToString(digest.digest(input.toByteArray(Charsets.UTF_8)))
        return encoded.take(8).padEnd(8, '0')
    }

    private fun buildRelationships(chunks: List<ChunkRecord>): List<GraphRelationship> {
        val relationships = mutableListOf<GraphRelationship>()
        val entityByName = linkedMapOf<String, GraphNode>()
        val seenEntityTriples = linkedSetOf<EntityTripleKey>()

        chunks.forEach { chunk ->
            val extraction = extractChunkKnowledge(chunk.text)
            if (mode == "agent") {
                updateSchemaWithNewTypes(extraction.newSchemaTypes)
            }

            val extractedCount =
                addExtractionRelationships(
                    chunk = chunk,
                    extraction = extraction,
                    relationships = relationships,
                    entityByName = entityByName,
                    seenEntityTriples = seenEntityTriples,
                )
            if (extractedCount == 0) {
                logger.debug { "No extraction relationships generated for chunk ${chunk.id}" }
            }
        }

        addCommunityRelationships(
            relationships = relationships,
            entityByName = entityByName,
        )
        return relationships
    }

    private fun addCommunityRelationships(
        relationships: MutableList<GraphRelationship>,
        entityByName: Map<String, GraphNode>,
    ) {
        val entityRelationships =
            relationships.filter { relationship ->
                relationship.startNode.label == "entity" && relationship.endNode.label == "entity"
            }
        if (entityRelationships.isEmpty()) {
            return
        }

        val maxCommunities = config.treeComm.maxTotalCommunities.coerceAtLeast(1)
        val treeCommOptions =
            TreeCommOptions(
                embeddingModel = config.treeComm.embeddingModel,
                structWeight = config.treeComm.structWeight,
                enableFastMode = config.treeComm.enableFastMode,
                maxTotalCommunities = maxCommunities,
                enableSummary = config.treeComm.enableSummary,
                mergeThreshold = config.treeComm.mergeThreshold,
                maxIterations = config.treeComm.maxIterations,
                summaryMaxWords = config.treeComm.summaryMaxWords,
            )
        val communities =
            FastTreeComm(llmClient)
                .detectCommunities(
                    relationships = entityRelationships,
                    options = treeCommOptions,
                )

        communities.forEachIndexed { index, community ->
            val members =
                (community["nodes"] as? List<*>)
                    .orEmpty()
                    .mapNotNull { value -> value?.toString()?.trim()?.takeIf { it.isNotBlank() } }
                    .distinct()
            if (members.size < 2) {
                return@forEachIndexed
            }

            val communityId = community["community_id"]?.toString().orEmpty().ifBlank { "community_${index + 1}" }
            val communityNode =
                GraphNode(
                    label = "community",
                    properties =
                        mapOf(
                            "name" to community["name"]?.toString().orEmpty().ifBlank { "Community_${index + 1}" },
                            "description" to
                                community["summary"]?.toString().orEmpty().ifBlank {
                                    "Community of ${members.size} members"
                                },
                            "community_id" to communityId,
                            "node_count" to members.size.toString(),
                            "edge_count" to community["edge_count"]?.toString().orEmpty(),
                            "members" to members,
                        ),
                )

            members.forEach { memberName ->
                val memberNode =
                    entityByName[memberName]
                        ?: GraphNode(
                            label = "entity",
                            properties = mapOf("name" to memberName),
                        )
                relationships.add(
                    GraphRelationship(
                        startNode = memberNode,
                        relation = "member_of",
                        endNode = communityNode,
                    ),
                )
            }

            val keywords =
                (community["keywords"] as? List<*>)
                    .orEmpty()
                    .mapNotNull { value -> value?.toString()?.trim()?.takeIf { it.isNotBlank() } }
                    .distinct()

            keywords.forEach { keywordName ->
                val keywordEntity = entityByName[keywordName] ?: return@forEach
                val keywordNode =
                    GraphNode(
                        label = "keyword",
                        properties = mapOf("name" to keywordName),
                    )
                relationships.add(
                    GraphRelationship(
                        startNode = keywordEntity,
                        relation = "represented_by",
                        endNode = keywordNode,
                    ),
                )
                relationships.add(
                    GraphRelationship(
                        startNode = keywordEntity,
                        relation = "kw_filter_by",
                        endNode = keywordNode,
                    ),
                )
                relationships.add(
                    GraphRelationship(
                        startNode = keywordNode,
                        relation = "keyword_of",
                        endNode = communityNode,
                    ),
                )
            }
        }
    }

    private fun extractChunkKnowledge(chunkText: String): ExtractionPayload {
        if (llmExtractionDisabled) {
            return emptyExtractionPayload()
        }

        val prompt = renderConstructionPrompt(chunkText)
        val llmResponse = llmClient.complete(prompt)
        if (llmResponse.isBlank()) {
            llmExtractionDisabled = true
            logger.warn { "Disabling constructor LLM extraction for '$datasetName' after empty completion response." }
            return emptyExtractionPayload()
        }
        val parsed = LlmOutputParser.parseJsonObject(llmResponse)
        if (parsed == null) {
            return emptyExtractionPayload()
        }

        return ExtractionPayload(
            attributes = normalizeAttributes(parsed["attributes"]),
            triples = normalizeTriples(parsed["triples"]),
            entityTypes = normalizeEntityTypes(parsed["entity_types"]),
            newSchemaTypes = normalizeSchemaTypes(parsed["new_schema_types"]),
        )
    }

    private fun renderConstructionPrompt(chunkText: String): String {
        val schemaJson =
            runCatching { mapper.writeValueAsString(schema) }
                .getOrElse { "{}" }
        val promptCandidates = constructionPromptCandidates()

        promptCandidates.forEach { promptType ->
            val template = config.prompts["construction"]?.get(promptType) ?: return@forEach
            val rendered =
                runCatching {
                    config.getPromptFormatted(
                        category = "construction",
                        promptType = promptType,
                        variables =
                            mapOf(
                                "schema" to schemaJson,
                                "chunk" to chunkText,
                            ),
                    )
                }.getOrElse {
                    template
                        .replace("{schema}", schemaJson)
                        .replace("{chunk}", chunkText)
                }
            return rendered
        }

        return inlineFallbackPrompt(schemaJson = schemaJson, chunkText = chunkText)
    }

    private fun constructionPromptCandidates(): List<String> {
        val basePromptType =
            when (datasetName.lowercase()) {
                "anony_chs",
                "novel",
                "novel_chs",
                -> "novel"

                "anony_eng",
                "novel_eng",
                -> "novel_eng"

                else -> "general"
            }
        val requestedPromptType = if (mode == "agent") "${basePromptType}_agent" else basePromptType
        val candidates = linkedSetOf<String>()
        candidates.add(requestedPromptType)
        if (requestedPromptType.endsWith("_agent")) {
            candidates.add(requestedPromptType.removeSuffix("_agent"))
            candidates.add("general_agent")
        }
        candidates.add("general")
        return candidates.toList()
    }

    private fun inlineFallbackPrompt(
        schemaJson: String,
        chunkText: String,
    ): String =
        """
        You are an expert information extractor and structured data organizer.
        Extract entities, attributes, relations, and entity types from the chunk.
        Return a JSON object with keys: attributes, triples, entity_types, and optional new_schema_types.
        Schema:
        $schemaJson
        Chunk:
        $chunkText
        """.trimIndent()

    private fun addExtractionRelationships(
        chunk: ChunkRecord,
        extraction: ExtractionPayload,
        relationships: MutableList<GraphRelationship>,
        entityByName: MutableMap<String, GraphNode>,
        seenEntityTriples: MutableSet<EntityTripleKey>,
    ): Int {
        var addedCount = 0

        extraction.attributes.forEach { (entityName, attributes) ->
            if (entityName.isBlank()) {
                return@forEach
            }
            val entityNode =
                findOrCreateEntityNode(
                    entityName = entityName,
                    chunkId = chunk.id,
                    entityType = extraction.entityTypes[entityName],
                    entityByName = entityByName,
                )
            attributes.forEach { attribute ->
                if (attribute.isBlank()) {
                    return@forEach
                }
                val attributeNode =
                    GraphNode(
                        label = "attribute",
                        properties =
                            mapOf(
                                "name" to attribute,
                                "chunk id" to chunk.id,
                            ),
                    )
                relationships.add(
                    GraphRelationship(
                        startNode = entityNode,
                        relation = "has_attribute",
                        endNode = attributeNode,
                    ),
                )
                addedCount += 1
            }
        }

        extraction.triples.forEach { triple ->
            val sourceNode =
                findOrCreateEntityNode(
                    entityName = triple.subject,
                    chunkId = chunk.id,
                    entityType = extraction.entityTypes[triple.subject],
                    entityByName = entityByName,
                )
            val targetNode =
                findOrCreateEntityNode(
                    entityName = triple.target,
                    chunkId = chunk.id,
                    entityType = extraction.entityTypes[triple.target],
                    entityByName = entityByName,
                )

            val dedupKey =
                EntityTripleKey(
                    source = readPropertyAsString(sourceNode, "name"),
                    relation = triple.relation,
                    target = readPropertyAsString(targetNode, "name"),
                )
            if (!seenEntityTriples.add(dedupKey)) {
                return@forEach
            }

            relationships.add(
                GraphRelationship(
                    startNode = sourceNode,
                    relation = triple.relation,
                    endNode = targetNode,
                ),
            )
            addedCount += 1
        }

        return addedCount
    }

    private fun findOrCreateEntityNode(
        entityName: String,
        chunkId: String,
        entityType: String?,
        entityByName: MutableMap<String, GraphNode>,
    ): GraphNode {
        val normalizedName = entityName.trim()
        entityByName[normalizedName]?.let { return it }

        val properties = linkedMapOf<String, Any?>()
        properties["name"] = normalizedName
        properties["chunk id"] = chunkId
        if (!entityType.isNullOrBlank()) {
            properties["schema_type"] = entityType.trim()
        }

        val entityNode =
            GraphNode(
                label = "entity",
                properties = properties,
            )
        entityByName[normalizedName] = entityNode
        return entityNode
    }

    private fun readPropertyAsString(
        node: GraphNode,
        key: String,
    ): String = node.properties[key]?.toString().orEmpty()

    private fun normalizeAttributes(value: Any?): Map<String, List<String>> {
        if (value !is Map<*, *>) {
            return emptyMap()
        }

        return value.entries
            .mapNotNull { (key, rawAttributes) ->
                val entity = key?.toString()?.trim().orEmpty()
                if (entity.isBlank()) {
                    null
                } else {
                    entity to toStringList(rawAttributes)
                }
            }.toMap()
    }

    private fun normalizeTriples(value: Any?): List<ParsedTriple> {
        if (value !is List<*>) {
            return emptyList()
        }

        return value.mapNotNull { entry -> parseTriple(entry) }
    }

    private fun parseTriple(value: Any?): ParsedTriple? {
        val parsed =
            when (value) {
                is List<*> -> {
                    if (value.size < 3) {
                        return null
                    }
                    ParsedTriple(
                        subject = value[0]?.toString()?.trim().orEmpty(),
                        relation = value[1]?.toString()?.trim().orEmpty(),
                        target = value[2]?.toString()?.trim().orEmpty(),
                    )
                }

                is Map<*, *> -> {
                    val subject =
                        value["subject"]
                            ?: value["source"]
                            ?: value["head"]
                            ?: value["entity_mention1"]
                    val relation =
                        value["relation"]
                            ?: value["predicate"]
                    val target =
                        value["object"]
                            ?: value["target"]
                            ?: value["tail"]
                            ?: value["entity_mention2"]

                    ParsedTriple(
                        subject = subject?.toString()?.trim().orEmpty(),
                        relation = relation?.toString()?.trim().orEmpty(),
                        target = target?.toString()?.trim().orEmpty(),
                    )
                }

                is String -> {
                    parseTriple(LlmOutputParser.parseJsonValue(value))
                }

                else -> {
                    null
                }
            } ?: return null

        return if (parsed.subject.isBlank() || parsed.relation.isBlank() || parsed.target.isBlank()) {
            null
        } else {
            parsed
        }
    }

    private fun normalizeEntityTypes(value: Any?): Map<String, String> {
        if (value !is Map<*, *>) {
            return emptyMap()
        }

        return value.entries
            .mapNotNull { (key, typeValue) ->
                val entity = key?.toString()?.trim().orEmpty()
                val schemaType = typeValue?.toString()?.trim().orEmpty()
                if (entity.isBlank() || schemaType.isBlank()) {
                    null
                } else {
                    entity to schemaType
                }
            }.toMap()
    }

    private fun normalizeSchemaTypes(value: Any?): Map<String, List<String>> {
        if (value !is Map<*, *>) {
            return emptyMap()
        }

        val normalized =
            value.entries.associate { (key, rawValue) ->
                key.toString().lowercase() to toStringList(rawValue)
            }

        return mapOf(
            "nodes" to normalized["nodes"].orEmpty(),
            "relations" to normalized["relations"].orEmpty(),
            "attributes" to normalized["attributes"].orEmpty(),
        ).filterValues { it.isNotEmpty() }
    }

    private fun toStringList(value: Any?): List<String> =
        when (value) {
            is List<*> -> {
                value
                    .mapNotNull { entry -> entry?.toString()?.trim() }
                    .filter { entry -> entry.isNotBlank() }
            }

            is String -> {
                listOf(value)
                    .map { entry -> entry.trim() }
                    .filter { entry -> entry.isNotBlank() }
            }

            else -> {
                emptyList()
            }
        }

    private fun updateSchemaWithNewTypes(newSchemaTypes: Map<String, List<String>>) {
        if (newSchemaTypes.isEmpty()) {
            return
        }

        var updated = false
        updated =
            mergeSchemaField(
                current = schema,
                canonicalKey = "Nodes",
                aliases = listOf("Nodes", "nodes"),
                additions = newSchemaTypes["nodes"].orEmpty(),
            ) || updated
        updated =
            mergeSchemaField(
                current = schema,
                canonicalKey = "Relations",
                aliases = listOf("Relations", "relations"),
                additions = newSchemaTypes["relations"].orEmpty(),
            ) || updated
        updated =
            mergeSchemaField(
                current = schema,
                canonicalKey = "Attributes",
                aliases = listOf("Attributes", "attributes"),
                additions = newSchemaTypes["attributes"].orEmpty(),
            ) || updated

        if (!updated) {
            return
        }

        runCatching {
            schemaFile.parent?.createDirectories()
            mapper.writerWithDefaultPrettyPrinter().writeValue(schemaFile.toFile(), schema)
        }.onFailure { error ->
            logger.warn(error) { "Failed to persist evolved schema to '$schemaFile'" }
        }
    }

    private fun mergeSchemaField(
        current: MutableMap<String, Any?>,
        canonicalKey: String,
        aliases: List<String>,
        additions: List<String>,
    ): Boolean {
        if (additions.isEmpty()) {
            return false
        }

        val targetKey = aliases.firstOrNull { key -> current.containsKey(key) } ?: canonicalKey
        val existingValues =
            (current[targetKey] as? List<*>)
                ?.mapNotNull { value -> value?.toString()?.trim() }
                ?.filter { value -> value.isNotBlank() }
                ?.toMutableList()
                ?: mutableListOf()

        var updated = false
        additions.forEach { candidate ->
            val normalized = candidate.trim()
            if (normalized.isBlank()) {
                return@forEach
            }
            if (normalized !in existingValues) {
                existingValues.add(normalized)
                updated = true
            }
        }

        if (updated) {
            current[targetKey] = existingValues
        }
        return updated
    }

    private fun emptyExtractionPayload(): ExtractionPayload =
        ExtractionPayload(
            attributes = emptyMap(),
            triples = emptyList(),
            entityTypes = emptyMap(),
            newSchemaTypes = emptyMap(),
        )

    private fun writeChunksFile(chunks: List<ChunkRecord>) {
        val chunkFile = resolveOutputPath(config.output.chunksDir).resolve("$datasetName.txt")
        chunkFile.parent?.createDirectories()

        val content =
            buildString {
                chunks.forEach { chunk ->
                    append("id: ")
                    append(chunk.id)
                    append('\t')
                    append("Chunk: ")
                    append(
                        chunk.text
                            .replace("\n", "\\n")
                            .replace("\t", "\\t"),
                    )
                    append('\n')
                }
            }

        Files.writeString(
            chunkFile,
            content,
            StandardOpenOption.CREATE,
            StandardOpenOption.TRUNCATE_EXISTING,
            StandardOpenOption.WRITE,
        )
    }

    private fun writeGraphFile(relationships: List<GraphRelationship>) {
        val graphFile = resolveOutputPath(config.output.graphsDir).resolve("${datasetName}_new.json")
        graphFile.parent?.createDirectories()
        mapper.writerWithDefaultPrettyPrinter().writeValue(graphFile.toFile(), relationships)
    }

    private fun resolveOutputPath(configPath: String): Path {
        val configured = Path.of(configPath)
        return if (configured.isAbsolute) configured else rootDir.resolve(configured)
    }

    private fun resolvePath(path: String): Path {
        val candidate = Path.of(path)
        return if (candidate.isAbsolute) candidate else rootDir.resolve(candidate)
    }

    companion object {
        private val TOKENIZER_REGISTRY = Encodings.newLazyEncodingRegistry()
    }
}
