package com.youtu.graphrag.shared.retriever

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.graph.GraphRelationship
import com.youtu.graphrag.shared.retriever.nlp.QueryNlp
import com.youtu.graphrag.shared.retriever.nlp.QueryNlpFactory
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withTimeoutOrNull
import java.nio.file.Path
import java.util.Locale
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.readLines

private data class TripleRecord(
    val source: String,
    val relation: String,
    val target: String,
    val sourceLabel: String,
    val targetLabel: String,
    val sourceProperties: Map<String, String>,
    val targetProperties: Map<String, String>,
) {
    val serialized: String
        get() = "[\"${escape(source)}\", \"${escape(relation)}\", \"${escape(target)}\"]"

    val contextFormat: String
        get() = "(${source}${formatProps(sourceProperties)}, $relation, ${target}${formatProps(targetProperties)})"

    private fun formatProps(props: Map<String, String>): String {
        val filtered = props.filterKeys { it != "name" && it != "chunk id" }
        if (filtered.isEmpty()) return ""
        return " " + filtered.toString()
    }

    val searchableText: String
        get() =
            buildString {
                append(source)
                append(' ')
                append(relation)
                append(' ')
                append(target)
                append(' ')
                append(sourceLabel)
                append(' ')
                append(targetLabel)
                append(' ')
                append(sourceProperties.values.joinToString(" "))
                append(' ')
                append(targetProperties.values.joinToString(" "))
            }

    val chunkIds: Set<String>
        get() =
            setOfNotNull(
                sourceProperties["chunk id"]?.takeIf { it.isNotBlank() },
                targetProperties["chunk id"]?.takeIf { it.isNotBlank() },
            )

    private fun escape(value: String): String = value.replace("\"", "\\\"")
}

private data class RetrievalChunk(
    val id: String,
    val text: String,
)

enum class IrcotPromptSource {
    BACKEND,
    MAIN,
}

private data class ChunkRanking(
    val chunkId: String,
    val score: Double,
)

private data class ScoredTriple(
    val record: TripleRecord,
    val score: Double,
    val ordinal: Int,
)

private data class StrategyScore<T>(
    val strategy: String,
    val scores: Map<T, Double>,
)

private data class FusedScore(
    val score: Double,
    val strategies: Set<String>,
    val dominantStrategy: String,
)

private data class EmbeddingCacheFile(
    val modelTag: String,
    val dimensions: Int,
    val vectors: Map<String, List<Float>> = emptyMap(),
)

class KTRetriever private constructor(
    private val datasetName: String,
    private val graphPath: String,
    private val recallPaths: Int,
    private val schemaPath: String,
    private val topK: Int,
    private val mode: String,
    private val config: ConfigManager,
    private val rootDir: Path,
    private val tripleRecords: List<TripleRecord>,
    private val adjacencyByNode: Map<String, List<TripleRecord>>,
    private val tripleVectorIndex: SemanticVectorIndex<TripleRecord>?,
    private val tripleOrdinalByRecord: Map<TripleRecord, Int>,
    private val chunkRecords: List<RetrievalChunk>,
    private val chunkVectorIndex: SemanticVectorIndex<RetrievalChunk>?,
    private val chunkById: Map<String, String>,
    private val vectorStrategyTag: String,
    private val queryNlp: QueryNlp,
) {
    private val logger = KotlinLogging.logger {}
    private val mapper = ObjectMapper().registerKotlinModule()

    companion object {
        private val logger = KotlinLogging.logger {}
        private val mapper = ObjectMapper().registerKotlinModule()
        private val EXCLUDED_OUTPUT_RELATIONS = setOf("represented_by", "kw_filter_by")
        private val COMMUNITY_RELATIONS = setOf("member_of", "keyword_of", "represented_by", "kw_filter_by")

        fun createAndBuild(
            datasetName: String,
            graphPath: String,
            recallPaths: Int,
            schemaPath: String,
            topK: Int,
            mode: String,
            config: ConfigManager,
            rootDir: Path = Path.of("."),
        ): KTRetriever {
            require(datasetName.isNotBlank() && graphPath.isNotBlank() && schemaPath.isNotBlank()) {
                "datasetName, graphPath, and schemaPath must be non-blank"
            }
            require(topK >= 0) { "topK must be >= 0, but was $topK" }

            config.createOutputDirectories()

            val embedder = RetrieverEmbedderFactory.fromConfig(config = config)
            val builder = IndexBuilder(datasetName, config, rootDir, embedder)
            val queryNlp = QueryNlpFactory.create(config = config, rootDir = rootDir)

            val triples = builder.loadTriples(builder.resolvePath(graphPath))
            val adjacency = builder.buildAdjacency(triples)
            val tripleOrdinal = triples.withIndex().associate { (index, record) -> record to index }
            val chunks = builder.loadChunks(builder.resolvePath("${config.output.chunksDir}/$datasetName.txt"))
            val chunkRecords = chunks.entries.map { (chunkId, text) -> RetrievalChunk(id = chunkId, text = text) }

            val tripleVectors = builder.prepareTripleVectors(triples, embedder)
            val chunkVectors = builder.prepareChunkVectors(chunkRecords, embedder)

            val tripleIndex =
                builder.buildSemanticIndex(
                    items = triples,
                    textSelector = { record -> record.searchableText },
                    vectorSelector = { record -> tripleVectors[record] ?: embedder.embed(record.searchableText) },
                    embedder = embedder,
                )
            val chunkIndex =
                builder.buildSemanticIndex(
                    items = chunkRecords,
                    textSelector = { chunk -> chunk.text },
                    vectorSelector = { chunk -> chunkVectors[chunk] ?: embedder.embed(chunk.text) },
                    embedder = embedder,
                )
            val strategyTag = tripleIndex.strategyTag

            logger.info {
                "Built retrieval indices for '$datasetName' with ${triples.size} triples and ${chunks.size} chunks"
            }

            return KTRetriever(
                datasetName = datasetName,
                graphPath = graphPath,
                recallPaths = recallPaths,
                schemaPath = schemaPath,
                topK = topK,
                mode = mode,
                config = config,
                rootDir = rootDir,
                tripleRecords = triples,
                adjacencyByNode = adjacency,
                tripleVectorIndex = tripleIndex,
                tripleOrdinalByRecord = tripleOrdinal,
                chunkRecords = chunkRecords,
                chunkVectorIndex = chunkIndex,
                chunkById = chunks,
                vectorStrategyTag = strategyTag,
                queryNlp = queryNlp,
            )
        }
    }

    fun processRetrievalResults(
        question: String,
        involvedTypes: Map<String, List<String>> = emptyMap(),
    ): Pair<Map<String, Any>, Double> {
        val startNs = System.nanoTime()
        val normalizedTopK = topK.coerceAtLeast(1)
        val vectorSearchK =
            config.retrieval.faiss.searchK
                .coerceAtLeast(normalizedTopK)
        val vectorEnabled = config.retrieval.enableReranking
        val nlpAnalysis = queryNlp.analyze(question)
        val enhancedQuestion =
            if (config.retrieval.enableQueryEnhancement) {
                nlpAnalysis.enhancedQuestion(question)
            } else {
                question
            }
        val strategyConfig = config.retrieval.strategy
        val enabledStrategies = strategyConfig.enabled.map { it.lowercase(Locale.ROOT) }.toSet()
        val normalizedQuestionKeywords =
            nlpAnalysis
                .normalizedKeywordSet()
                .ifEmpty { tokenize(question) }
        val typeFilteredTriples = filterTriplesByInvolvedTypes(tripleRecords, involvedTypes)
        val retrievalPool = typeFilteredTriples.ifEmpty { tripleRecords }

        val seedTriples =
            rankedItems(
                items = retrievalPool,
                questionKeywords = normalizedQuestionKeywords,
                topLimit = minOf(vectorSearchK, retrievalPool.size.coerceAtLeast(1)),
                textSelector = { record -> record.searchableText },
            )
        val expandedTriples = expandTriplesFromSeeds(seedTriples, retrievalPool).ifEmpty { seedTriples }
        val tripleStrategies =
            executeStrategies(
                strategyDefinitions =
                    listOf(
                        "lexical_triple" to {
                            retrievalPool.associateWith { record ->
                                keywordMatchCount(record.searchableText, normalizedQuestionKeywords).toDouble()
                            }
                        },
                        "semantic_triple" to {
                            if (!vectorEnabled) {
                                emptyMap()
                            } else {
                                vectorSearchScoresForTriples(
                                    question = enhancedQuestion,
                                    searchLimit = vectorSearchK,
                                    allowedTriples = retrievalPool,
                                )
                            }
                        },
                        "path_expand" to {
                            if (!config.retrieval.enableHighRecall) {
                                emptyMap()
                            } else {
                                val seedSet = seedTriples.toSet()
                                expandedTriples.associateWith { triple ->
                                    if (triple in seedSet) 1.0 else 0.6
                                }
                            }
                        },
                        "community_triple" to {
                            retrievalPool
                                .filter { record ->
                                    record.relation.lowercase(Locale.ROOT) in COMMUNITY_RELATIONS
                                }.associateWith { record ->
                                    0.4 + keywordMatchCount(record.searchableText, normalizedQuestionKeywords).toDouble()
                                }
                        },
                    ),
                enabledStrategies = enabledStrategies,
                enableParallel = strategyConfig.enableParallel,
                timeoutMs = strategyConfig.timeoutMs,
                maxConcurrency = strategyConfig.maxConcurrency,
            )
        val fusedTripleScores = fuseScores(tripleStrategies, strategyConfig.weights)
        val selectedTripleRankings =
            fusedTripleScores.entries
                .map { (record, fused) ->
                    ScoredTriple(
                        record = record,
                        score = fused.score,
                        ordinal = tripleOrdinalByRecord[record] ?: Int.MAX_VALUE,
                    )
                }.sortedWith(
                    compareByDescending<ScoredTriple> { value -> value.score }
                        .thenBy { value -> value.ordinal },
                ).take(normalizedTopK)
        val selectedTripleRaw =
            selectedTripleRankings
                .map { ranking -> ranking.record }
                .filterNot { record -> record.relation.lowercase(Locale.ROOT) in EXCLUDED_OUTPUT_RELATIONS }
        val selectedTripleFormatted =
            selectedTripleRaw.map { record ->
                val score = selectedTripleRankings.firstOrNull { ranking -> ranking.record == record }?.score ?: 0.0
                formatTripleWithScore(record, score)
            }

        val chunkIdsFromTriples =
            linkedSetOf<String>().apply {
                selectedTripleRankings.forEach { ranking ->
                    val record = ranking.record
                    record.chunkIds.forEach { chunkId ->
                        if (chunkId in chunkById) {
                            add(chunkId)
                        }
                    }
                }
            }

        val chunkVectorRankings =
            run {
                val bridgeChunkScores =
                    selectedTripleRankings
                        .flatMap { ranking ->
                            ranking.record.chunkIds.map { chunkId -> chunkId to ranking.score }
                        }.groupBy(keySelector = { pair -> pair.first }, valueTransform = { pair -> pair.second })
                        .mapValues { (_, values) -> values.sum() }
                val chunkStrategies =
                    executeStrategies(
                        strategyDefinitions =
                            listOf(
                                "lexical_chunk" to {
                                    chunkRecords.associate { chunk ->
                                        chunk.id to keywordMatchCount(chunk.text, normalizedQuestionKeywords).toDouble()
                                    }
                                },
                                "semantic_chunk" to {
                                    if (!vectorEnabled) {
                                        emptyMap()
                                    } else {
                                        chunkVectorIndex
                                            ?.search(enhancedQuestion, vectorSearchK)
                                            .orEmpty()
                                            .associate { hit -> hit.item.id to hit.score }
                                    }
                                },
                                "triple_chunk_bridge" to {
                                    bridgeChunkScores
                                },
                            ),
                        enabledStrategies = enabledStrategies,
                        enableParallel = strategyConfig.enableParallel,
                        timeoutMs = strategyConfig.timeoutMs,
                        maxConcurrency = strategyConfig.maxConcurrency,
                    )

                val fusedChunkScores = fuseScores(chunkStrategies, strategyConfig.weights)
                if (fusedChunkScores.isEmpty()) {
                    rankedItems(
                        items = chunkRecords,
                        questionKeywords = normalizedQuestionKeywords,
                        topLimit = normalizedTopK,
                        textSelector = { chunk -> chunk.text },
                    ).map { chunk ->
                        ChunkRanking(
                            chunkId = chunk.id,
                            score = keywordMatchCount(chunk.text, normalizedQuestionKeywords).toDouble(),
                        )
                    }
                } else {
                    fusedChunkScores.entries
                        .filter { (chunkId, _) -> chunkId in chunkById }
                        .sortedWith(
                            compareByDescending<Map.Entry<String, FusedScore>> { entry -> entry.value.score }
                                .thenBy { entry -> entry.key },
                        ).take(normalizedTopK)
                        .map { entry -> ChunkRanking(chunkId = entry.key, score = entry.value.score) }
                }
            }

        val chunkRetrievalIds =
            linkedSetOf<String>().apply {
                chunkVectorRankings.forEach { ranking ->
                    if (ranking.chunkId in chunkById) {
                        add(ranking.chunkId)
                    }
                }
            }

        val allChunkIds =
            linkedSetOf<String>().apply {
                addAll(chunkRetrievalIds)
                addAll(chunkIdsFromTriples)
            }
        val chunkIdList = allChunkIds.toList()
        val chunkContents =
            chunkIdList.map { chunkId ->
                chunkById[chunkId].orEmpty()
            }
        val chunkRetrievalResults =
            chunkVectorRankings.map { ranking ->
                val content = chunkById[ranking.chunkId].orEmpty()
                val preview = content.take(200) + if (content.length > 200) "..." else ""
                val formattedScore = String.format(Locale.US, "%.3f", ranking.score)
                "[Chunk ${ranking.chunkId}] $preview [score: $formattedScore]"
            }

        val retrievalTime = (System.nanoTime() - startNs).toDouble() / 1_000_000_000.0
        logger.info { "retrieval time: ${String.format(Locale.US, "%.4f", retrievalTime)}" }

        val retrievalResults =
            mapOf(
                "triples" to selectedTripleFormatted,
                "chunk_ids" to chunkIdList,
                "chunk_contents" to chunkContents,
                "chunk_retrieval_results" to chunkRetrievalResults,
            )

        return retrievalResults to retrievalTime
    }

    fun enhanceQuery(question: String): String = queryNlp.analyze(question).enhancedQuestion(question)

    fun extractQueryEntities(question: String): List<String> = queryNlp.analyze(question).entities

    fun extractQueryKeywords(question: String): List<String> = queryNlp.analyze(question).normalizedKeywordSet().toList()

    fun buildKnowledgeContext(
        triples: List<String>,
        chunks: List<String>,
        tripleLimit: Int = 20,
        chunkLimit: Int = 10,
    ): String {
        val tripleSection = triples.take(tripleLimit).joinToString("\n")
        val chunkSection = chunks.take(chunkLimit).joinToString("\n")
        return buildString {
            append("=== Triples ===\n")
            append(tripleSection)
            append("\n=== Chunks ===\n")
            append(chunkSection)
        }
    }

    fun generatePrompt(
        question: String,
        context: String,
    ): String {
        retrievalPromptCandidates().forEach { promptType ->
            val template = config.prompts["retrieval"]?.get(promptType) ?: return@forEach
            val rendered =
                runCatching {
                    config.getPromptFormatted(
                        category = "retrieval",
                        promptType = promptType,
                        variables =
                            mapOf(
                                "question" to question,
                                "context" to context,
                            ),
                    )
                }.getOrElse {
                    template
                        .replace("{question}", question)
                        .replace("{context}", context)
                }
            return rendered
        }

        return """
            Question: $question

            Knowledge Context:
            $context

            Answer (be specific and direct):
            """.trimIndent()
    }

    private fun retrievalPromptCandidates(): List<String> =
        when (datasetName.lowercase(Locale.ROOT)) {
            "anony_chs",
            "novel",
            "novel_chs",
            -> listOf("novel_chs", "novel", "general")

            "anony_eng",
            "novel_eng",
            -> listOf("novel_eng", "general")

            else -> listOf("general")
        }

    fun generateIrcotPrompt(
        currentQuery: String,
        context: String,
        previousThoughts: String,
        step: Int,
        originalQuestion: String = currentQuery,
        promptSource: IrcotPromptSource = IrcotPromptSource.BACKEND,
    ): String {
        val promptCandidates =
            when (promptSource) {
                IrcotPromptSource.BACKEND -> listOf("ircot_backend", "ircot")
                IrcotPromptSource.MAIN -> listOf("ircot_main")
            }
        val variables =
            mapOf(
                "current_query" to currentQuery,
                "current_iteration_query" to currentQuery,
                "original_question" to originalQuestion,
                "context" to context,
                "previous_thoughts" to previousThoughts,
                "step" to step,
            )

        promptCandidates.forEach { promptType ->
            val template = config.prompts["retrieval"]?.get(promptType) ?: return@forEach
            val rendered =
                runCatching {
                    config.getPromptFormatted(
                        category = "retrieval",
                        promptType = promptType,
                        variables = variables,
                    )
                }.getOrElse {
                    template
                        .replace("{current_query}", currentQuery)
                        .replace("{current_iteration_query}", currentQuery)
                        .replace("{original_question}", originalQuestion)
                        .replace("{context}", context)
                        .replace("{previous_thoughts}", previousThoughts)
                        .replace("{step}", step.toString())
                }
            return rendered
        }

        return inlineFallbackIrcotPrompt(
            currentQuery = currentQuery,
            originalQuestion = originalQuestion,
            context = context,
            previousThoughts = previousThoughts,
            step = step,
            promptSource = promptSource,
        )
    }

    private fun inlineFallbackIrcotPrompt(
        currentQuery: String,
        originalQuestion: String,
        context: String,
        previousThoughts: String,
        step: Int,
        promptSource: IrcotPromptSource,
    ): String =
        when (promptSource) {
            IrcotPromptSource.MAIN -> {
                """
                You are an expert knowledge assistant using iterative retrieval with chain-of-thought reasoning.

                Current Question: $currentQuery

                Available Knowledge Context:
                $context

                Previous Thoughts: $previousThoughts

                Step $step: Please think step by step about what additional information you need to answer the question completely and accurately.

                Instructions:
                1. Analyze the current knowledge context and the question
                2. Consider the initial analysis from noagent mode (if available in previous thoughts)
                3. Think about what information might be missing or unclear
                4. If you have enough information to answer, in the end of your response, write "So the answer is:" followed by your final answer
                5. If you need more information, in the end of your response, write a specific query begin with "The new query is:" to retrieve additional relevant information
                6. Be specific and focused in your reasoning
                7. Build upon the initial analysis to provide deeper insights

                Your reasoning:
                """.trimIndent()
            }

            IrcotPromptSource.BACKEND -> {
                """
                You are an expert knowledge assistant using iterative retrieval with chain-of-thought reasoning.
                Current Question: $originalQuestion
                Current Iteration Query: $currentQuery
                Knowledge Context:
                $context
                Previous Thoughts: $previousThoughts
                Instructions:
                1. If enough info answer with: So the answer is: <answer>
                2. Else propose new query with: The new query is: <query>
                Your reasoning:
                """.trimIndent()
            }
        }

    private fun <T> executeStrategies(
        strategyDefinitions: List<Pair<String, () -> Map<T, Double>>>,
        enabledStrategies: Set<String>,
        enableParallel: Boolean,
        timeoutMs: Long,
        maxConcurrency: Int,
    ): List<StrategyScore<T>> {
        val active =
            strategyDefinitions.filter { (strategyName, _) ->
                strategyName.lowercase(Locale.ROOT) in enabledStrategies
            }
        if (active.isEmpty()) {
            return emptyList()
        }

        return if (!enableParallel || active.size == 1) {
            active.map { (strategyName, strategyFn) ->
                val scores =
                    runCatching { strategyFn() }
                        .getOrElse { error ->
                            logger.warn(error) { "Retriever strategy '$strategyName' failed in sequential mode." }
                            emptyMap()
                        }
                StrategyScore(strategy = strategyName, scores = scores)
            }
        } else {
            runBlocking {
                val semaphore = Semaphore(maxConcurrency.coerceAtLeast(1))
                active
                    .map { (strategyName, strategyFn) ->
                        async(Dispatchers.IO) {
                            semaphore.withPermit {
                                val scores =
                                    withTimeoutOrNull(timeoutMs) {
                                        runCatching { strategyFn() }
                                            .getOrElse { error ->
                                                logger.warn(error) { "Retriever strategy '$strategyName' failed." }
                                                emptyMap()
                                            }
                                    } ?: run {
                                        logger.warn {
                                            "Retriever strategy '$strategyName' timed out after ${timeoutMs}ms."
                                        }
                                        emptyMap()
                                    }
                                StrategyScore(strategy = strategyName, scores = scores)
                            }
                        }
                    }.awaitAll()
            }
        }
    }

    private fun <T> fuseScores(
        strategyScores: List<StrategyScore<T>>,
        weights: Map<String, Double>,
    ): Map<T, FusedScore> {
        if (strategyScores.isEmpty()) {
            return emptyMap()
        }

        val fused = linkedMapOf<T, Double>()
        val contributingStrategies = linkedMapOf<T, MutableSet<String>>()
        val dominantContributions = linkedMapOf<T, Pair<String, Double>>()

        strategyScores.forEach { strategyScore ->
            if (strategyScore.scores.isEmpty()) {
                return@forEach
            }
            val normalized = normalizeScores(strategyScore.scores)
            val weight = weights[strategyScore.strategy] ?: 1.0
            if (weight <= 0.0) {
                return@forEach
            }
            normalized.forEach { (item, score) ->
                if (score <= 0.0) {
                    return@forEach
                }
                val weighted = score * weight
                fused[item] = (fused[item] ?: 0.0) + weighted
                contributingStrategies.getOrPut(item) { linkedSetOf() }.add(strategyScore.strategy)
                val previous = dominantContributions[item]
                if (previous == null || weighted > previous.second ||
                    (weighted == previous.second && strategyScore.strategy < previous.first)
                ) {
                    dominantContributions[item] = strategyScore.strategy to weighted
                }
            }
        }

        return fused.mapValues { (item, score) ->
            FusedScore(
                score = score,
                strategies = contributingStrategies[item].orEmpty(),
                dominantStrategy = dominantContributions[item]?.first.orEmpty(),
            )
        }
    }

    private fun <T> normalizeScores(scores: Map<T, Double>): Map<T, Double> {
        if (scores.isEmpty()) {
            return emptyMap()
        }
        val positives = scores.filterValues { value -> value.isFinite() && value > 0.0 }
        if (positives.isEmpty()) {
            return emptyMap()
        }
        val maxScore = positives.values.maxOrNull() ?: return emptyMap()
        if (maxScore <= 0.0) {
            return emptyMap()
        }
        return positives.mapValues { (_, value) -> value / maxScore }
    }

    private fun filterTriplesByInvolvedTypes(
        records: List<TripleRecord>,
        involvedTypes: Map<String, List<String>>,
    ): List<TripleRecord> {
        val nodeFilters = normalizeFilters(involvedTypes["nodes"])
        val relationFilters = normalizeFilters(involvedTypes["relations"])
        val attributeFilters = normalizeFilters(involvedTypes["attributes"])
        if (nodeFilters.isEmpty() && relationFilters.isEmpty() && attributeFilters.isEmpty()) {
            return records
        }

        return records.filter { record ->
            val nodeMatch =
                nodeFilters.isNotEmpty() && matchesNodeFilter(record, nodeFilters)
            val relationMatch =
                relationFilters.isNotEmpty() &&
                    relationFilters.any { filter -> record.relation.lowercase().contains(filter) }
            val attributeMatch =
                attributeFilters.isNotEmpty() && matchesAttributeFilter(record, attributeFilters)

            nodeMatch || relationMatch || attributeMatch
        }
    }

    private fun matchesNodeFilter(
        record: TripleRecord,
        nodeFilters: Set<String>,
    ): Boolean {
        val candidates =
            listOf(
                record.sourceLabel,
                record.targetLabel,
                record.source,
                record.target,
                record.sourceProperties["schema_type"].orEmpty(),
                record.targetProperties["schema_type"].orEmpty(),
            ).map { value -> value.lowercase() }

        return nodeFilters.any { filter ->
            candidates.any { candidate -> candidate.contains(filter) }
        }
    }

    private fun matchesAttributeFilter(
        record: TripleRecord,
        attributeFilters: Set<String>,
    ): Boolean {
        val candidates =
            buildList {
                addAll(record.sourceProperties.keys)
                addAll(record.sourceProperties.values)
                addAll(record.targetProperties.keys)
                addAll(record.targetProperties.values)
            }.map { value -> value.lowercase() }

        return attributeFilters.any { filter ->
            candidates.any { candidate -> candidate.contains(filter) }
        }
    }

    private fun expandTriplesFromSeeds(
        seedTriples: List<TripleRecord>,
        retrievalPool: List<TripleRecord>,
    ): List<TripleRecord> {
        if (seedTriples.isEmpty()) {
            return emptyList()
        }

        val pooledSet = retrievalPool.toSet()
        val collected = linkedSetOf<TripleRecord>()
        val visitedNodes = mutableSetOf<String>()
        var frontier =
            seedTriples
                .flatMap { seed -> listOf(seed.source, seed.target) }
                .toMutableSet()

        seedTriples.forEach { seed -> collected.add(seed) }
        visitedNodes.addAll(frontier)

        val hops = (recallPaths.coerceAtLeast(1) - 1)
        repeat(hops) {
            if (frontier.isEmpty()) {
                return@repeat
            }

            val nextFrontier = mutableSetOf<String>()
            frontier.forEach { node ->
                adjacencyByNode[node].orEmpty().forEach { candidate ->
                    if (candidate !in pooledSet) {
                        return@forEach
                    }
                    collected.add(candidate)
                    if (candidate.source !in visitedNodes) {
                        nextFrontier.add(candidate.source)
                    }
                    if (candidate.target !in visitedNodes) {
                        nextFrontier.add(candidate.target)
                    }
                }
            }
            frontier = nextFrontier
            visitedNodes.addAll(nextFrontier)
        }

        return collected.toList()
    }

    private fun vectorSearchScoresForTriples(
        question: String,
        searchLimit: Int,
        allowedTriples: List<TripleRecord>,
    ): Map<TripleRecord, Double> {
        val allowed = allowedTriples.toSet()
        return tripleVectorIndex
            ?.search(question, searchLimit)
            .orEmpty()
            .asSequence()
            .filter { hit -> hit.item in allowed }
            .associate { hit -> hit.item to hit.score }
    }

    private fun formatTripleWithScore(
        triple: TripleRecord,
        score: Double,
    ): String {
        val formattedScore = String.format(Locale.US, "%.3f", score)
        return "${triple.contextFormat} [score: $formattedScore]"
    }

    private fun keywordMatchCount(
        text: String,
        questionKeywords: Set<String>,
    ): Int {
        if (questionKeywords.isEmpty()) {
            return 0
        }

        val normalizedText = text.lowercase()
        return questionKeywords.count { keyword -> keyword in normalizedText }
    }

    private fun normalizeFilters(rawValues: List<String>?): Set<String> =
        rawValues
            .orEmpty()
            .asSequence()
            .map { value -> value.trim().lowercase() }
            .filter { value -> value.isNotBlank() }
            .toSet()

    private fun tokenize(input: String): Set<String> {
        val tokenRegex = Regex("[\\p{L}\\p{N}_]{2,}")
        return tokenRegex.findAll(input.lowercase()).map { it.value }.toSet()
    }

    private fun <T> rankedItems(
        items: List<T>,
        questionKeywords: Set<String>,
        topLimit: Int,
        textSelector: (T) -> String = { it.toString() },
    ): List<T> {
        if (items.isEmpty()) {
            return emptyList()
        }

        val scored =
            items.mapIndexed { index, item ->
                val text = textSelector(item).lowercase()
                val score = if (questionKeywords.isEmpty()) 0 else questionKeywords.count { keyword -> keyword in text }
                Triple(item, score, index)
            }

        val sorted =
            scored.sortedWith(
                compareByDescending<Triple<T, Int, Int>> { it.second }
                    .thenBy { it.third },
            )

        val selected =
            if (sorted.any { it.second > 0 }) {
                sorted.take(topLimit)
            } else {
                sorted.take(minOf(topLimit, sorted.size))
            }

        return selected.map { it.first }
    }

    private fun resolvePath(path: String): Path {
        val candidate = Path.of(path)
        return if (candidate.isAbsolute) candidate else rootDir.resolve(candidate)
    }
}

private class IndexBuilder(
    private val datasetName: String,
    private val config: ConfigManager,
    private val rootDir: Path,
    private val embedder: TextEmbedder,
) {
    private val logger = KotlinLogging.logger {}
    private val mapper = ObjectMapper().registerKotlinModule()
    private val cacheModelTag = embedder.modelTag
    private val tripleCacheFileName = "triple_embedding_cache.json"
    private val chunkCacheFileName = "chunk_embedding_cache.json"
    private val tripleCacheNpzFileName = "triple_embedding_cache.npz"
    private val chunkCacheNpzFileName = "chunk_embedding_cache.npz"

    fun loadTriples(graphFile: Path): List<TripleRecord> {
        if (!graphFile.exists()) {
            return emptyList()
        }

        val relationships =
            runCatching {
                mapper.readValue(graphFile.toFile(), object : TypeReference<List<GraphRelationship>>() {})
            }.getOrElse { error ->
                logger.warn(error) { "Failed to load graph relationships from $graphFile" }
                emptyList()
            }

        return relationships.map { relationship ->
            val source =
                relationship.startNode.properties["name"]
                    ?.toString()
                    ?.takeIf { value -> value.isNotBlank() }
                    ?: relationship.startNode.label
            val target =
                relationship.endNode.properties["name"]
                    ?.toString()
                    ?.takeIf { value -> value.isNotBlank() }
                    ?: relationship.endNode.label
            TripleRecord(
                source = source,
                relation = relationship.relation,
                target = target,
                sourceLabel = relationship.startNode.label,
                targetLabel = relationship.endNode.label,
                sourceProperties = relationship.startNode.properties.toStringValueMap(),
                targetProperties = relationship.endNode.properties.toStringValueMap(),
            )
        }
    }

    fun loadChunks(chunkFile: Path): Map<String, String> {
        if (!chunkFile.exists()) {
            return emptyMap()
        }

        val map = linkedMapOf<String, String>()
        chunkFile.readLines().forEach { line ->
            if (!line.startsWith("id:")) {
                return@forEach
            }

            val idPart = line.substringAfter("id:").substringBefore('\t').trim()
            if (idPart.isBlank()) {
                return@forEach
            }

            val chunkContent =
                if ("\tChunk:" in line) {
                    line.substringAfter("\tChunk:").trim()
                } else {
                    line.substringAfter("Chunk:", "").trim()
                }

            map[idPart] = chunkContent
        }

        return map
    }

    fun buildAdjacency(records: List<TripleRecord>): Map<String, List<TripleRecord>> {
        if (records.isEmpty()) {
            return emptyMap()
        }

        val adjacency = linkedMapOf<String, MutableList<TripleRecord>>()
        records.forEach { record ->
            adjacency.getOrPut(record.source) { mutableListOf() }.add(record)
            adjacency.getOrPut(record.target) { mutableListOf() }.add(record)
        }
        return adjacency.mapValues { (_, value) -> value.toList() }
    }

    fun prepareTripleVectors(
        tripleRecords: List<TripleRecord>,
        embedder: TextEmbedder,
    ): Map<TripleRecord, FloatArray> {
        if (tripleRecords.isEmpty()) {
            return emptyMap()
        }

        val expectedKeys = tripleRecords.map { record -> record.serialized }.toSet()
        val cachePath = resolveCacheFile(tripleCacheFileName)
        val cacheNpzPath = resolveCacheFile(tripleCacheNpzFileName)
        val cachedByKey =
            if (config.retrieval.enableCaching) {
                loadEmbeddingCache(cachePath, cacheNpzPath)
            } else {
                mutableMapOf()
            }

        val vectorsByRecord = linkedMapOf<TripleRecord, FloatArray>()
        var changed = false

        tripleRecords.forEach { record ->
            val key = record.serialized
            val vector =
                cachedByKey[key] ?: embedder.embed(record.searchableText).also {
                    if (config.retrieval.enableCaching) {
                        cachedByKey[key] = it
                        changed = true
                    }
                }
            vectorsByRecord[record] = vector
        }

        if (config.retrieval.enableCaching) {
            val staleKeys = cachedByKey.keys - expectedKeys
            if (staleKeys.isNotEmpty()) {
                staleKeys.forEach { staleKey -> cachedByKey.remove(staleKey) }
                changed = true
            }

            if (changed || !cachePath.exists() || !cacheNpzPath.exists()) {
                saveEmbeddingCache(cachePath, cacheNpzPath, cachedByKey)
            }
        }

        return vectorsByRecord
    }

    fun prepareChunkVectors(
        chunkRecords: List<RetrievalChunk>,
        embedder: TextEmbedder,
    ): Map<RetrievalChunk, FloatArray> {
        if (chunkRecords.isEmpty()) {
            return emptyMap()
        }

        val expectedKeys = chunkRecords.map { chunk -> chunk.id }.toSet()
        val cachePath = resolveCacheFile(chunkCacheFileName)
        val cacheNpzPath = resolveCacheFile(chunkCacheNpzFileName)
        val cachedByKey =
            if (config.retrieval.enableCaching) {
                loadEmbeddingCache(cachePath, cacheNpzPath)
            } else {
                mutableMapOf()
            }

        val vectorsByChunk = linkedMapOf<RetrievalChunk, FloatArray>()
        var changed = false

        chunkRecords.forEach { chunk ->
            val key = chunk.id
            val vector =
                cachedByKey[key] ?: embedder.embed(chunk.text).also {
                    if (config.retrieval.enableCaching) {
                        cachedByKey[key] = it
                        changed = true
                    }
                }
            vectorsByChunk[chunk] = vector
        }

        if (config.retrieval.enableCaching) {
            val staleKeys = cachedByKey.keys - expectedKeys
            if (staleKeys.isNotEmpty()) {
                staleKeys.forEach { staleKey -> cachedByKey.remove(staleKey) }
                changed = true
            }

            if (changed || !cachePath.exists() || !cacheNpzPath.exists()) {
                saveEmbeddingCache(cachePath, cacheNpzPath, cachedByKey)
            }
        }

        return vectorsByChunk
    }

    fun loadEmbeddingCache(
        cachePath: Path,
        cacheNpzPath: Path,
    ): MutableMap<String, FloatArray> {
        if (cacheNpzPath.exists()) {
            val npzVectors =
                runCatching {
                    NpzEmbeddingCache.read(cacheNpzPath, expectedDimensions = embedder.dimensions)
                }.getOrElse { error ->
                    logger.warn(error) { "Failed to load NPZ embedding cache from $cacheNpzPath. Falling back to JSON cache." }
                    emptyMap()
                }
            if (npzVectors.isNotEmpty()) {
                return npzVectors.toMutableMap()
            }
        }

        if (!cachePath.exists()) {
            return linkedMapOf()
        }

        val cacheFile =
            runCatching {
                mapper.readValue(cachePath.toFile(), EmbeddingCacheFile::class.java)
            }.getOrElse { error ->
                logger.warn(error) { "Failed to load embedding cache from $cachePath. Rebuilding cache entries." }
                return linkedMapOf()
            }

        if (cacheFile.modelTag != cacheModelTag || cacheFile.dimensions != embedder.dimensions) {
            logger.info {
                "Embedding cache metadata mismatch at $cachePath (modelTag='${cacheFile.modelTag}', dimensions=${cacheFile.dimensions}), rebuilding."
            }
            return linkedMapOf()
        }

        return cacheFile.vectors.entries
            .mapNotNull { (key, values) ->
                if (values.size != embedder.dimensions) {
                    null
                } else {
                    key to values.toFloatArray()
                }
            }.toMap(linkedMapOf())
    }

    fun saveEmbeddingCache(
        cachePath: Path,
        cacheNpzPath: Path,
        vectorsByKey: Map<String, FloatArray>,
    ) {
        runCatching {
            cachePath.parent?.createDirectories()
            val cacheFile =
                EmbeddingCacheFile(
                    modelTag = cacheModelTag,
                    dimensions = embedder.dimensions,
                    vectors = vectorsByKey.mapValues { (_, vector) -> vector.toList() },
                )
            mapper.writerWithDefaultPrettyPrinter().writeValue(cachePath.toFile(), cacheFile)
        }.onFailure { error ->
            logger.warn(error) { "Failed to persist embedding cache to $cachePath" }
        }

        runCatching {
            NpzEmbeddingCache.write(
                path = cacheNpzPath,
                vectorsByKey = vectorsByKey,
                expectedDimensions = embedder.dimensions,
            )
        }.onFailure { error ->
            logger.warn(error) { "Failed to persist NPZ embedding cache to $cacheNpzPath" }
        }
    }

    fun resolveCacheFile(fileName: String): Path =
        resolvePath(config.retrieval.cacheDir)
            .resolve(datasetName)
            .resolve(fileName)

    fun <T> buildSemanticIndex(
        items: List<T>,
        textSelector: (T) -> String,
        vectorSelector: ((T) -> FloatArray)? = null,
        embedder: TextEmbedder,
    ): SemanticVectorIndex<T> {
        if (items.isEmpty()) {
            return LocalVectorIndex.build(
                items = items,
                embedder = embedder,
                textSelector = textSelector,
                vectorSelector = vectorSelector,
            )
        }

        return runCatching {
            LuceneAnnIndex.build(
                items = items,
                embedder = embedder,
                textSelector = textSelector,
                vectorSelector = vectorSelector,
            )
        }.getOrElse { error ->
            logger.warn(error) { "Failed to initialize Lucene ANN index. Falling back to local vector index." }
            LocalVectorIndex.build(
                items = items,
                embedder = embedder,
                textSelector = textSelector,
                vectorSelector = vectorSelector,
            )
        }
    }

    fun resolvePath(path: String): Path {
        val candidate = Path.of(path)
        return if (candidate.isAbsolute) candidate else rootDir.resolve(candidate)
    }
}

private fun Map<String, Any?>.toStringValueMap(): Map<String, String> =
    entries.associate { (key, value) ->
        key to
            when (value) {
                is List<*> -> value.joinToString(", ") { item -> item?.toString().orEmpty() }
                null -> ""
                else -> value.toString()
            }
    }
