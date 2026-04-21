package hipporag

import hipporag.config.BaseConfig
import hipporag.embeddingmodel.BaseEmbeddingModel
import hipporag.embeddingmodel.getEmbeddingModel
import hipporag.embeddingstore.EmbeddingStore
import hipporag.evaluation.QAExactMatch
import hipporag.evaluation.QAF1Score
import hipporag.evaluation.RetrievalRecall
import hipporag.graph.SimpleGraph
import hipporag.informationextraction.OpenIE
import hipporag.informationextraction.OpenIEBase
import hipporag.informationextraction.TransformersOfflineOpenIE
import hipporag.informationextraction.VllmOfflineOpenIE
import hipporag.llm.BaseLLM
import hipporag.llm.getLlm
import hipporag.prompts.PromptTemplateManager
import hipporag.prompts.getQueryInstruction
import hipporag.rerank.DSPyFilter
import hipporag.utils.EmbeddingRow
import hipporag.utils.LlmResult
import hipporag.utils.Message
import hipporag.utils.NerRawOutput
import hipporag.utils.OpenieDoc
import hipporag.utils.OpenieResults
import hipporag.utils.QuerySolution
import hipporag.utils.RagQaResult
import hipporag.utils.TripleRawOutput
import hipporag.utils.computeMdHashId
import hipporag.utils.extractEntityNodes
import hipporag.utils.flattenFacts
import hipporag.utils.jsonWithDefaults
import hipporag.utils.minMaxNormalize
import hipporag.utils.parseTripleString
import hipporag.utils.reformatOpenieResults
import hipporag.utils.retrieveKnn
import hipporag.utils.textProcessing
import io.github.oshai.kotlinlogging.KotlinLogging
import java.io.File
import java.util.Locale
import kotlin.math.min

// NOTE: This is a direct Kotlin translation of ../src/hipporag/HippoRAG.py.
// It intentionally references external components (LLM, embeddings, OpenIE, etc.)
// that will be implemented elsewhere in the Kotlin port.

/**
 * Core HIPPO-RAG pipeline that builds the graph index and answers queries.
 *
 * @param initialConfig optional base configuration to copy.
 * @param saveDir optional override for [BaseConfig.saveDir].
 * @param llmModelName optional override for [BaseConfig.llmName].
 * @param llmBaseUrl optional override for [BaseConfig.llmBaseUrl].
 * @param embeddingModelName optional override for [BaseConfig.embeddingModelName].
 * @param embeddingBaseUrl optional override for [BaseConfig.embeddingBaseUrl].
 */
class HippoRag(
    initialConfig: BaseConfig? = null,
    saveDir: String? = null,
    llmModelName: String? = null,
    llmBaseUrl: String? = null,
    embeddingModelName: String? = null,
    embeddingBaseUrl: String? = null,
) {
    constructor(config: BaseConfig? = null) : this(initialConfig = config)

    private val logger = KotlinLogging.logger {}

    /** Effective configuration copied from [initialConfig] with any overrides applied. */
    val globalConfig: BaseConfig =
        run {
            (initialConfig ?: BaseConfig())
                .copy()
                .apply {
                    if (saveDir != null) this.saveDir = saveDir
                    if (llmModelName != null) this.llmName = llmModelName
                    if (embeddingModelName != null) this.embeddingModelName = embeddingModelName
                    if (llmBaseUrl != null) this.llmBaseUrl = llmBaseUrl
                    if (embeddingBaseUrl != null) this.embeddingBaseUrl = embeddingBaseUrl
                }
        }

    private val workingDir: String

    private val llmModel: BaseLLM
    private val openie: OpenIEBase
    private val graph: SimpleGraph

    private val embeddingModel: BaseEmbeddingModel?
    private val chunkEmbeddingStore: EmbeddingStore
    private val entityEmbeddingStore: EmbeddingStore
    private val factEmbeddingStore: EmbeddingStore

    private val promptTemplateManager: PromptTemplateManager

    private val openieResultsPath: String
    private val rerankFilter: DSPyFilter

    private var readyToRetrieve: Boolean = false

    private var pprTimeSeconds: Double = 0.0
    private var rerankTimeSeconds: Double = 0.0
    private var allRetrievalTimeSeconds: Double = 0.0

    private var entNodeToChunkIds: MutableMap<String, MutableSet<String>>? = null
    private var nodeToNodeStats: MutableMap<Pair<String, String>, Double> = mutableMapOf()

    // Retrieval caches
    private var queryToEmbedding: MutableMap<String, MutableMap<String, DoubleArray>> =
        mutableMapOf("triple" to mutableMapOf(), "passage" to mutableMapOf())

    private var entityNodeKeys: List<String> = emptyList()
    private var passageNodeKeys: List<String> = emptyList()
    private var factNodeKeys: List<String> = emptyList()

    private var nodeNameToVertexIdx: Map<String, Int> = emptyMap()
    private var entityNodeIdxs: IntArray = intArrayOf()
    private var passageNodeIdxs: IntArray = intArrayOf()

    private var entityEmbeddings: Array<DoubleArray> = emptyArray()
    private var passageEmbeddings: Array<DoubleArray> = emptyArray()
    private var factEmbeddings: Array<DoubleArray> = emptyArray()

    private var procTriplesToDocs: MutableMap<String, MutableSet<String>> = mutableMapOf()

    init {
        val configDump = globalConfig.toMap().entries.joinToString(",\n  ") { (k, v) -> "$k = $v" }
        logger.debug { "HippoRAG init with config:\n  $configDump\n" }

        val llmLabel = globalConfig.llmName.replace("/", "_")
        val embeddingLabel = globalConfig.embeddingModelName.replace("/", "_")
        workingDir = File(globalConfig.saveDir, "${llmLabel}_$embeddingLabel").path

        if (!File(workingDir).exists()) {
            logger.info { "Creating working directory: $workingDir" }
            File(workingDir).mkdirs()
        }

        llmModel = getLlm(globalConfig)

        val openieMode = globalConfig.openieMode.lowercase()

        openie =
            when (openieMode) {
                "online" -> {
                    OpenIE(llmModel)
                }

                "offline" -> {
                    VllmOfflineOpenIE(llmModel)
                }

                "transformers-offline" -> {
                    TransformersOfflineOpenIE(llmModel)
                }

                else -> {
                    logger.warn { "Unknown openieMode '${globalConfig.openieMode}', defaulting to 'online'." }
                    OpenIE(llmModel)
                }
            }

        graph = initializeGraph()

        embeddingModel =
            if (openieMode == "offline") {
                null
            } else {
                getEmbeddingModel().create(globalConfig, globalConfig.embeddingModelName)
            }

        chunkEmbeddingStore =
            EmbeddingStore(
                embeddingModel,
                File(workingDir, "chunk_embeddings").path,
                "chunk",
            )
        entityEmbeddingStore =
            EmbeddingStore(
                embeddingModel,
                File(workingDir, "entity_embeddings").path,
                "entity",
            )
        factEmbeddingStore =
            EmbeddingStore(
                embeddingModel,
                File(workingDir, "fact_embeddings").path,
                "fact",
            )

        promptTemplateManager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        openieResultsPath =
            File(
                globalConfig.saveDir,
                "openie_results_ner_${globalConfig.llmName.replace("/", "_")}.json",
            ).path

        rerankFilter = DSPyFilter(llmModel, globalConfig.rerankDspyFilePath)
    }

    private fun initializeGraph(): SimpleGraph {
        val graphFile = File(workingDir, "graph.json")
        return if (!globalConfig.forceIndexFromScratch && graphFile.exists()) {
            SimpleGraph.load(graphFile)
        } else {
            SimpleGraph(directed = globalConfig.isDirectedGraph)
        }
    }

    /**
     * Runs offline OpenIE extraction over [docs] and persists results for later indexing.
     */
    fun preOpenie(docs: List<String>) {
        logger.info { "Indexing Documents" }
        logger.info { "Performing OpenIE Offline" }

        val cleanedDocs = docs.filter { it.isNotBlank() }
        if (cleanedDocs.size < docs.size) {
            logger.warn { "Skipping ${docs.size - cleanedDocs.size} blank documents during preOpenie." }
        }
        if (cleanedDocs.isEmpty()) {
            logger.warn { "No non-blank documents provided for preOpenie; skipping." }
            return
        }

        val chunks = chunkEmbeddingStore.getMissingStringHashIds(cleanedDocs)
        val (allOpenieInfo, chunkKeysToProcess) = loadExistingOpenie(chunks.keys.toList())
        val newOpenieRows = chunks.filterKeys { it in chunkKeysToProcess }

        if (chunkKeysToProcess.isNotEmpty()) {
            val (newNerResults, newTripleResults) = openie.batchOpenie(newOpenieRows)
            mergeOpenieResults(allOpenieInfo, newOpenieRows, newNerResults, newTripleResults)
        }

        if (globalConfig.saveOpenie) {
            saveOpenieResults(allOpenieInfo)
        }

        logger.info { "Offline OpenIE complete. Run online indexing for future retrieval." }
    }

    /**
     * Indexes documents into the graph and embedding stores.
     */
    fun index(docs: List<String>) {
        logger.info { "Indexing Documents" }
        logger.info { "Performing OpenIE" }

        if (globalConfig.openieMode.lowercase() == "offline") {
            logger.info { "Offline OpenIE mode: run preOpenie() separately before calling index()." }
            return
        }

        val cleanedDocs = docs.filter { it.isNotBlank() }
        if (cleanedDocs.size < docs.size) {
            logger.warn { "Skipping ${docs.size - cleanedDocs.size} blank documents during index." }
        }
        if (cleanedDocs.isEmpty()) {
            logger.warn { "No non-blank documents provided for index; skipping." }
            return
        }

        chunkEmbeddingStore.insertStrings(cleanedDocs)
        val chunkToRows = chunkEmbeddingStore.getAllIdToRows()

        val (allOpenieInfo, chunkKeysToProcess) = loadExistingOpenie(chunkToRows.keys.toList())
        val newOpenieRows = chunkToRows.filterKeys { it in chunkKeysToProcess }

        if (chunkKeysToProcess.isNotEmpty()) {
            val (newNerResults, newTripleResults) = openie.batchOpenie(newOpenieRows)
            mergeOpenieResults(allOpenieInfo, newOpenieRows, newNerResults, newTripleResults)
        }

        if (globalConfig.saveOpenie) {
            saveOpenieResults(allOpenieInfo)
        }

        val (nerResults, tripleResults) = reformatOpenieResults(allOpenieInfo)

        check(chunkToRows.size == nerResults.size && chunkToRows.size == tripleResults.size) {
            "len(chunkToRows): ${chunkToRows.size}, len(nerResults): ${nerResults.size}, len(tripleResults): ${tripleResults.size}"
        }

        val chunkIds = chunkToRows.keys.toList()
        val chunkTriples =
            chunkIds.map { chunkId ->
                tripleResults.getValue(chunkId).triples.map { t -> textProcessing(t) }
            }

        val (entityNodes, chunkTripleEntities) = extractEntityNodes(chunkTriples)
        val facts = flattenFacts(chunkTriples)

        logger.info { "Encoding Entities" }
        entityEmbeddingStore.insertStrings(entityNodes)

        logger.info { "Encoding Facts" }
        factEmbeddingStore.insertStrings(facts.map { it.toString() })

        logger.info { "Constructing Graph" }

        nodeToNodeStats = mutableMapOf()
        entNodeToChunkIds = mutableMapOf()

        addFactEdges(chunkIds, chunkTriples)
        val numNewChunks = addPassageEdges(chunkIds, chunkTripleEntities)

        if (numNewChunks > 0) {
            logger.info { "Found $numNewChunks new chunks to save into graph." }
            addSynonymyEdges()
            augmentGraph()
            saveIgraph()
        }
    }

    /**
     * Deletes documents and related graph nodes from the index.
     */
    fun delete(docsToDelete: List<String>) {
        if (!readyToRetrieve) {
            prepareRetrievalObjects()
        }

        val currentDocs = chunkEmbeddingStore.getAllTexts()
        val actualDocsToDelete = docsToDelete.filter { it in currentDocs }

        val chunkIdsToDelete = actualDocsToDelete.mapNotNull { chunkEmbeddingStore.textToHashId[it] }.toSet()

        val (allOpenieInfo, _) = loadExistingOpenie(emptyList())
        val triplesToDelete = mutableListOf<List<List<String>>>()
        val allOpenieInfoWithDeletes = mutableListOf<OpenieDoc>()

        for (openieDoc in allOpenieInfo) {
            if (openieDoc.idx in chunkIdsToDelete) {
                triplesToDelete.add(openieDoc.extractedTriples)
            } else {
                allOpenieInfoWithDeletes.add(openieDoc)
            }
        }

        val flattenedTriplesToDelete = flattenFacts(triplesToDelete)

        val trueTriplesToDelete = mutableListOf<List<String>>()
        for (triple in flattenedTriplesToDelete) {
            val procTriple = textProcessing(triple)
            val docIds = procTriplesToDocs[procTriple.toString()] ?: emptySet()
            val nonDeletedDocs = docIds.subtract(chunkIdsToDelete)
            if (nonDeletedDocs.isEmpty()) {
                trueTriplesToDelete.add(triple)
            }
        }

        val processedTrueTriplesToDelete = trueTriplesToDelete.map { textProcessing(it) }
        val (entitiesToDelete, _) = extractEntityNodes(listOf(processedTrueTriplesToDelete))
        val processedTrueTriplesFlat = flattenFacts(listOf(processedTrueTriplesToDelete))

        val tripleIdsToDelete =
            processedTrueTriplesFlat
                .mapNotNull { factEmbeddingStore.textToHashId[it.toString()] }
                .toSet()

        val entIdsToDelete = entitiesToDelete.mapNotNull { entityEmbeddingStore.textToHashId[it] }
        val filteredEntIdsToDelete = mutableListOf<String>()

        for (entNode in entIdsToDelete) {
            val docIds = entNodeToChunkIds?.get(entNode) ?: emptySet()
            val nonDeletedDocs = docIds.subtract(chunkIdsToDelete)
            if (nonDeletedDocs.isEmpty()) {
                filteredEntIdsToDelete.add(entNode)
            }
        }

        logger.info { "Deleting ${chunkIdsToDelete.size} Chunks" }
        logger.info { "Deleting ${tripleIdsToDelete.size} Triples" }
        logger.info { "Deleting ${filteredEntIdsToDelete.size} Entities" }

        saveOpenieResults(allOpenieInfoWithDeletes)

        entityEmbeddingStore.delete(filteredEntIdsToDelete)
        factEmbeddingStore.delete(tripleIdsToDelete)
        chunkEmbeddingStore.delete(chunkIdsToDelete)

        graph.deleteVertices((filteredEntIdsToDelete + chunkIdsToDelete).toList())
        saveIgraph()

        readyToRetrieve = false
    }

    /**
     * Retrieves passages for [queries] using graph-aware retrieval.
     *
     * @return a pair of solutions and optional recall metrics (when [goldDocs] is provided).
     */
    fun retrieve(
        queries: List<String>,
        numToRetrieve: Int? = null,
        goldDocs: List<List<String>>? = null,
    ): Pair<List<QuerySolution>, Map<String, Double>?> =
        retrieveInternal(
            queries = queries,
            numToRetrieve = numToRetrieve,
            goldDocs = goldDocs,
            perQueryRetrieval = { query ->
                val rerankStart = nowSeconds()
                val queryFactScores = getFactScores(query)
                val (topKFactIndices, topKFacts, _) = rerankFacts(query, queryFactScores)
                val rerankEnd = nowSeconds()

                rerankTimeSeconds += rerankEnd - rerankStart

                if (topKFacts.isEmpty()) {
                    logger.info { "No facts found after reranking, return DPR results" }
                    densePassageRetrieval(query)
                } else {
                    graphSearchWithFactEntities(
                        query = query,
                        linkTopK = globalConfig.linkingTopK,
                        queryFactScores = queryFactScores,
                        topKFacts = topKFacts,
                        topKFactIndices = topKFactIndices,
                        passageNodeWeight = globalConfig.passageNodeWeight,
                    )
                }
            },
            logTimings = {
                logger.info { "Total Retrieval Time ${String.format(Locale.US, "%.2f", allRetrievalTimeSeconds)}s" }
                logger.info { "Total Recognition Memory Time ${String.format(Locale.US, "%.2f", rerankTimeSeconds)}s" }
                logger.info { "Total PPR Time ${String.format(Locale.US, "%.2f", pprTimeSeconds)}s" }
                logger.info {
                    "Total Misc Time ${String.format(Locale.US, "%.2f", allRetrievalTimeSeconds - (rerankTimeSeconds + pprTimeSeconds))}s"
                }
            },
        )

    /**
     * Runs retrieval + LLM QA for [queries].
     *
     * @return a [RagQaResult] with answers and optional evaluation metrics.
     */
    fun ragQa(
        queries: List<String>,
        goldDocs: List<List<String>>? = null,
        goldAnswers: List<List<String>>? = null,
    ): RagQaResult {
        val (retrieved, retrievalResult) =
            retrieve(
                queries = queries,
                goldDocs = goldDocs,
            )
        return ragQaFromSolutions(
            querySolutions = retrieved,
            goldDocs = goldDocs,
            goldAnswers = goldAnswers,
            overallRetrievalResult = retrievalResult,
        )
    }

    /**
     * Runs LLM QA over precomputed retrieval [queries].
     */
    fun ragQaWithSolutions(
        queries: List<QuerySolution>,
        goldDocs: List<List<String>>? = null,
        goldAnswers: List<List<String>>? = null,
    ): RagQaResult =
        ragQaFromSolutions(
            querySolutions = queries,
            goldDocs = goldDocs,
            goldAnswers = goldAnswers,
            overallRetrievalResult = null,
        )

    private fun ragQaFromSolutions(
        querySolutions: List<QuerySolution>,
        goldDocs: List<List<String>>?,
        goldAnswers: List<List<String>>?,
        overallRetrievalResult: Map<String, Double>?,
    ): RagQaResult {
        val qaEmEvaluator = if (goldAnswers != null) QAExactMatch() else null
        val qaF1Evaluator = if (goldAnswers != null) QAF1Score() else null

        val (qaSolutions, allResponseMessage, allMetadata) = qa(querySolutions)

        if (goldAnswers != null && qaEmEvaluator != null && qaF1Evaluator != null) {
            val (overallQaEm, _) =
                qaEmEvaluator.calculateMetricScores(
                    goldAnswers = goldAnswers,
                    predictedAnswers = qaSolutions.map { it.answer ?: "" },
                    aggregationFn = { values -> values.maxOrNull() ?: 0.0 },
                )
            val (overallQaF1, _) =
                qaF1Evaluator.calculateMetricScores(
                    goldAnswers = goldAnswers,
                    predictedAnswers = qaSolutions.map { it.answer ?: "" },
                    aggregationFn = { values -> values.maxOrNull() ?: 0.0 },
                )

            val overallQaResults =
                (overallQaEm + overallQaF1).mapValues { (_, v) ->
                    String.format(Locale.US, "%.4f", v).toDouble()
                }
            logger.info { "Evaluation results for QA: $overallQaResults" }

            val qaSolutionsWithGold =
                qaSolutions.mapIndexed { idx, solution ->
                    solution.copy(
                        goldAnswers = goldAnswers[idx].toMutableList(),
                        goldDocs = goldDocs?.getOrNull(idx)?.toMutableList(),
                    )
                }

            return RagQaResult(
                solutions = qaSolutionsWithGold,
                responseMessages = allResponseMessage,
                metadata = allMetadata,
                overallRetrievalResult = overallRetrievalResult,
                overallQaResults = overallQaResults,
            )
        }

        return RagQaResult(
            solutions = qaSolutions,
            responseMessages = allResponseMessage,
            metadata = allMetadata,
            overallRetrievalResult = overallRetrievalResult,
            overallQaResults = null,
        )
    }

    /**
     * Retrieves passages using dense passage retrieval (DPR) only.
     *
     * @return a pair of solutions and optional recall metrics (when [goldDocs] is provided).
     */
    fun retrieveDpr(
        queries: List<String>,
        numToRetrieve: Int? = null,
        goldDocs: List<List<String>>? = null,
    ): Pair<List<QuerySolution>, Map<String, Double>?> =
        retrieveInternal(
            queries = queries,
            numToRetrieve = numToRetrieve,
            goldDocs = goldDocs,
            perQueryRetrieval = { query ->
                logger.info { "Performing DPR retrieval for query." }
                densePassageRetrieval(query)
            },
            logTimings = {
                logger.info { "Total Retrieval Time ${String.format(Locale.US, "%.2f", allRetrievalTimeSeconds)}s" }
            },
        )

    /**
     * Runs DPR-only retrieval + LLM QA for [queries].
     */
    fun ragQaDpr(
        queries: List<String>,
        goldDocs: List<List<String>>? = null,
        goldAnswers: List<List<String>>? = null,
    ): RagQaResult {
        val (retrieved, retrievalResult) =
            retrieveDpr(
                queries = queries,
                goldDocs = goldDocs,
            )
        return ragQaFromSolutions(
            querySolutions = retrieved,
            goldDocs = goldDocs,
            goldAnswers = goldAnswers,
            overallRetrievalResult = retrievalResult,
        )
    }

    /**
     * Runs LLM QA over precomputed DPR retrieval [queries].
     */
    fun ragQaDprWithSolutions(
        queries: List<QuerySolution>,
        goldDocs: List<List<String>>? = null,
        goldAnswers: List<List<String>>? = null,
    ): RagQaResult =
        ragQaFromSolutions(
            querySolutions = queries,
            goldDocs = goldDocs,
            goldAnswers = goldAnswers,
            overallRetrievalResult = null,
        )

    /**
     * Executes QA prompts over retrieved passages.
     *
     * @return updated solutions, response messages, and per-response metadata.
     */
    fun qa(queries: List<QuerySolution>): Triple<List<QuerySolution>, List<String>, List<Map<String, Any?>>> {
        val allQaMessages = mutableListOf<List<Message>>()

        for (querySolution in queries) {
            val retrievedPassages = querySolution.docs.take(globalConfig.qaTopK)

            val promptUser =
                buildString {
                    for (passage in retrievedPassages) {
                        append("Wikipedia Title: ")
                        append(passage)
                        append("\n\n")
                    }
                    append("Question: ")
                    append(querySolution.question)
                    append("\nThought: ")
                }

            val promptDatasetName =
                if (
                    promptTemplateManager.isTemplateNameValid("rag_qa_${globalConfig.dataset}")
                ) {
                    globalConfig.dataset
                } else {
                    logger.debug {
                        "rag_qa_${globalConfig.dataset} does not have a customized prompt template. " +
                            "Using MUSIQUE's prompt template instead."
                    }
                    "musique"
                }

            allQaMessages.add(
                promptTemplateManager.render(
                    name = "rag_qa_$promptDatasetName",
                    promptUser = promptUser,
                ),
            )
        }

        val allQaResults = allQaMessages.map { qaMessages -> llmModel.infer(qaMessages) }

        val allResponseMessage = allQaResults.map { it.response }
        val allMetadata = allQaResults.map { it.metadata }

        val querySolutions =
            queries.mapIndexed { idx, querySolution ->
                val responseContent = allResponseMessage[idx]
                val predAns = responseContent.substringAfter("Answer:", responseContent).trim()
                querySolution.copy(answer = predAns)
            }

        return Triple(querySolutions, allResponseMessage, allMetadata)
    }

    private fun addFactEdges(
        chunkIds: List<String>,
        chunkTriples: List<List<List<String>>>,
    ) {
        val currentGraphNodes = graph.vertexNames().toSet()
        logger.info { "Adding OpenIE triples to graph." }

        chunkIds.zip(chunkTriples).forEach { (chunkKey, triples) ->
            val entitiesInChunk = mutableSetOf<String>()
            if (chunkKey !in currentGraphNodes) {
                for (triple in triples) {
                    if (triple.size < 3) continue
                    val nodeKey = computeMdHashId(triple[0], prefix = "entity-")
                    val node2Key = computeMdHashId(triple[2], prefix = "entity-")

                    nodeToNodeStats[nodeKey to node2Key] = (nodeToNodeStats[nodeKey to node2Key] ?: 0.0) + 1.0
                    nodeToNodeStats[node2Key to nodeKey] = (nodeToNodeStats[node2Key to nodeKey] ?: 0.0) + 1.0

                    entitiesInChunk.add(nodeKey)
                    entitiesInChunk.add(node2Key)
                }

                for (node in entitiesInChunk) {
                    val existing = entNodeToChunkIds?.get(node) ?: mutableSetOf()
                    existing.add(chunkKey)
                    entNodeToChunkIds?.set(node, existing)
                }
            }
        }
    }

    private fun addPassageEdges(
        chunkIds: List<String>,
        chunkTripleEntities: List<List<String>>,
    ): Int {
        val currentGraphNodes = graph.vertexNames().toSet()
        var numNewChunks = 0

        logger.info { "Connecting passage nodes to phrase nodes." }

        chunkIds.forEachIndexed { idx, chunkKey ->
            if (chunkKey !in currentGraphNodes) {
                for (chunkEnt in chunkTripleEntities[idx]) {
                    val nodeKey = computeMdHashId(chunkEnt, prefix = "entity-")
                    nodeToNodeStats[chunkKey to nodeKey] = 1.0
                }
                numNewChunks += 1
            }
        }

        return numNewChunks
    }

    private fun addSynonymyEdges() {
        logger.info { "Expanding graph with synonymy edges" }

        val entityIdToRow = entityEmbeddingStore.getAllIdToRows()
        val entityNodeKeys = entityIdToRow.keys.toList()

        logger.info { "Performing KNN retrieval for each phrase nodes (${entityNodeKeys.size})." }

        val entityEmbs = entityEmbeddingStore.getEmbeddings(entityNodeKeys)

        val queryNodeKey2KnnNodeKeys =
            retrieveKnn(
                queryIds = entityNodeKeys,
                keyIds = entityNodeKeys,
                queryVecs = entityEmbs,
                keyVecs = entityEmbs,
                k = globalConfig.synonymyEdgeTopK,
            )

        var numSynonymTriple = 0
        val synonymCandidates = mutableListOf<Pair<String, List<Pair<String, Double>>>>()

        for (nodeKey in queryNodeKey2KnnNodeKeys.keys) {
            val synonyms = mutableListOf<Pair<String, Double>>()
            val entity = entityIdToRow.getValue(nodeKey).content

            if (entity.replace(Regex("[^A-Za-z0-9]"), "").length > 2) {
                val (nns, scores) = queryNodeKey2KnnNodeKeys.getValue(nodeKey)
                var numNns = 0

                for ((nn, score) in nns.zip(scores)) {
                    if (score < globalConfig.synonymyEdgeSimThreshold || numNns > 100) break
                    val nnPhrase = entityIdToRow.getValue(nn).content

                    if (nn != nodeKey && nnPhrase.isNotEmpty()) {
                        val simEdge = nodeKey to nn
                        synonyms.add(nn to score)
                        numSynonymTriple += 1
                        nodeToNodeStats[simEdge] = score
                        numNns += 1
                    }
                }
            }

            synonymCandidates.add(nodeKey to synonyms)
        }
    }

    private fun loadExistingOpenie(chunkKeys: List<String>): Pair<MutableList<OpenieDoc>, Set<String>> {
        val chunkKeysToSave = mutableSetOf<String>()

        val openieFile = File(openieResultsPath)
        val allOpenieInfo =
            if (!globalConfig.forceOpenieFromScratch && openieFile.exists()) {
                val json = jsonWithDefaults { ignoreUnknownKeys = true }
                val openieResults = json.decodeFromString(OpenieResults.serializer(), openieFile.readText())
                val renamed =
                    openieResults.docs.map { doc ->
                        doc.copy(idx = computeMdHashId(doc.passage, prefix = "chunk-"))
                    }

                // Cache-safe behavior: when indexing a specific chunk set, only reuse matching
                // OpenIE docs and ignore stale entries from previous samples/runs.
                val dedupedByChunkId = linkedMapOf<String, OpenieDoc>()
                for (doc in renamed) {
                    dedupedByChunkId[doc.idx] = doc
                }

                val scopedExisting =
                    if (chunkKeys.isEmpty()) {
                        dedupedByChunkId.values.toList()
                    } else {
                        dedupedByChunkId.values.filter { it.idx in chunkKeys }
                    }

                if (chunkKeys.isNotEmpty()) {
                    val dropped = dedupedByChunkId.size - scopedExisting.size
                    if (dropped > 0) {
                        logger.info {
                            "Ignoring $dropped stale OpenIE cache docs not present in current chunk set."
                        }
                    }
                }

                val existingKeys = scopedExisting.map { it.idx }.toSet()

                for (chunkKey in chunkKeys) {
                    if (chunkKey !in existingKeys) {
                        chunkKeysToSave.add(chunkKey)
                    }
                }

                scopedExisting.toMutableList()
            } else {
                chunkKeysToSave.addAll(chunkKeys)
                mutableListOf()
            }

        return allOpenieInfo to chunkKeysToSave
    }

    private fun mergeOpenieResults(
        allOpenieInfo: MutableList<OpenieDoc>,
        chunksToSave: Map<String, EmbeddingRow>,
        nerResultsDict: Map<String, NerRawOutput>,
        tripleResultsDict: Map<String, TripleRawOutput>,
    ) {
        for ((chunkKey, row) in chunksToSave) {
            val passage = row.content
            val nerOutput = nerResultsDict[chunkKey]
            val tripleOutput = tripleResultsDict[chunkKey]
            if (nerOutput == null || tripleOutput == null) {
                logger.error { "Missing OpenIE results for chunk $chunkKey" }
            }
            val chunkOpenieInfo =
                OpenieDoc(
                    idx = chunkKey,
                    passage = passage,
                    extractedEntities = nerOutput?.uniqueEntities ?: emptyList(),
                    extractedTriples = tripleOutput?.triples ?: emptyList(),
                )
            allOpenieInfo.add(chunkOpenieInfo)
        }
    }

    private fun retrieveInternal(
        queries: List<String>,
        numToRetrieve: Int?,
        goldDocs: List<List<String>>?,
        perQueryRetrieval: (String) -> Pair<List<Int>, DoubleArray>,
        logTimings: () -> Unit,
    ): Pair<List<QuerySolution>, Map<String, Double>?> {
        val retrieveStart = nowSeconds()

        pprTimeSeconds = 0.0
        rerankTimeSeconds = 0.0
        allRetrievalTimeSeconds = 0.0

        check(embeddingModel != null) {
            "Embedding model is required for retrieval. If you used openieMode='offline', " +
                "set an embedding model or switch to online mode before calling retrieve()."
        }

        val k = numToRetrieve ?: globalConfig.retrievalTopK
        val retrievalRecallEvaluator = if (goldDocs != null) RetrievalRecall() else null

        if (!readyToRetrieve) {
            prepareRetrievalObjects()
        }

        getQueryEmbeddings(queries)

        val retrievalResults = mutableListOf<QuerySolution>()

        for (query in queries) {
            val (sortedDocIds, sortedDocScores) = perQueryRetrieval(query)
            val docPairs = sortedDocIds.zip(sortedDocScores.toList())
            val validPairs = docPairs.filter { (idx, _) -> idx in passageNodeKeys.indices }
            if (validPairs.size < docPairs.size) {
                logger.error {
                    "Retrieval returned ${docPairs.size - validPairs.size} out-of-range indices; " +
                        "passageNodeKeys size=${passageNodeKeys.size}"
                }
            }

            val topKDocs =
                validPairs.take(k).map { (idx, _) ->
                    chunkEmbeddingStore.getRow(passageNodeKeys[idx]).content
                }
            val topKScores = validPairs.take(k).map { it.second }.toDoubleArray()

            retrievalResults.add(
                QuerySolution(
                    question = query,
                    docs = topKDocs,
                    docScores = topKScores,
                ),
            )
        }

        val retrieveEnd = nowSeconds()
        allRetrievalTimeSeconds += retrieveEnd - retrieveStart

        logTimings()

        if (goldDocs != null && retrievalRecallEvaluator != null) {
            val kList = listOf(1, 2, 5, 10, 20, 30, 50, 100, 150, 200)
            val (overall, _) =
                retrievalRecallEvaluator.calculateMetricScores(
                    goldDocs = goldDocs,
                    retrievedDocs = retrievalResults.map { it.docs },
                    kList = kList,
                )
            logger.info { "Evaluation results for retrieval: $overall" }
            return retrievalResults to overall
        }

        return retrievalResults to null
    }

    private fun saveOpenieResults(allOpenieInfo: List<OpenieDoc>) {
        val sumPhraseChars = allOpenieInfo.sumOf { doc -> doc.extractedEntities.sumOf { it.length } }
        val sumPhraseWords = allOpenieInfo.sumOf { doc -> doc.extractedEntities.sumOf { it.split(" ").size } }
        val numPhrases = allOpenieInfo.sumOf { it.extractedEntities.size }

        if (allOpenieInfo.isNotEmpty()) {
            val avgEntChars = if (numPhrases > 0) sumPhraseChars.toDouble() / numPhrases else 0.0
            val avgEntWords = if (numPhrases > 0) sumPhraseWords.toDouble() / numPhrases else 0.0

            val openieDict =
                OpenieResults(
                    docs = allOpenieInfo,
                    avgEntChars = round4(avgEntChars),
                    avgEntWords = round4(avgEntWords),
                )

            val json = jsonWithDefaults { prettyPrint = false }
            File(openieResultsPath).writeText(json.encodeToString(OpenieResults.serializer(), openieDict))
            logger.info { "OpenIE results saved to $openieResultsPath" }
        }
    }

    private fun augmentGraph() {
        addNewNodes()
        addNewEdges()

        logger.info { "Graph construction completed!" }
        logger.info { getGraphInfo().toString() }
    }

    private fun addNewNodes() {
        val existingNodes = graph.vertexNames().toSet()

        val entityToRow = entityEmbeddingStore.getAllIdToRows()
        val passageToRow = chunkEmbeddingStore.getAllIdToRows()

        val nodeToRows = entityToRow.toMutableMap()
        nodeToRows.putAll(passageToRow)

        val newNodes = mutableMapOf<String, MutableList<Any>>()
        for ((nodeId, node) in nodeToRows) {
            val enriched = node.copy(name = nodeId)
            if (nodeId !in existingNodes) {
                for ((k, v) in enriched.toAttributes()) {
                    newNodes.getOrPut(k) { mutableListOf() }.add(v)
                }
            }
        }

        if (newNodes.isNotEmpty()) {
            graph.addVertices(newNodes)
        }
    }

    private fun addNewEdges() {
        val graphAdjList = mutableMapOf<String, MutableMap<String, Double>>()
        val graphInverseAdjList = mutableMapOf<String, MutableMap<String, Double>>()

        val edgeSourceNodeKeys = mutableListOf<String>()
        val edgeTargetNodeKeys = mutableListOf<String>()
        val edgeWeights = mutableListOf<Double>()

        for ((edge, weight) in nodeToNodeStats) {
            if (edge.first == edge.second) continue
            graphAdjList.getOrPut(edge.first) { mutableMapOf() }[edge.second] = weight
            graphInverseAdjList.getOrPut(edge.second) { mutableMapOf() }[edge.first] = weight
            edgeSourceNodeKeys.add(edge.first)
            edgeTargetNodeKeys.add(edge.second)
            edgeWeights.add(weight)
        }

        val currentNodeIds = graph.vertexNames().toSet()
        val validEdges = mutableListOf<Pair<String, String>>()
        val validWeights = mutableListOf<Double>()

        edgeSourceNodeKeys.zip(edgeTargetNodeKeys).zip(edgeWeights).forEach { (edgePair, weight) ->
            val (sourceNodeId, targetNodeId) = edgePair
            if (sourceNodeId in currentNodeIds && targetNodeId in currentNodeIds) {
                validEdges.add(sourceNodeId to targetNodeId)
                validWeights.add(weight)
            } else {
                logger.warn { "Edge $sourceNodeId -> $targetNodeId is not valid." }
            }
        }

        graph.addEdges(validEdges, validWeights)
    }

    private fun saveIgraph() {
        logger.info { "Writing graph with ${graph.vcount()} nodes, ${graph.ecount()} edges" }
        graph.save(File(workingDir, "graph.json"))
        logger.info { "Saving graph completed!" }
    }

    private fun getGraphInfo(): Map<String, Int> {
        val graphInfo = mutableMapOf<String, Int>()

        val phraseNodesKeys = entityEmbeddingStore.getAllIds()
        graphInfo["num_phrase_nodes"] = phraseNodesKeys.toSet().size

        val passageNodesKeys = chunkEmbeddingStore.getAllIds()
        graphInfo["num_passage_nodes"] = passageNodesKeys.toSet().size

        graphInfo["num_total_nodes"] = graphInfo.getValue("num_phrase_nodes") + graphInfo.getValue("num_passage_nodes")

        graphInfo["num_extracted_triples"] = factEmbeddingStore.getAllIds().size

        val passageNodesSet = passageNodesKeys.toSet()
        val numTriplesWithPassageNode =
            nodeToNodeStats.count { (pair, _) ->
                pair.first in passageNodesSet || pair.second in passageNodesSet
            }
        graphInfo["num_triples_with_passage_node"] = numTriplesWithPassageNode

        graphInfo["num_synonymy_triples"] = nodeToNodeStats.size -
            graphInfo.getValue("num_extracted_triples") - numTriplesWithPassageNode

        graphInfo["num_total_triples"] = nodeToNodeStats.size

        return graphInfo
    }

    private fun prepareRetrievalObjects() {
        logger.info { "Preparing for fast retrieval." }
        logger.info { "Loading keys." }

        queryToEmbedding = mutableMapOf("triple" to mutableMapOf(), "passage" to mutableMapOf())

        entityNodeKeys = entityEmbeddingStore.getAllIds()
        passageNodeKeys = chunkEmbeddingStore.getAllIds()
        factNodeKeys = factEmbeddingStore.getAllIds()

        val expectedNodeCount = entityNodeKeys.size + passageNodeKeys.size
        val actualNodeCount = graph.vcount()

        if (expectedNodeCount != actualNodeCount) {
            logger.warn { "Graph node count mismatch: expected $expectedNodeCount, got $actualNodeCount" }
            if (actualNodeCount == 0 && expectedNodeCount > 0) {
                logger.info { "Initializing graph with $expectedNodeCount nodes" }
                addNewNodes()
                saveIgraph()
            }
        }

        val nameToIdx = graph.vertexNames().mapIndexed { idx, name -> name to idx }.toMap()
        nodeNameToVertexIdx = nameToIdx

        val missingEntityNodes = entityNodeKeys.filter { it !in nameToIdx }
        val missingPassageNodes = passageNodeKeys.filter { it !in nameToIdx }

        if (missingEntityNodes.isNotEmpty() || missingPassageNodes.isNotEmpty()) {
            logger.warn {
                "Missing nodes in graph: ${missingEntityNodes.size} entity nodes, ${missingPassageNodes.size} passage nodes"
            }
            addNewNodes()
            saveIgraph()
            val refreshed = graph.vertexNames().mapIndexed { idx, name -> name to idx }.toMap()
            nodeNameToVertexIdx = refreshed
        }

        entityNodeIdxs =
            entityNodeKeys
                .mapNotNull { key ->
                    val idx = nodeNameToVertexIdx[key]
                    if (idx == null) {
                        logger.warn { "Missing entity node index for $key" }
                    }
                    idx
                }.toIntArray()
        passageNodeIdxs =
            passageNodeKeys
                .mapNotNull { key ->
                    val idx = nodeNameToVertexIdx[key]
                    if (idx == null) {
                        logger.warn { "Missing passage node index for $key" }
                    }
                    idx
                }.toIntArray()

        logger.info { "Loading embeddings." }
        entityEmbeddings = entityEmbeddingStore.getEmbeddings(entityNodeKeys)
        passageEmbeddings = chunkEmbeddingStore.getEmbeddings(passageNodeKeys)
        factEmbeddings = factEmbeddingStore.getEmbeddings(factNodeKeys)

        val (allOpenieInfo, _) = loadExistingOpenie(emptyList())

        procTriplesToDocs = mutableMapOf()
        for (doc in allOpenieInfo) {
            val triples = flattenFacts(listOf(doc.extractedTriples))
            for (triple in triples) {
                if (triple.size == 3) {
                    val procTriple = textProcessing(triple)
                    val key = procTriple.toString()
                    val existing = procTriplesToDocs[key] ?: mutableSetOf()
                    existing.add(doc.idx)
                    procTriplesToDocs[key] = existing
                }
            }
        }

        if (entNodeToChunkIds == null) {
            val (nerResults, tripleResults) = reformatOpenieResults(allOpenieInfo)

            if (!(passageNodeKeys.size == nerResults.size && passageNodeKeys.size == tripleResults.size)) {
                logger.warn {
                    "Length mismatch: passageNodeKeys=${passageNodeKeys.size}, " +
                        "nerResults=${nerResults.size}, tripleResults=${tripleResults.size}"
                }

                for (chunkId in passageNodeKeys) {
                    if (!nerResults.containsKey(chunkId)) {
                        nerResults[chunkId] =
                            NerRawOutput(
                                chunkId = chunkId,
                                response = null,
                                metadata = emptyMap(),
                                uniqueEntities = emptyList(),
                            )
                    }
                    if (!tripleResults.containsKey(chunkId)) {
                        tripleResults[chunkId] =
                            TripleRawOutput(
                                chunkId = chunkId,
                                response = null,
                                metadata = emptyMap(),
                                triples = emptyList(),
                            )
                    }
                }
            }

            val chunkTriples =
                passageNodeKeys.map { chunkId ->
                    tripleResults.getValue(chunkId).triples.map { textProcessing(it) }
                }

            nodeToNodeStats = mutableMapOf()
            entNodeToChunkIds = mutableMapOf()
            addFactEdges(passageNodeKeys, chunkTriples)
        }

        readyToRetrieve = true
    }

    @JvmName("getQueryEmbeddingsFromStrings")
    private fun getQueryEmbeddings(queries: List<String>) {
        val allQueryStrings =
            queries.filter {
                !queryToEmbedding.getValue("triple").containsKey(it) ||
                    !queryToEmbedding.getValue("passage").containsKey(it)
            }

        if (allQueryStrings.isNotEmpty()) {
            logger.info { "Encoding ${allQueryStrings.size} queries for query_to_fact." }
            val queryEmbeddingsForTriple =
                embeddingModel?.batchEncode(
                    allQueryStrings,
                    instruction = getQueryInstruction("query_to_fact"),
                    norm = true,
                ) ?: run {
                    logger.warn { "Embedding model is null; cannot encode queries. Retrieval will fail." }
                    emptyArray()
                }

            for ((query, embedding) in allQueryStrings.zip(queryEmbeddingsForTriple)) {
                queryToEmbedding.getValue("triple")[query] = embedding
            }

            logger.info { "Encoding ${allQueryStrings.size} queries for query_to_passage." }
            val queryEmbeddingsForPassage =
                embeddingModel?.batchEncode(
                    allQueryStrings,
                    instruction = getQueryInstruction("query_to_passage"),
                    norm = true,
                ) ?: run {
                    logger.warn { "Embedding model is null; cannot encode queries. Retrieval will fail." }
                    emptyArray()
                }

            for ((query, embedding) in allQueryStrings.zip(queryEmbeddingsForPassage)) {
                queryToEmbedding.getValue("passage")[query] = embedding
            }
        }
    }

    private fun getQueryEmbeddings(queries: List<QuerySolution>) {
        getQueryEmbeddings(queries.map { it.question })
    }

    private fun getFactScores(query: String): DoubleArray {
        val queryEmbedding =
            queryToEmbedding.getValue("triple")[query]
                ?: embeddingModel
                    ?.batchEncode(
                        listOf(query),
                        instruction = getQueryInstruction("query_to_fact"),
                        norm = true,
                    )?.firstOrNull()

        if (queryEmbedding == null) {
            return DoubleArray(0)
        }

        if (factEmbeddings.isEmpty()) {
            logger.warn { "No facts available for scoring. Returning empty array." }
            return DoubleArray(0)
        }

        if (factEmbeddings.any { it.size != queryEmbedding.size }) {
            logger.error { "Fact embedding dimension mismatch for query embedding size ${queryEmbedding.size}" }
            return DoubleArray(0)
        }

        val scores = dot(factEmbeddings, queryEmbedding)
        return minMaxNormalize(scores)
    }

    private fun densePassageRetrieval(query: String): Pair<List<Int>, DoubleArray> {
        val queryEmbedding =
            queryToEmbedding.getValue("passage")[query]
                ?: embeddingModel
                    ?.batchEncode(
                        listOf(query),
                        instruction = getQueryInstruction("query_to_passage"),
                        norm = true,
                    )?.firstOrNull()

        if (queryEmbedding == null) {
            return emptyList<Int>() to DoubleArray(0)
        }

        val scores = dot(passageEmbeddings, queryEmbedding)
        val normalized = minMaxNormalize(scores)
        val sortedDocIds = argsortDesc(normalized)
        val sortedScores = sortedDocIds.map { normalized[it] }.toDoubleArray()

        return sortedDocIds to sortedScores
    }

    private fun getTopKWeights(
        linkTopK: Int,
        allPhraseWeights: DoubleArray,
        linkingScoreMap: MutableMap<String, Double>,
    ): Pair<DoubleArray, MutableMap<String, Double>> {
        val sortedMap = linkingScoreMap.entries.sortedByDescending { it.value }.take(linkTopK)
        val filteredMap = sortedMap.associate { it.key to it.value }.toMutableMap()

        val topKPhrases = filteredMap.keys.toSet()
        val topKPhrasesKeys = topKPhrases.map { computeMdHashId(it, prefix = "entity-") }.toSet()

        nodeNameToVertexIdx.keys.forEach { phraseKey ->
            if (phraseKey !in topKPhrasesKeys) {
                val phraseId = nodeNameToVertexIdx[phraseKey]
                if (phraseId != null) {
                    allPhraseWeights[phraseId] = 0.0
                }
            }
        }

        val nonZeroCount = allPhraseWeights.count { it != 0.0 }
        if (nonZeroCount != filteredMap.size) {
            logger.warn {
                "Phrase weight count ($nonZeroCount) does not match filtered map size " +
                    "(${filteredMap.size}). This may be due to zero-weight phrases."
            }
        }
        return allPhraseWeights to filteredMap
    }

    private fun graphSearchWithFactEntities(
        query: String,
        linkTopK: Int,
        queryFactScores: DoubleArray,
        topKFacts: List<List<String>>,
        topKFactIndices: List<Int>,
        passageNodeWeight: Double = 0.05,
    ): Pair<List<Int>, DoubleArray> {
        val linkingScoreMap = mutableMapOf<String, Double>()
        val phraseScores = mutableMapOf<String, MutableList<Double>>()

        val phraseWeights = DoubleArray(graph.vcount())
        val passageWeights = DoubleArray(graph.vcount())
        val numberOfOccurs = DoubleArray(graph.vcount())

        val phrasesAndIds = mutableSetOf<Pair<String, Int?>>()

        for ((rank, fact) in topKFacts.withIndex()) {
            if (fact.size < 3) continue
            val subjectPhrase = fact[0].lowercase()
            val objectPhrase = fact[2].lowercase()
            val factScore = if (queryFactScores.isNotEmpty()) queryFactScores[topKFactIndices[rank]] else 0.0

            for (phrase in listOf(subjectPhrase, objectPhrase)) {
                val phraseKey = computeMdHashId(phrase, prefix = "entity-")
                val phraseId = nodeNameToVertexIdx[phraseKey]

                if (phraseId != null) {
                    var weightedFactScore = factScore
                    val docCount = entNodeToChunkIds?.get(phraseKey)?.size ?: 0
                    if (docCount > 0) {
                        weightedFactScore /= docCount.toDouble()
                    }

                    phraseWeights[phraseId] += weightedFactScore
                    numberOfOccurs[phraseId] += 1.0
                }

                phrasesAndIds.add(phrase to phraseId)
            }
        }

        for (i in phraseWeights.indices) {
            if (numberOfOccurs[i] != 0.0) {
                phraseWeights[i] /= numberOfOccurs[i]
            }
        }

        for ((phrase, phraseId) in phrasesAndIds) {
            if (!phraseScores.containsKey(phrase)) {
                phraseScores[phrase] = mutableListOf()
            }
            if (phraseId != null) {
                phraseScores.getValue(phrase).add(phraseWeights[phraseId])
            }
        }

        for ((phrase, scores) in phraseScores) {
            linkingScoreMap[phrase] = scores.average()
        }

        var finalPhraseWeights = phraseWeights
        var finalLinkingScoreMap = linkingScoreMap

        if (linkTopK > 0) {
            val (newWeights, newMap) = getTopKWeights(linkTopK, finalPhraseWeights, finalLinkingScoreMap)
            finalPhraseWeights = newWeights
            finalLinkingScoreMap = newMap
        }

        val (dprSortedDocIds, dprSortedDocScores) = densePassageRetrieval(query)
        val normalizedDprSortedScores = minMaxNormalize(dprSortedDocScores)

        for ((i, dprSortedDocId) in dprSortedDocIds.withIndex()) {
            val passageNodeKey = passageNodeKeys[dprSortedDocId]
            val passageDprScore = normalizedDprSortedScores[i]
            val passageNodeId = nodeNameToVertexIdx.getValue(passageNodeKey)
            passageWeights[passageNodeId] = passageDprScore * passageNodeWeight
            val passageNodeText = chunkEmbeddingStore.getRow(passageNodeKey).content
            finalLinkingScoreMap[passageNodeText] = passageDprScore * passageNodeWeight
        }

        val nodeWeights =
            DoubleArray(finalPhraseWeights.size) { idx ->
                finalPhraseWeights[idx] + passageWeights[idx]
            }

        if (finalLinkingScoreMap.size > 30) {
            finalLinkingScoreMap =
                finalLinkingScoreMap.entries
                    .sortedByDescending { it.value }
                    .take(30)
                    .associate { it.key to it.value }
                    .toMutableMap()
        }

        if (nodeWeights.sum() <= 0.0) {
            logger.warn { "No phrases found in the graph for the given facts; falling back to DPR." }
            return dprSortedDocIds to dprSortedDocScores
        }

        logger.info {
            "Running PPR with ${nodeWeights.size} nodes and total weight " +
                String.format(Locale.US, "%.6f", nodeWeights.sum())
        }
        val pprStart = nowSeconds()
        val (pprSortedDocIds, pprSortedDocScores) = runPpr(nodeWeights, damping = globalConfig.damping)
        val pprEnd = nowSeconds()

        pprTimeSeconds += (pprEnd - pprStart)

        check(pprSortedDocIds.size == passageNodeIdxs.size) {
            "Doc prob length ${pprSortedDocIds.size} != corpus length ${passageNodeIdxs.size}"
        }

        return pprSortedDocIds to pprSortedDocScores
    }

    private fun rerankFacts(
        query: String,
        queryFactScores: DoubleArray,
    ): Triple<List<Int>, List<List<String>>, Map<String, Any?>> {
        val linkTopK = globalConfig.linkingTopK

        if (queryFactScores.isEmpty() || factNodeKeys.isEmpty()) {
            logger.warn { "No facts available for reranking. Returning empty lists." }
            return Triple(
                emptyList(),
                emptyList(),
                mapOf(
                    "facts_before_rerank" to emptyList<List<String>>(),
                    "facts_after_rerank" to emptyList<List<String>>(),
                ),
            )
        }

        return runCatching {
            val candidateFactIndices =
                if (queryFactScores.size <= linkTopK) {
                    argsortDesc(queryFactScores)
                } else {
                    argsortDesc(queryFactScores).take(linkTopK)
                }

            val realCandidateFactIds = candidateFactIndices.map { factNodeKeys[it] }
            val factRowDict = factEmbeddingStore.getRows(realCandidateFactIds)
            val candidateFacts =
                realCandidateFactIds.map { id ->
                    parseTripleString(factRowDict.getValue(id).content)
                }

            val (topKFactIndices, topKFacts, rerankerDict) =
                rerankFilter.rerank(
                    query,
                    candidateFacts,
                    candidateFactIndices,
                    lenAfterRerank = linkTopK,
                )

            val rerankLog =
                mapOf(
                    "facts_before_rerank" to candidateFacts,
                    "facts_after_rerank" to topKFacts,
                ) + rerankerDict

            Triple(topKFactIndices, topKFacts, rerankLog)
        }.getOrElse { e ->
            logger.error(e) { "Error in rerankFacts" }
            Triple(
                emptyList(),
                emptyList(),
                mapOf(
                    "facts_before_rerank" to emptyList<List<String>>(),
                    "facts_after_rerank" to emptyList<List<String>>(),
                    "error" to e.message.orEmpty(),
                ),
            )
        }
    }

    private fun runPpr(
        resetProb: DoubleArray,
        damping: Double = 0.5,
    ): Pair<List<Int>, DoubleArray> {
        val cleanedReset = resetProb.map { if (it.isNaN() || it < 0.0) 0.0 else it }.toDoubleArray()
        val pagerankScores = graph.personalizedPageRank(cleanedReset, damping)

        val docScores = passageNodeIdxs.map { pagerankScores[it] }.toDoubleArray()
        val sortedDocIds = argsortDesc(docScores)
        val sortedDocScores = sortedDocIds.map { docScores[it] }.toDoubleArray()

        return sortedDocIds to sortedDocScores
    }

    private fun nowSeconds(): Double = System.nanoTime() / 1_000_000_000.0

    private fun round4(value: Double): Double = String.format(Locale.US, "%.4f", value).toDouble()

    private fun dot(
        matrix: Array<DoubleArray>,
        vector: DoubleArray,
    ): DoubleArray {
        val result = DoubleArray(matrix.size)
        for (i in matrix.indices) {
            val row = matrix[i]
            require(row.size == vector.size) {
                "Matrix row $i size (${row.size}) does not match vector size (${vector.size})"
            }
            var sum = 0.0
            for (j in row.indices) {
                sum += row[j] * vector[j]
            }
            result[i] = sum
        }
        return result
    }

    private fun argsortDesc(x: DoubleArray): List<Int> = x.indices.sortedByDescending { x[it] }
}
