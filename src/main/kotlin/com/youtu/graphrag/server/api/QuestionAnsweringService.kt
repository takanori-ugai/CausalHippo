package com.youtu.graphrag.server.api

import com.youtu.graphrag.server.api.contracts.QuestionResponse
import com.youtu.graphrag.server.api.contracts.ReasoningStep
import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.decomposer.GraphQ
import com.youtu.graphrag.shared.llm.LlmClient
import com.youtu.graphrag.shared.llm.LlmClientFactory
import com.youtu.graphrag.shared.retriever.IrcotPromptSource
import com.youtu.graphrag.shared.retriever.KTRetriever
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.system.measureTimeMillis

class GraphArtifactNotFoundException(
    message: String,
) : RuntimeException(message)

data class QaStageUpdate(
    val stage: String,
    val payload: Map<String, Any?> = emptyMap(),
)

private data class GraphResolution(
    val graphPath: Path,
    val usedDemoFallback: Boolean,
)

private data class SubQuestionRetrieval(
    val index: Int,
    val subQuestionText: String,
    val triples: List<String>,
    val triplesFormatted: List<String>,
    val chunkIds: List<String>,
    val chunkContents: Map<String, String>,
    val processingTime: Double,
)

private data class TripleEntry(
    val raw: String,
    val formatted: String,
)

class QuestionAnsweringService(
    private val config: ConfigManager,
    private val rootDir: Path = Path.of("."),
    private val llmClient: LlmClient = LlmClientFactory.fromEnvironment(),
    private val ircotPromptSource: IrcotPromptSource = IrcotPromptSource.MAIN,
) {
    private val logger = KotlinLogging.logger {}

    private fun requireSafeDatasetName(datasetName: String): String {
        require(Regex("^[A-Za-z0-9_-]+$").matches(datasetName)) {
            "Invalid dataset name"
        }
        return datasetName
    }

    suspend fun answerQuestion(
        datasetName: String,
        question: String,
        onQaUpdate: suspend (QaStageUpdate) -> Unit = {},
    ): QuestionResponse {
        val safeDatasetName = requireSafeDatasetName(datasetName)
        val graphResolution =
            resolveGraphPathWithDemoFallback(safeDatasetName)
                ?: throw GraphArtifactNotFoundException("Graph not found. Please construct graph first.")
        val graphPath = graphResolution.graphPath
        val schemaPath = resolveSchemaPath(safeDatasetName)

        val graphq = GraphQ(safeDatasetName, config, llmClient = llmClient)
        val ktRetriever =
            KTRetriever.createAndBuild(
                datasetName = safeDatasetName,
                graphPath = graphPath.toString(),
                recallPaths = config.retrieval.recallPaths,
                schemaPath = schemaPath.toString(),
                topK = config.retrieval.topKFilter,
                mode = config.triggers.mode,
                config = config,
                rootDir = rootDir,
            )

        val decomposition =
            withContext(Dispatchers.IO) {
                graphq.decompose(question, schemaPath.toString())
            }
        val subQuestions =
            parseSubQuestions(decomposition["sub_questions"])
                .ifEmpty { listOf(mapOf("sub-question" to question)) }
        val involvedTypes = parseInvolvedTypes(decomposition["involved_types"])
        val parallelSubquestions = shouldProcessSubquestionsInParallel(subQuestions.size)

        onQaUpdate(
            QaStageUpdate(
                stage = "decompose",
                payload =
                    mapOf(
                        "sub_questions_count" to subQuestions.size,
                        "sub_questions" to subQuestions.map { it["sub-question"].orEmpty() }.take(5),
                        "parallel_subquestions" to parallelSubquestions,
                    ),
            ),
        )

        val tripleEntriesByRaw = linkedMapOf<String, String>()
        val chunkById = mutableMapOf<String, String>()

        val reasoningSteps = mutableListOf<ReasoningStep>()

        val subQuestionRetrievals =
            retrieveSubQuestionResults(
                subQuestions = subQuestions,
                originalQuestion = question,
                involvedTypes = involvedTypes,
                ktRetriever = ktRetriever,
                parallelSubquestions = parallelSubquestions,
            )

        val mergedSubQuestionTriples = mutableListOf<TripleEntry>()
        subQuestionRetrievals.forEach { retrieval ->
            mergedSubQuestionTriples += zipTripleEntries(retrieval.triples, retrieval.triplesFormatted)
            retrieval.chunkIds.forEach { chunkId ->
                val chunkContent = retrieval.chunkContents[chunkId]
                if (chunkContent != null) {
                    chunkById[chunkId] = chunkContent
                }
            }

            val chunkPreview = retrieval.chunkIds.mapNotNull { retrieval.chunkContents[it] }.take(3)

            val step =
                ReasoningStep(
                    type = "sub_question",
                    question = retrieval.subQuestionText,
                    triples = retrieval.triplesFormatted.take(10),
                    triplesCount = retrieval.triples.size,
                    chunkContents = chunkPreview,
                    chunksCount = retrieval.chunkIds.size,
                    processingTime = retrieval.processingTime,
                )
            reasoningSteps.add(step)

            onQaUpdate(
                QaStageUpdate(
                    stage = "sub_question",
                    payload =
                        mapOf(
                            "index" to (retrieval.index + 1),
                            "total" to subQuestions.size,
                            "question" to retrieval.subQuestionText,
                            "triples_preview" to retrieval.triplesFormatted.distinct().take(5),
                            "triples_count" to step.triplesCount,
                            "chunks_count" to step.chunksCount,
                            "processing_time" to step.processingTime,
                        ),
                ),
            )
        }

        val topLimit = config.retrieval.topKFilter.coerceAtLeast(1)
        val initialTripleEntries = selectTopTriplesForQuestion(mergedSubQuestionTriples, question, topLimit)
        initialTripleEntries.forEach { entry ->
            tripleEntriesByRaw.putIfAbsent(entry.raw, entry.formatted)
        }

        val initialChunkIds = selectTopChunkIdsForQuestion(chunkById, question, topLimit)
        val trimmedChunkById = linkedMapOf<String, String>()
        initialChunkIds.forEach { chunkId ->
            val content = chunkById[chunkId]
            if (content != null) {
                trimmedChunkById[chunkId] = content
            }
        }
        chunkById.clear()
        chunkById.putAll(trimmedChunkById)

        val initialTriplesFormatted = tripleEntriesByRaw.values.toList()
        val initialChunks = chunkById.values.toList()
        val initialContext =
            ktRetriever.buildKnowledgeContext(
                triples = initialTriplesFormatted,
                chunks = initialChunks,
                tripleLimit = Int.MAX_VALUE,
                chunkLimit = Int.MAX_VALUE,
            )
        val initialAnswer =
            generateAnswerWithFallback(
                prompt = ktRetriever.generatePrompt(question, initialContext),
                fallback = synthesizeAnswer(tripleEntriesByRaw.keys.toList(), initialChunks),
            )
        var finalAnswer = initialAnswer

        reasoningSteps.add(
            ReasoningStep(
                type = "initial_answer",
                question = question,
                triples = initialTriplesFormatted.take(10),
                triplesCount = tripleEntriesByRaw.size,
                chunkContents = initialChunks.take(3),
                chunksCount = initialChunks.size,
                thought = initialAnswer.take(300),
            ),
        )

        if (shouldRunIrcot()) {
            onQaUpdate(
                QaStageUpdate(
                    stage = "ircot_start",
                    payload = mapOf("message" to "Starting iterative reasoning"),
                ),
            )

            val thoughts = mutableListOf("Initial analysis (noagent mode): $initialAnswer")
            var currentQuery = question
            val maxSteps =
                config.retrieval.agent.maxSteps
                    .coerceAtLeast(1)

            for (stepIndex in 1..maxSteps) {
                var thought = ""
                val triplesSnapshotBefore = tripleEntriesByRaw.values.toList()
                val chunksSnapshotBefore = chunkById.values.toList()

                val stepElapsedMs =
                    measureTimeMillis {
                        val loopContext =
                            ktRetriever.buildKnowledgeContext(
                                triples = triplesSnapshotBefore,
                                chunks = chunksSnapshotBefore,
                                tripleLimit = Int.MAX_VALUE,
                                chunkLimit = Int.MAX_VALUE,
                            )
                        val previousThoughts = thoughts.joinToString(" | ").ifBlank { "None" }
                        val ircotPrompt =
                            ktRetriever.generateIrcotPrompt(
                                currentQuery = currentQuery,
                                originalQuestion = question,
                                context = loopContext,
                                previousThoughts = previousThoughts,
                                step = stepIndex,
                                promptSource = ircotPromptSource,
                            )
                        val heuristicFallback =
                            if (tripleEntriesByRaw.isNotEmpty() || chunksSnapshotBefore.isNotEmpty()) {
                                "So the answer is: ${synthesizeAnswer(tripleEntriesByRaw.keys.toList(), chunksSnapshotBefore)}"
                            } else {
                                "The new query is: ${buildFollowUpQuery(question, currentQuery, thoughts)}"
                            }
                        thought =
                            withContext(Dispatchers.IO) {
                                llmClient.complete(ircotPrompt)
                            }.trim()
                                .ifBlank { heuristicFallback }
                    }

                thoughts.add(thought)

                val triplesSnapshotAfter = tripleEntriesByRaw.values.toList()
                val chunksSnapshotAfter = chunkById.values.toList()

                val ircotStep =
                    ReasoningStep(
                        type = "ircot_step",
                        question = currentQuery,
                        triples = triplesSnapshotAfter.take(10),
                        triplesCount = triplesSnapshotAfter.size,
                        chunkContents = chunksSnapshotAfter.take(3),
                        chunksCount = chunksSnapshotAfter.size,
                        processingTime = stepElapsedMs / 1000.0,
                        thought = thought.take(300),
                    )
                reasoningSteps.add(ircotStep)

                onQaUpdate(
                    QaStageUpdate(
                        stage = "ircot",
                        payload =
                            mapOf(
                                "step" to stepIndex,
                                "max_steps" to maxSteps,
                                "current_query" to currentQuery,
                                "thought_preview" to thought.take(200),
                            ),
                    ),
                )

                extractFinalAnswer(thought)?.let { answer ->
                    finalAnswer = answer
                    break
                }

                if (!containsNewQueryMarker(thought)) {
                    finalAnswer = finalAnswer.ifBlank { thought }
                    break
                }

                val newQuery = extractNewQuery(thought)
                if (newQuery.isNullOrBlank() || isRepeatedQuery(newQuery, currentQuery)) {
                    finalAnswer = finalAnswer.ifBlank { thought }
                    break
                }

                currentQuery = newQuery
                val (additionalRetrieval, _) =
                    ktRetriever.processRetrievalResults(
                        question = newQuery,
                    )
                mergeRetrieval(
                    retrievalResults = additionalRetrieval,
                    tripleEntriesByRaw = tripleEntriesByRaw,
                    chunkById = chunkById,
                )
            }
        }

        val finalTriples = tripleEntriesByRaw.keys.toList()
        val finalTriplesFormatted = tripleEntriesByRaw.values.toList()
        val finalChunks = chunkById.values.toList()
        val responseTriples =
            if (graphResolution.usedDemoFallback) {
                emptyList()
            } else {
                finalTriples
            }

        val finalContext = buildFinalKnowledgeContext(finalTriplesFormatted, finalChunks)
        finalAnswer =
            generateAnswerWithFallback(
                prompt = ktRetriever.generatePrompt(question, finalContext),
                fallback = finalAnswer.ifBlank { synthesizeAnswer(finalTriples, finalChunks) },
            )

        logger.info {
            "Answered question for dataset '$datasetName' with ${finalTriples.size} triples and ${finalChunks.size} chunks"
        }

        return QuestionResponse(
            answer = finalAnswer,
            subQuestions = subQuestions,
            retrievedTriples = responseTriples,
            retrievedChunks = finalChunks,
            reasoningSteps = reasoningSteps,
            visualizationData =
                buildVisualizationData(
                    subQuestions = subQuestions,
                    reasoningSteps = reasoningSteps,
                    finalTriples = responseTriples,
                    finalChunks = finalChunks,
                ),
        )
    }

    private fun shouldRunIrcot(): Boolean = config.triggers.mode == "agent" && config.retrieval.agent.enableIrcot

    private suspend fun generateAnswerWithFallback(
        prompt: String,
        fallback: String,
    ): String {
        val maxRetries = 20
        repeat(maxRetries) {
            val response =
                runCatching {
                    withContext(Dispatchers.IO) { llmClient.complete(prompt) }
                }.getOrDefault("")
                    .trim()
            if (response.isNotBlank()) {
                return response
            }
        }
        return fallback
    }

    private fun extractFinalAnswer(thought: String): String? {
        val marker = FINAL_ANSWER_MARKER
        val index = thought.indexOf(marker, ignoreCase = true)
        if (index < 0) {
            return null
        }

        val tail = thought.substring(index + marker.length)
        val newQueryIndex = tail.indexOf(NEW_QUERY_MARKER, ignoreCase = true)
        val answerBody =
            if (newQueryIndex >= 0) {
                tail.substring(0, newQueryIndex)
            } else {
                tail
            }

        val normalized =
            answerBody
                .lineSequence()
                .map { line -> line.trim() }
                .filter { line -> line.isNotBlank() }
                .joinToString(" ")
                .trim('"', '\'')
                .trim()

        return normalized.ifBlank { null }
    }

    private fun containsNewQueryMarker(thought: String): Boolean = thought.contains(NEW_QUERY_MARKER, ignoreCase = true)

    private fun extractNewQuery(thought: String): String? {
        val marker = NEW_QUERY_MARKER
        val index = thought.indexOf(marker, ignoreCase = true)
        if (index < 0) {
            return null
        }

        val tail =
            thought
                .substring(index + marker.length)
                .replace("`", "")

        val normalized =
            tail
                .lineSequence()
                .map { line -> line.trim() }
                .firstOrNull { line -> line.isNotBlank() }
                ?.trim('"', '\'')
                .orEmpty()
                .trim()

        return normalized.ifBlank { null }
    }

    private fun isRepeatedQuery(
        candidate: String,
        currentQuery: String,
    ): Boolean = normalizeQuery(candidate) == normalizeQuery(currentQuery)

    private fun normalizeQuery(query: String): String =
        query
            .lowercase()
            .replace(Regex("\\s+"), " ")
            .trim()
            .trimEnd('.', '?', '!', ';', ':')

    private fun shouldProcessSubquestionsInParallel(subQuestionCount: Int): Boolean =
        subQuestionCount > 1 && config.retrieval.agent.enableParallelSubquestions

    private suspend fun retrieveSubQuestionResults(
        subQuestions: List<Map<String, String>>,
        originalQuestion: String,
        involvedTypes: Map<String, List<String>>,
        ktRetriever: KTRetriever,
        parallelSubquestions: Boolean,
    ): List<SubQuestionRetrieval> {
        if (!parallelSubquestions) {
            return subQuestions.mapIndexed { index, subQuestion ->
                retrieveSingleSubQuestion(
                    index = index,
                    subQuestion = subQuestion,
                    originalQuestion = originalQuestion,
                    involvedTypes = involvedTypes,
                    ktRetriever = ktRetriever,
                )
            }
        }

        return coroutineScope {
            subQuestions
                .mapIndexed { index, subQuestion ->
                    async(Dispatchers.Default) {
                        retrieveSingleSubQuestion(
                            index = index,
                            subQuestion = subQuestion,
                            originalQuestion = originalQuestion,
                            involvedTypes = involvedTypes,
                            ktRetriever = ktRetriever,
                        )
                    }
                }.awaitAll()
                .sortedBy { retrieval -> retrieval.index }
        }
    }

    private fun retrieveSingleSubQuestion(
        index: Int,
        subQuestion: Map<String, String>,
        originalQuestion: String,
        involvedTypes: Map<String, List<String>>,
        ktRetriever: KTRetriever,
    ): SubQuestionRetrieval {
        val subQuestionText = subQuestion["sub-question"].orEmpty().ifBlank { originalQuestion }
        val (retrievalResults, retrievalTime) =
            ktRetriever.processRetrievalResults(
                question = subQuestionText,
                involvedTypes = involvedTypes,
            )
        val triples = parseStringList(retrievalResults["triples"])
        val chunkIds = parseStringList(retrievalResults["chunk_ids"])

        return SubQuestionRetrieval(
            index = index,
            subQuestionText = subQuestionText,
            triples = triples,
            triplesFormatted = triples,
            chunkIds = chunkIds,
            chunkContents =
                parseChunkContentMap(
                    chunkIds = chunkIds,
                    chunkContentsRaw = retrievalResults["chunk_contents"],
                    chunkContentsByIdRaw = null,
                ),
            processingTime = retrievalTime,
        )
    }

    private fun mergeRetrieval(
        retrievalResults: Map<String, Any>,
        tripleEntriesByRaw: MutableMap<String, String>,
        chunkById: MutableMap<String, String>,
    ) {
        val triples = parseStringList(retrievalResults["triples"])
        val triplesFormatted = parseStringList(retrievalResults["triples_formatted"]).ifEmpty { triples }
        val chunkIds = parseStringList(retrievalResults["chunk_ids"])
        val chunkContents =
            parseChunkContentMap(
                chunkIds = chunkIds,
                chunkContentsRaw = retrievalResults["chunk_contents"],
                chunkContentsByIdRaw = null,
            )

        zipTripleEntries(triples, triplesFormatted).forEach { entry ->
            tripleEntriesByRaw.putIfAbsent(entry.raw, entry.formatted)
        }
        chunkIds.forEach { chunkId ->
            val content = chunkContents[chunkId]
            if (content != null) {
                chunkById[chunkId] = content
            }
        }
    }

    private fun synthesizeAnswer(
        triples: List<String>,
        chunks: List<String>,
    ): String {
        if (chunks.isNotEmpty()) {
            return chunks.first().take(400)
        }

        if (triples.isNotEmpty()) {
            return triples.first()
        }

        return "I do not have enough knowledge context to answer this question."
    }

    private fun buildFinalKnowledgeContext(
        triples: List<String>,
        chunks: List<String>,
    ): String =
        buildString {
            append("=== Final Triples ===\n")
            append(triples.joinToString("\n"))
            append("\n=== Final Chunks ===\n")
            append(chunks.joinToString("\n"))
        }

    private fun buildFollowUpQuery(
        originalQuestion: String,
        currentQuery: String,
        thoughts: List<String>,
    ): String {
        val lastThought = thoughts.lastOrNull().orEmpty()
        val tokens = tokenize(originalQuestion).filter { token -> token !in tokenize(lastThought) }
        val refinement = tokens.take(5).joinToString(" ")

        if (refinement.isBlank()) {
            return currentQuery
        }

        return "$currentQuery $refinement".trim()
    }

    private fun tokenize(input: String): Set<String> {
        val tokenRegex = Regex("[\\p{L}\\p{N}_]{2,}")
        return tokenRegex.findAll(input.lowercase()).map { it.value }.toSet()
    }

    private fun zipTripleEntries(
        triples: List<String>,
        triplesFormatted: List<String>,
    ): List<TripleEntry> {
        if (triples.isEmpty()) {
            return emptyList()
        }

        return triples.mapIndexedNotNull { index, raw ->
            val rawNormalized = raw.trim()
            if (rawNormalized.isBlank()) {
                return@mapIndexedNotNull null
            }
            val formatted =
                triplesFormatted
                    .getOrNull(index)
                    ?.trim()
                    ?.takeIf { value -> value.isNotBlank() }
                    ?: rawNormalized
            TripleEntry(raw = rawNormalized, formatted = formatted)
        }
    }

    private fun selectTopTriplesForQuestion(
        entries: List<TripleEntry>,
        question: String,
        topLimit: Int,
    ): List<TripleEntry> {
        if (entries.isEmpty()) {
            return emptyList()
        }
        if (entries.size <= topLimit) {
            return entries
        }

        val keywords =
            question
                .lowercase()
                .split(Regex("\\s+"))
                .filter { it.isNotBlank() }
                .toSet()
        val scored =
            entries.mapIndexed { index, entry ->
                val tripleLower = entry.formatted.lowercase()
                val score = keywords.count { keyword -> keyword in tripleLower }
                Triple(entry, score, index)
            }

        return scored
            .sortedWith(compareByDescending<Triple<TripleEntry, Int, Int>> { it.second }.thenBy { it.third })
            .take(topLimit)
            .map { it.first }
    }

    private fun selectTopChunkIdsForQuestion(
        chunksById: Map<String, String>,
        question: String,
        topLimit: Int,
    ): List<String> {
        if (chunksById.isEmpty()) {
            return emptyList()
        }
        val orderedIds = chunksById.keys.toList()
        if (orderedIds.size <= topLimit) {
            return orderedIds
        }

        val keywords =
            question
                .lowercase()
                .split(Regex("\\s+"))
                .filter { it.isNotBlank() }
                .toSet()
        val scored =
            orderedIds.mapIndexed { index, chunkId ->
                val chunkLower = chunksById[chunkId].orEmpty().lowercase()
                val score = keywords.count { keyword -> keyword in chunkLower }
                Triple(chunkId, score, index)
            }

        return scored
            .sortedWith(compareByDescending<Triple<String, Int, Int>> { it.second }.thenBy { it.third })
            .take(topLimit)
            .map { it.first }
    }

    private fun buildVisualizationData(
        subQuestions: List<Map<String, String>>,
        reasoningSteps: List<ReasoningStep>,
        finalTriples: List<String>,
        finalChunks: List<String>,
    ): JsonObject {
        val subqueryGraph = buildSubqueryVisualization(subQuestions)
        val retrievedGraph = buildRetrievedGraphVisualization(finalTriples)
        val reasoningFlow = buildReasoningFlowVisualization(reasoningSteps)

        val triplesBySubquery = reasoningSteps.filter { it.type == "sub_question" }.map { it.triplesCount }

        return buildJsonObject {
            put("subqueries", subqueryGraph)
            put("knowledge_graph", retrievedGraph)
            put("reasoning_flow", reasoningFlow)
            put(
                "retrieval_details",
                buildJsonObject {
                    put("total_triples", JsonPrimitive(finalTriples.size))
                    put("total_chunks", JsonPrimitive(finalChunks.size))
                    put("sub_questions_count", JsonPrimitive(subQuestions.size))
                    put(
                        "triples_by_subquery",
                        buildJsonArray {
                            triplesBySubquery.forEach { count -> add(JsonPrimitive(count)) }
                        },
                    )
                },
            )
        }
    }

    private fun buildSubqueryVisualization(subQuestions: List<Map<String, String>>): JsonObject {
        val nodes =
            buildJsonArray {
                add(
                    buildJsonObject {
                        put("id", JsonPrimitive("original"))
                        put("name", JsonPrimitive("Original Question"))
                        put("category", JsonPrimitive("question"))
                        put("symbolSize", JsonPrimitive(40))
                    },
                )
                subQuestions.forEachIndexed { index, subQuestion ->
                    val subText = subQuestion["sub-question"].orEmpty()
                    add(
                        buildJsonObject {
                            put("id", JsonPrimitive("sub_$index"))
                            put("name", JsonPrimitive(subText.take(20) + if (subText.length > 20) "..." else ""))
                            put("category", JsonPrimitive("sub_question"))
                            put("symbolSize", JsonPrimitive(30))
                        },
                    )
                }
            }

        val links =
            buildJsonArray {
                subQuestions.forEachIndexed { index, _ ->
                    add(
                        buildJsonObject {
                            put("source", JsonPrimitive("original"))
                            put("target", JsonPrimitive("sub_$index"))
                            put("name", JsonPrimitive("decomposed to"))
                        },
                    )
                }
            }

        return buildJsonObject {
            put("nodes", nodes)
            put("links", links)
            put(
                "categories",
                buildJsonArray {
                    add(buildJsonObject { put("name", JsonPrimitive("question")) })
                    add(buildJsonObject { put("name", JsonPrimitive("sub_question")) })
                },
            )
        }
    }

    private fun buildRetrievedGraphVisualization(triples: List<String>): JsonObject {
        val nodes = mutableListOf<JsonObject>()
        val links = mutableListOf<JsonObject>()
        val seenNodes = mutableSetOf<String>()

        triples.take(10).forEach { tripleText ->
            val parsed = parseTriple(tripleText) ?: return@forEach
            val (source, relation, target) = parsed

            if (seenNodes.add(source)) {
                nodes.add(
                    buildJsonObject {
                        put("id", JsonPrimitive(source))
                        put("name", JsonPrimitive(source.take(20)))
                        put("category", JsonPrimitive("entity"))
                        put("symbolSize", JsonPrimitive(20))
                    },
                )
            }
            if (seenNodes.add(target)) {
                nodes.add(
                    buildJsonObject {
                        put("id", JsonPrimitive(target))
                        put("name", JsonPrimitive(target.take(20)))
                        put("category", JsonPrimitive("entity"))
                        put("symbolSize", JsonPrimitive(20))
                    },
                )
            }

            links.add(
                buildJsonObject {
                    put("source", JsonPrimitive(source))
                    put("target", JsonPrimitive(target))
                    put("name", JsonPrimitive(relation))
                },
            )
        }

        return buildJsonObject {
            put("nodes", JsonArray(nodes))
            put("links", JsonArray(links))
            put(
                "categories",
                buildJsonArray {
                    add(buildJsonObject { put("name", JsonPrimitive("entity")) })
                },
            )
        }
    }

    private fun buildReasoningFlowVisualization(reasoningSteps: List<ReasoningStep>): JsonObject {
        val stepsJson =
            buildJsonArray {
                reasoningSteps.forEachIndexed { index, step ->
                    add(
                        buildJsonObject {
                            put("step", JsonPrimitive(index + 1))
                            put("type", JsonPrimitive(step.type))
                            put("question", JsonPrimitive(step.question.take(50)))
                            put("triples_count", JsonPrimitive(step.triplesCount))
                            put("chunks_count", JsonPrimitive(step.chunksCount))
                            put("processing_time", JsonPrimitive(step.processingTime))
                        },
                    )
                }
            }

        val timelineJson =
            buildJsonArray {
                reasoningSteps.forEach { step -> add(JsonPrimitive(step.processingTime)) }
            }

        return buildJsonObject {
            put("steps", stepsJson)
            put("timeline", timelineJson)
        }
    }

    private fun parseTriple(triple: String): Triple<String, String, String>? {
        val quotedParts = Regex("\"([^\"]*)\"").findAll(triple).map { it.groupValues[1] }.toList()
        if (quotedParts.size == 3) {
            return Triple(quotedParts[0], quotedParts[1], quotedParts[2])
        }

        if (!triple.startsWith("[") || !triple.endsWith("]")) {
            return null
        }

        val rawParts =
            triple
                .removePrefix("[")
                .removeSuffix("]")
                .split(",")
                .map { it.trim().trim('"', '\'', ' ') }
                .filter { it.isNotBlank() }

        return if (rawParts.size == 3) {
            Triple(rawParts[0], rawParts[1], rawParts[2])
        } else {
            null
        }
    }

    private fun resolveGraphPathWithDemoFallback(datasetName: String): GraphResolution? {
        val datasetGraph = resolvePath("${config.output.graphsDir}/${datasetName}_new.json")
        if (datasetGraph.exists()) {
            return GraphResolution(
                graphPath = datasetGraph,
                usedDemoFallback = false,
            )
        }

        val demoGraph = resolvePath("${config.output.graphsDir}/demo_new.json")
        if (demoGraph.exists()) {
            return GraphResolution(
                graphPath = demoGraph,
                usedDemoFallback = true,
            )
        }

        return null
    }

    private fun resolveSchemaPath(datasetName: String): Path {
        val datasetSchema = resolvePath("schemas/$datasetName.json")
        if (datasetName != "demo" && datasetSchema.exists()) {
            return datasetSchema
        }

        return resolvePath("schemas/demo.json")
    }

    private fun resolvePath(path: String): Path {
        val candidate = Path.of(path)
        return if (candidate.isAbsolute) candidate else rootDir.resolve(candidate)
    }

    private fun parseSubQuestions(raw: Any?): List<Map<String, String>> {
        if (raw !is List<*>) {
            return emptyList()
        }

        return raw.mapNotNull { item ->
            if (item !is Map<*, *>) {
                return@mapNotNull null
            }

            mapOf("sub-question" to (item["sub-question"]?.toString().orEmpty()))
        }
    }

    private fun parseInvolvedTypes(raw: Any?): Map<String, List<String>> {
        if (raw !is Map<*, *>) {
            return emptyMap()
        }

        return raw.entries.associate { (key, value) ->
            key.toString() to parseStringList(value)
        }
    }

    private fun parseStringList(raw: Any?): List<String> {
        if (raw !is List<*>) {
            return emptyList()
        }

        return raw.mapNotNull { it?.toString() }
    }

    private fun parseChunkContentMap(raw: Any?): Map<String, String> =
        parseChunkContentMap(
            chunkIds = emptyList(),
            chunkContentsRaw = raw,
            chunkContentsByIdRaw = null,
        )

    private fun parseChunkContentMap(
        chunkIds: List<String>,
        chunkContentsRaw: Any?,
        chunkContentsByIdRaw: Any?,
    ): Map<String, String> {
        val byId = parseStringMap(chunkContentsByIdRaw)
        if (byId.isNotEmpty()) {
            val ordered = linkedMapOf<String, String>()
            chunkIds.forEach { chunkId ->
                ordered[chunkId] = byId[chunkId] ?: "[Missing content for chunk $chunkId]"
            }
            byId.forEach { (chunkId, content) ->
                if (chunkId !in ordered) {
                    ordered[chunkId] = content
                }
            }
            return ordered
        }

        when (chunkContentsRaw) {
            is Map<*, *> -> {
                val map = parseStringMap(chunkContentsRaw)
                if (chunkIds.isEmpty()) {
                    return map
                }
                val ordered = linkedMapOf<String, String>()
                chunkIds.forEach { chunkId ->
                    ordered[chunkId] = map[chunkId] ?: "[Missing content for chunk $chunkId]"
                }
                map.forEach { (chunkId, content) ->
                    if (chunkId !in ordered) {
                        ordered[chunkId] = content
                    }
                }
                return ordered
            }

            is List<*> -> {
                if (chunkIds.isEmpty()) {
                    return emptyMap()
                }
                val contentValues = chunkContentsRaw.map { value -> value?.toString().orEmpty() }
                return chunkIds
                    .mapIndexed { index, chunkId ->
                        val content = contentValues.getOrNull(index)?.takeIf { it.isNotBlank() } ?: "[Missing content for chunk $chunkId]"
                        chunkId to content
                    }.toMap(linkedMapOf())
            }

            else -> {
                return emptyMap()
            }
        }
    }

    private fun parseStringMap(raw: Any?): Map<String, String> {
        if (raw !is Map<*, *>) {
            return emptyMap()
        }
        return raw.entries
            .filter { (key, value) -> key != null && value != null }
            .associate { (key, value) -> key.toString() to value.toString() }
    }

    companion object {
        private const val FINAL_ANSWER_MARKER = "So the answer is:"
        private const val NEW_QUERY_MARKER = "The new query is:"
    }
}
