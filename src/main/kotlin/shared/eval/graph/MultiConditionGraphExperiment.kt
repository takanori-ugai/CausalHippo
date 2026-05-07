package shared.eval.graph

import causalrag.examples.CliUtils
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList
import com.microsoft.graphrag.index.GraphRagConfig
import com.microsoft.graphrag.index.NoopWorkflowCallbacks
import com.microsoft.graphrag.index.defaultEmbeddingModel
import com.microsoft.graphrag.index.defaultPipeline
import com.microsoft.graphrag.index.runPipeline
import com.microsoft.graphrag.query.BasicQueryEngine
import com.microsoft.graphrag.query.QueryIndexLoader
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import io.github.ugaikit.bertscore.BertScore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.llm.LLMFactory
import lightrag.operate.removeThinkTags
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager
import pathrag.PathRAG
import pathrag.eval.RagasContextExtractor
import ragas.metrics.collections.ResponseGroundednessMetric
import ragas.metrics.defaults.FaithfulnessMetric
import ragas.model.SingleTurnSample
import shared.config.CommonRagConfig
import shared.config.CommonRagConfigLoader
import shared.rag.unified.RagId
import shared.rag.unified.UnifiedMode
import shared.rag.unified.UnifiedQuery
import shared.rag.unified.UnifiedRagFactory
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant
import java.util.Comparator
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.io.path.createDirectories
import kotlin.math.log2
import kotlin.math.max
import lightrag.core.QueryParam as LightRagQueryParam
import pathrag.base.QueryParam as PathRagQueryParam

private val json = Json { ignoreUnknownKeys = true }
private val NON_ALNUM_WHITESPACE_REGEX = Regex("[^a-z0-9\\s]")
private val ARTICLES_REGEX = Regex("\\b(a|an|the)\\b")
private val MULTISPACE_REGEX = Regex("\\s+")
private val JSON_CODE_FENCE_REGEX = Regex("^```(?:json)?\\s*(\\{[\\s\\S]*})\\s*```$", setOf(RegexOption.IGNORE_CASE))
private val faithfulnessMetric = FaithfulnessMetric(allowHeuristicFallback = true)
private val responseGroundednessMetric = ResponseGroundednessMetric()
private val bertScoreThreadLocal = ThreadLocal.withInitial { BertScore() }
private val bertScoreWarningLogged = AtomicBoolean(false)

@Serializable
private data class ExperimentParagraph(
    val idx: Int = 0,
    val title: String = "",
    @SerialName("paragraph_text")
    val paragraphText: String,
    @SerialName("is_supporting")
    val isSupporting: Boolean = false,
)

@Serializable
private data class ExperimentSample(
    val id: String,
    val paragraphs: List<ExperimentParagraph>,
    val question: String,
    val answer: String,
    @SerialName("answer_aliases")
    val answerAliases: List<String> = emptyList(),
    @SerialName("question_decomposition")
    val questionDecomposition: List<JsonElement> = emptyList(),
    val answerable: Boolean = true,
    val metadata: JsonObject = JsonObject(emptyMap()),
)

@Serializable
private data class ManifestRow(
    val id: String,
    @SerialName("hop_count")
    val hopCount: Int? = null,
    @SerialName("overlap_bucket")
    val overlapBucket: String? = null,
)

@Serializable
private data class PerQuestionResult(
    val condition: String,
    val sampleId: String,
    val hopCount: Int,
    val overlapBucket: String,
    val category: String,
    val labelTrue: Boolean? = null,
    val supportCount: Int,
    val exactMatch: Double,
    val precision: Double,
    val recall: Double,
    val f1: Double,
    val bertScorePrecision: Double,
    val bertScoreRecall: Double,
    val bertScoreF1: Double,
    val supportRecallAt1: Double,
    val supportRecallAt3: Double,
    val supportRecallAt5: Double,
    val bridgeCoverageAt5: Double,
    val mrrAt5: Double,
    val ndcgAt5: Double,
    val faithfulness: Double,
    val responseGroundedness: Double,
    val indexLatencyMs: Double,
    val queryLatencyMs: Double,
    val totalLatencyMs: Double,
    val prediction: String,
    val error: String? = null,
)

private enum class Condition(
    val id: String,
    val description: String,
) {
    LIGHTRAG(
        id = "lightrag",
        description = "LightRAG (lightrag)",
    ),
    PATHRAG(
        id = "pathrag",
        description = "PathRAG (pathrag)",
    ),
    GRAPHRAG(
        id = "graphrag",
        description = "GraphRAG (com/microsoft/graphrag)",
    ),
    ;

    companion object {
        fun parse(raw: String): List<Condition> {
            if (raw == "all") return entries
            val wanted =
                raw
                    .split(',')
                    .map { it.trim() }
                    .filter { it.isNotBlank() }
                    .toSet()
            require(wanted.isNotEmpty()) {
                "--conditions must be 'all' or a comma-separated list of condition IDs"
            }
            val resolved = entries.filter { it.id in wanted }
            require(resolved.size == wanted.size) {
                val known = entries.joinToString(",") { it.id }
                val missing = wanted - resolved.map { it.id }.toSet()
                "Unknown conditions: ${missing.joinToString(",")}. Known: $known"
            }
            return resolved
        }
    }
}

private data class RunConfig(
    val dataPath: Path,
    val outputDir: Path,
    val manifestPath: Path?,
    val commonConfigPath: Path?,
    val commonConfig: CommonRagConfig?,
    val conditions: List<Condition>,
    val topK: Int,
    val llmModel: String,
    val embeddingModel: String,
    val llmProvider: String,
    val llmBaseUrl: String?,
    val limit: Int?,
    val parallelism: Int,
    val useUnifiedApi: Boolean,
    val useUnifiedPersistence: Boolean,
)

private data class RetrievalAndAnswer(
    val context: List<String>,
    val prediction: String,
    val indexLatencyMs: Double,
    val queryLatencyMs: Double,
    val totalLatencyMs: Double,
)

private data class BertScoreMetrics(
    val precision: Double,
    val recall: Double,
    val f1: Double,
)

fun main(args: Array<String>) {
    val config = parseArgs(args)
    if (!config.useUnifiedApi) {
        printLegacyModeDeprecationWarning("shared.eval.graph.MultiConditionGraphExperimentKt")
    }

    val samples =
        Files.newBufferedReader(config.dataPath).useLines { lines ->
            val answerable = mutableListOf<ExperimentSample>()
            var sawAnySample = false
            for (rawLine in lines) {
                val line = rawLine.trim()
                if (line.isBlank()) continue
                sawAnySample = true
                val sample = json.decodeFromString(ExperimentSample.serializer(), line)
                if (!sample.answerable) continue
                answerable += sample
                if (config.limit != null && answerable.size >= config.limit) break
            }
            require(sawAnySample) { "No samples found in ${config.dataPath}" }
            answerable
        }

    require(samples.isNotEmpty()) { "No answerable samples to run." }

    val manifestById = loadManifest(config.manifestPath)

    config.outputDir.createDirectories()
    val perQuestionDir = config.outputDir.resolve("per_question")
    perQuestionDir.createDirectories()

    println("Running ${config.conditions.size} conditions on ${samples.size} samples")
    println("Parallelism: ${config.parallelism}")
    println("Unified API: ${config.useUnifiedApi}")
    println("Unified persistence SPI: ${config.useUnifiedPersistence}")
    println("Data: ${config.dataPath}")
    println("Output: ${config.outputDir}")

    for (condition in config.conditions) {
        println("\n=== Condition: ${condition.id} (${condition.description}) ===")
        val conditionRows =
            runBlocking {
                runConditionForSamples(
                    condition = condition,
                    config = config,
                    samples = samples,
                    manifestById = manifestById,
                )
            }

        val outputPath = perQuestionDir.resolve("${condition.id}.jsonl")
        writeJsonl(outputPath, conditionRows)
        println("Wrote ${conditionRows.size} rows -> $outputPath")
    }

    println("\nFinished. Aggregate CSV can be produced by scripts/aggregate_experiment_results.py")
}

private fun printLegacyModeDeprecationWarning(entryPoint: String) {
    println(
        "WARNING: --use-unified-api=false keeps '$entryPoint' on legacy condition runners. " +
            "This compatibility path is planned for deprecation; prefer --use-unified-api=true.",
    )
}

private data class IndexedPerQuestionResult(
    val index: Int,
    val row: PerQuestionResult,
)

private suspend fun runConditionForSamples(
    condition: Condition,
    config: RunConfig,
    samples: List<ExperimentSample>,
    manifestById: Map<String, ManifestRow>,
): List<PerQuestionResult> =
    coroutineScope {
        val orderedRows = arrayOfNulls<PerQuestionResult>(samples.size)
        val nextIndex = AtomicInteger(0)
        val completedCount = AtomicInteger(0)
        val workerCount = minOf(config.parallelism, samples.size)

        val resultsByWorker =
            (0 until workerCount)
                .map { workerIndex ->
                    async(Dispatchers.IO) {
                        val runner = createConditionRunner(condition, config, workerIndex)
                        val localResults = mutableListOf<IndexedPerQuestionResult>()
                        while (true) {
                            val sampleIndex = nextIndex.getAndIncrement()
                            if (sampleIndex >= samples.size) break

                            val sample = samples[sampleIndex]
                            val result =
                                try {
                                    runConditionForSample(
                                        runner = runner,
                                        sample = sample,
                                        manifestRow = manifestById[sample.id],
                                    )
                                } catch (ex: Exception) {
                                    errorResultForSample(
                                        conditionId = condition.id,
                                        sample = sample,
                                        manifestRow = manifestById[sample.id],
                                        errorMessage = ex.message ?: ex::class.simpleName,
                                    )
                                }
                            localResults += IndexedPerQuestionResult(index = sampleIndex, row = result)

                            val done = completedCount.incrementAndGet()
                            if (done % 10 == 0 || done == samples.size) {
                                println("${condition.id}: processed $done/${samples.size}")
                            }
                        }
                        localResults
                    }
                }.awaitAll()

        for (workerResults in resultsByWorker) {
            for (entry in workerResults) {
                orderedRows[entry.index] = entry.row
            }
        }

        orderedRows.mapIndexed { index, row ->
            row ?: error("Missing result for sample index $index in condition ${condition.id}")
        }
    }

private fun errorResultForSample(
    conditionId: String,
    sample: ExperimentSample,
    manifestRow: ManifestRow?,
    errorMessage: String?,
): PerQuestionResult =
    PerQuestionResult(
        condition = conditionId,
        sampleId = sample.id,
        hopCount = inferHopCount(sample, manifestRow?.hopCount),
        overlapBucket = manifestRow?.overlapBucket ?: "unknown",
        category = sampleCategory(sample),
        labelTrue = sampleLabelTrue(sample),
        supportCount = sample.paragraphs.count { it.isSupporting },
        exactMatch = 0.0,
        precision = 0.0,
        recall = 0.0,
        f1 = 0.0,
        bertScorePrecision = 0.0,
        bertScoreRecall = 0.0,
        bertScoreF1 = 0.0,
        supportRecallAt1 = 0.0,
        supportRecallAt3 = 0.0,
        supportRecallAt5 = 0.0,
        bridgeCoverageAt5 = 0.0,
        mrrAt5 = 0.0,
        ndcgAt5 = 0.0,
        faithfulness = 0.0,
        responseGroundedness = 0.0,
        indexLatencyMs = 0.0,
        queryLatencyMs = 0.0,
        totalLatencyMs = 0.0,
        prediction = "",
        error = errorMessage,
    )

private fun runConditionForSample(
    runner: ConditionRunner,
    sample: ExperimentSample,
    manifestRow: ManifestRow?,
): PerQuestionResult {
    val docs = sample.paragraphs.map { it.paragraphText }
    val supportDocs = sample.paragraphs.filter { it.isSupporting }.map { it.paragraphText }
    val retrieval = runner.run(sample = sample, docs = docs)
    val scoredPrediction = predictionForQaScoring(retrieval.prediction)

    val golds = listOf(sample.answer) + sample.answerAliases
    val qaMetrics = bestQaMetrics(scoredPrediction, golds)
    val em = bestExactMatch(scoredPrediction, golds)
    val bertScoreMetrics = bestBertScoreMetrics(scoredPrediction, golds)

    val recallAt1 = supportRecallAtK(retrieval.context, supportDocs, 1)
    val recallAt3 = supportRecallAtK(retrieval.context, supportDocs, 3)
    val recallAt5 = supportRecallAtK(retrieval.context, supportDocs, 5)
    val bridgeAt5 = bridgeCoverageAtK(retrieval.context, supportDocs, 5)
    val mrrAt5 = mrrAtK(retrieval.context, supportDocs, 5)
    val ndcgAt5 = nDCGAtK(retrieval.context, supportDocs, 5)
    val (faithfulness, responseGroundedness) =
        computeGroundingMetrics(
            question = sample.question,
            prediction = retrieval.prediction,
            contexts = retrieval.context,
        )

    return PerQuestionResult(
        condition = runner.id,
        sampleId = sample.id,
        hopCount = inferHopCount(sample, manifestRow?.hopCount),
        overlapBucket = manifestRow?.overlapBucket ?: "unknown",
        category = sampleCategory(sample),
        labelTrue = sampleLabelTrue(sample),
        supportCount = supportDocs.size,
        exactMatch = em,
        precision = qaMetrics.precision,
        recall = qaMetrics.recall,
        f1 = qaMetrics.f1,
        bertScorePrecision = bertScoreMetrics.precision,
        bertScoreRecall = bertScoreMetrics.recall,
        bertScoreF1 = bertScoreMetrics.f1,
        supportRecallAt1 = recallAt1,
        supportRecallAt3 = recallAt3,
        supportRecallAt5 = recallAt5,
        bridgeCoverageAt5 = bridgeAt5,
        mrrAt5 = mrrAt5,
        ndcgAt5 = ndcgAt5,
        faithfulness = faithfulness,
        responseGroundedness = responseGroundedness,
        indexLatencyMs = retrieval.indexLatencyMs,
        queryLatencyMs = retrieval.queryLatencyMs,
        totalLatencyMs = retrieval.totalLatencyMs,
        prediction = retrieval.prediction,
        error = null,
    )
}

private interface ConditionRunner {
    val id: String

    fun run(
        sample: ExperimentSample,
        docs: List<String>,
    ): RetrievalAndAnswer
}

private fun createConditionRunner(
    condition: Condition,
    config: RunConfig,
    workerIndex: Int = 0,
): ConditionRunner {
    if (config.useUnifiedApi) {
        return createUnifiedConditionRunner(
            condition = condition,
            config = config,
            workdirSuffix = scopedWorkdirSuffix("${condition.id}_unified", workerIndex),
        )
    }
    return when (condition) {
        Condition.LIGHTRAG -> createLightRagRunner(config, scopedWorkdirSuffix("lightrag", workerIndex))
        Condition.PATHRAG -> createPathRagRunner(config, scopedWorkdirSuffix("pathrag", workerIndex))
        Condition.GRAPHRAG -> createGraphRagRunner(config, scopedWorkdirSuffix("graphrag", workerIndex))
    }
}

private fun scopedWorkdirSuffix(
    base: String,
    workerIndex: Int,
): String = if (workerIndex == 0) base else "${base}_worker${workerIndex + 1}"

private fun createUnifiedConditionRunner(
    condition: Condition,
    config: RunConfig,
    workdirSuffix: String,
): ConditionRunner {
    val workdirRoot = config.outputDir.resolve("workdirs").resolve(workdirSuffix)
    workdirRoot.createDirectories()
    val configPath = config.commonConfigPath?.toString()

    return object : ConditionRunner {
        override val id: String = condition.id

        override fun run(
            sample: ExperimentSample,
            docs: List<String>,
        ): RetrievalAndAnswer {
            val sampleWorkdir = workdirRoot.resolve(sanitizeSampleId(sample.id))
            resetDirectory(sampleWorkdir)
            val ragId = condition.toUnifiedRagId()

            if (condition == Condition.GRAPHRAG) {
                require(config.llmProvider == "openai") {
                    "GraphRAG currently supports only --provider openai (got '${config.llmProvider}')"
                }
                val apiKey = config.commonConfig?.sharedModelSettings()?.apiKey ?: System.getenv("OPENAI_API_KEY")
                require(!apiKey.isNullOrBlank()) { "OPENAI_API_KEY is required for GraphRAG condition." }
            }

            val handle =
                UnifiedRagFactory.create(
                    ragId = ragId,
                    configPath = configPath,
                    overrides = buildUnifiedOverrides(condition, config, sampleWorkdir),
                )
            val rag = handle.rag
            return try {
                val indexStart = System.nanoTime()
                rag.drop()
                rag.upsert(docs.filter { it.isNotBlank() })
                val indexMs = elapsedMs(indexStart)

                val queryText =
                    if (condition == Condition.PATHRAG) {
                        "Answer in one or few words, no extra information: ${sample.question}"
                    } else {
                        sample.question
                    }
                val queryStart = System.nanoTime()
                val response =
                    rag.query(
                        queryText,
                        buildUnifiedQuery(condition, config),
                    )
                val queryMs = elapsedMs(queryStart)

                val contexts =
                    response.context
                        .map { it.text.trim() }
                        .filter { it.isNotBlank() }
                        .take(config.topK)
                val prediction =
                    if (condition == Condition.LIGHTRAG) {
                        removeThinkTags(response.answer.orEmpty()).trim()
                    } else {
                        response.answer.orEmpty().trim()
                    }

                RetrievalAndAnswer(
                    context = contexts,
                    prediction = prediction,
                    indexLatencyMs = indexMs,
                    queryLatencyMs = queryMs,
                    totalLatencyMs = indexMs + queryMs,
                )
            } finally {
                runCatching { rag.drop() }
            }
        }
    }
}

private fun Condition.toUnifiedRagId(): RagId =
    when (this) {
        Condition.LIGHTRAG -> RagId.LIGHT_RAG
        Condition.PATHRAG -> RagId.PATH_RAG
        Condition.GRAPHRAG -> RagId.GRAPH_RAG
    }

private fun buildUnifiedOverrides(
    condition: Condition,
    config: RunConfig,
    sampleWorkdir: Path,
): Map<String, Any?> {
    val persistenceOverrides: Map<String, Any?> =
        if (config.useUnifiedPersistence) {
            mapOf(
                "useUnifiedPersistence" to true,
                "persistenceBackend" to "filesystem_snapshot",
                "persistenceRootDir" to sampleWorkdir.resolve("unified_persistence").toString(),
            )
        } else {
            emptyMap()
        }

    return when (condition) {
        Condition.LIGHTRAG ->
            mapOf(
                "workingDir" to sampleWorkdir.toString(),
            ) + persistenceOverrides

        Condition.PATHRAG ->
            mapOf(
                "workingDir" to sampleWorkdir.toString(),
            ) + persistenceOverrides

        Condition.GRAPHRAG ->
            mapOf(
                "rootDir" to sampleWorkdir.toString(),
                "chatModelName" to config.llmModel,
                "embeddingModelName" to config.embeddingModel,
            ) + persistenceOverrides
    }
}

private fun buildUnifiedQuery(
    condition: Condition,
    config: RunConfig,
): UnifiedQuery =
    when (condition) {
        Condition.LIGHTRAG ->
            UnifiedQuery(
                mode = UnifiedMode.HYBRID,
                topK = config.topK,
                includeAnswer = true,
                includeContext = true,
                includeReferences = true,
                extras =
                    mapOf(
                        "chunkTopK" to config.topK,
                    ),
            )

        Condition.PATHRAG ->
            UnifiedQuery(
                mode = UnifiedMode.HYBRID,
                topK = config.topK,
                includeAnswer = true,
                includeContext = true,
                includeReferences = false,
                extras =
                    mapOf(
                        "responseType" to "One Sentence",
                    ),
            )

        Condition.GRAPHRAG ->
            UnifiedQuery(
                mode = UnifiedMode.BASIC,
                topK = config.topK,
                includeAnswer = true,
                includeContext = true,
                includeReferences = true,
                extras =
                    mapOf(
                        "responseType" to "Answer in one or few words.",
                    ),
            )
    }

private fun createLightRagRunner(
    config: RunConfig,
    workdirSuffix: String,
): ConditionRunner {
    val lightSettings = config.commonConfig?.toLightRagConfig()
    val provider = (lightSettings?.provider ?: config.llmProvider).lowercase()
    val llmModelName = lightSettings?.llmModelName ?: config.llmModel
    val embeddingModelName = lightSettings?.embeddingModelName ?: config.embeddingModel
    val baseUrl = lightSettings?.baseUrl ?: config.llmBaseUrl
    val apiKey = lightSettings?.apiKey ?: System.getenv("OPENAI_API_KEY")
    if (provider == "openai" && apiKey.isNullOrBlank()) {
        error("OPENAI_API_KEY is required for provider=openai")
    }

    val chatModel =
        LLMFactory.createChatModel(
            binding = provider,
            modelName = llmModelName,
            baseUrl = baseUrl,
            apiKey = apiKey,
            timeout = 120,
            temperature = 0.0,
            logRequests = false,
            logResponses = false,
        )
    val retrievalEmbeddingModel =
        LLMFactory.createEmbeddingModel(
            binding = provider,
            modelName = embeddingModelName,
            baseUrl = baseUrl,
            apiKey = apiKey,
            timeout = 120,
        )
    val tokenizerEncoding = Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.CL100K_BASE)
    val tokenizer: (String) -> List<Int> = { text ->
        val intArrayList = tokenizerEncoding.encode(text)
        val list = mutableListOf<Int>()
        for (i in 0 until intArrayList.size()) {
            list.add(intArrayList.get(i))
        }
        list
    }
    val decoder: (List<Int>) -> String = { tokenIds ->
        val intArrayList = IntArrayList()
        tokenIds.forEach { intArrayList.add(it) }
        tokenizerEncoding.decode(intArrayList)
    }
    val configuredRoot = lightSettings?.workingDir?.let { Path.of(it) }
    val workdirRoot = (configuredRoot ?: config.outputDir.resolve("workdirs")).resolve(workdirSuffix)
    workdirRoot.createDirectories()

    return object : ConditionRunner {
        override val id: String = Condition.LIGHTRAG.id

        override fun run(
            sample: ExperimentSample,
            docs: List<String>,
        ): RetrievalAndAnswer {
            val sampleWorkdir = workdirRoot.resolve(sanitizeSampleId(sample.id))
            resetDirectory(sampleWorkdir)
            val rag =
                createLightRagForSample(
                    workingDir = sampleWorkdir.toString(),
                    chatModel = chatModel,
                    embeddingModel = retrievalEmbeddingModel,
                    tokenizer = tokenizer,
                    decoder = decoder,
                    graphStorageName = lightSettings?.graphStorageName ?: "InMemoryGraphStorage",
                    vectorStorageName = lightSettings?.vectorStorageName ?: "InMemoryVectorStorage",
                    chunkTokenSize = lightSettings?.chunkTokenSize ?: 1200,
                    chunkOverlapTokenSize = lightSettings?.chunkOverlapTokenSize ?: 100,
                    entityTypes =
                        lightSettings?.entityTypes
                            ?: listOf("Person", "Organization", "Location", "Event", "Concept"),
                    language = lightSettings?.language ?: "English",
                    cosineBetterThreshold = lightSettings?.cosineBetterThreshold,
                )

            val indexStart = System.nanoTime()
            runBlocking {
                rag.storageManager.initialize()
                rag.insert(docs.filter { it.isNotBlank() })
                rag.rebuildDerivedStorageIfEmpty()
            }
            val indexMs = elapsedMs(indexStart)

            val queryStart = System.nanoTime()
            val queryResult =
                runBlocking {
                    rag.query(
                        sample.question,
                        LightRagQueryParam(
                            mode = "hybrid",
                            includeReferences = true,
                            topK = config.topK,
                            chunkTopK = config.topK,
                        ),
                    )
                }
            val queryMs = elapsedMs(queryStart)

            return RetrievalAndAnswer(
                context = extractLightRagChunkContents(queryResult?.rawData).take(config.topK),
                prediction = removeThinkTags(queryResult?.content.orEmpty()).trim(),
                indexLatencyMs = indexMs,
                queryLatencyMs = queryMs,
                totalLatencyMs = indexMs + queryMs,
            )
        }
    }
}

private fun createLightRagForSample(
    workingDir: String,
    chatModel: ChatModel,
    embeddingModel: EmbeddingModel,
    tokenizer: (String) -> List<Int>,
    decoder: (List<Int>) -> String,
    graphStorageName: String,
    vectorStorageName: String,
    chunkTokenSize: Int,
    chunkOverlapTokenSize: Int,
    entityTypes: List<String>,
    language: String,
    cosineBetterThreshold: Double?,
): LightRAG {
    val globalConfig =
        mapOf(
            "llm_model_func" to chatModel,
            "embedding_func" to embeddingModel,
            "chunk_token_size" to chunkTokenSize,
            "chunk_overlap_token_size" to chunkOverlapTokenSize,
            "entity_types" to entityTypes,
            "language" to language,
            "working_dir" to workingDir,
            "enable_llm_cache" to false,
        )

    val storageManager =
        StorageManager(
            workingDir = workingDir,
            embeddingModel = embeddingModel,
            graphStorageName = graphStorageName,
            vectorStorageName = vectorStorageName,
            addonConfig = AddonConfig(cosineBetterThreshold = cosineBetterThreshold),
            globalConfig = globalConfig,
        )
    val ingestionService = IngestionService(storageManager, globalConfig, tokenizer, decoder)
    val queryService =
        QueryService(
            storageManager = storageManager,
            chatModel = chatModel,
            hashingKv = null,
            globalConfig = globalConfig,
            tokenizer = tokenizer,
            decoder = decoder,
        )

    return LightRAG(ingestionService, queryService, storageManager)
}

private fun extractLightRagChunkContents(rawData: Map<String, Any?>?): List<String> {
    if (rawData == null) return emptyList()
    val fromRoot = extractLightRagChunkContentsFromMap(rawData)
    if (fromRoot.isNotEmpty()) return fromRoot
    val dataMap = rawData["data"] as? Map<*, *> ?: return emptyList()
    return extractLightRagChunkContentsFromMap(dataMap)
}

private fun extractLightRagChunkContentsFromMap(map: Map<*, *>): List<String> {
    val chunks = map["chunks"] as? List<*> ?: return emptyList()
    return chunks.mapNotNull { chunk ->
        val chunkMap = chunk as? Map<*, *> ?: return@mapNotNull null
        val value = chunkMap["content"] ?: chunkMap["text"] ?: return@mapNotNull null
        value.toString().trim().takeIf { it.isNotBlank() }
    }
}

private fun createPathRagRunner(
    config: RunConfig,
    workdirSuffix: String,
): ConditionRunner {
    val pathSettings = config.commonConfig?.toPathRagConfig()
    val pathRuntimeSettings = pathSettings?.toRuntimeSettingsMap() ?: emptyMap()
    val configuredRoot = pathSettings?.workingDir?.let { Path.of(it) }
    val workdirRoot = (configuredRoot ?: config.outputDir.resolve("workdirs")).resolve(workdirSuffix)
    workdirRoot.createDirectories()

    return object : ConditionRunner {
        override val id: String = Condition.PATHRAG.id

        override fun run(
            sample: ExperimentSample,
            docs: List<String>,
        ): RetrievalAndAnswer {
            val sampleWorkdir = workdirRoot.resolve(sanitizeSampleId(sample.id))
            resetDirectory(sampleWorkdir)
            val rag =
                PathRAG(
                    workingDir = sampleWorkdir.toString(),
                    kvStorage = pathSettings?.kvStorage ?: "JsonKVStorage",
                    vectorStorage = pathSettings?.vectorStorage ?: "NanoVectorDBStorage",
                    graphStorage = pathSettings?.graphStorage ?: "NetworkXStorage",
                    chunkTokenSize = pathSettings?.chunkTokenSize ?: 1200,
                    chunkOverlapTokenSize = pathSettings?.chunkOverlapTokenSize ?: 100,
                    language = pathSettings?.language ?: "English",
                    runtimeSettings = pathRuntimeSettings,
                )

            return try {
                val indexStart = System.nanoTime()
                rag.clear()
                rag.insert(docs.filter { it.isNotBlank() })
                val indexMs = elapsedMs(indexStart)

                val questionPrompt = "Answer in one or few words, no extra information: ${sample.question}"
                val queryStart = System.nanoTime()
                val prediction = rag.query(questionPrompt, param = PathRagQueryParam(mode = "hybrid", topK = config.topK))
                val contextRaw =
                    rag.query(
                        questionPrompt,
                        param = PathRagQueryParam(mode = "hybrid", topK = config.topK, onlyNeedContext = true),
                    )
                val queryMs = elapsedMs(queryStart)
                val contexts = RagasContextExtractor.extractContexts(contextRaw).take(config.topK)

                RetrievalAndAnswer(
                    context = contexts,
                    prediction = prediction,
                    indexLatencyMs = indexMs,
                    queryLatencyMs = queryMs,
                    totalLatencyMs = indexMs + queryMs,
                )
            } finally {
                rag.close()
            }
        }
    }
}

private fun createGraphRagRunner(
    config: RunConfig,
    workdirSuffix: String,
): ConditionRunner {
    require(config.llmProvider == "openai") {
        "GraphRAG currently supports only --provider openai (got '${config.llmProvider}')"
    }
    val apiKey = config.commonConfig?.sharedModelSettings()?.apiKey ?: System.getenv("OPENAI_API_KEY")
    require(!apiKey.isNullOrBlank()) { "OPENAI_API_KEY is required for GraphRAG condition." }

    val queryModelBuilder =
        OpenAiStreamingChatModel
            .builder()
            .apiKey(apiKey)
            .modelName(config.llmModel)
            .temperature(0.0)
    if (!config.llmBaseUrl.isNullOrBlank()) {
        queryModelBuilder.baseUrl(config.llmBaseUrl)
    }
    val queryModel = queryModelBuilder.build()
    val embeddingModel = defaultEmbeddingModel(apiKey, config.embeddingModel)
    val pipeline = defaultPipeline()
    val workdirRoot = config.outputDir.resolve("workdirs").resolve(workdirSuffix)
    workdirRoot.createDirectories()

    return object : ConditionRunner {
        override val id: String = Condition.GRAPHRAG.id

        override fun run(
            sample: ExperimentSample,
            docs: List<String>,
        ): RetrievalAndAnswer {
            val sampleRoot = workdirRoot.resolve(sanitizeSampleId(sample.id))
            val inputDir = sampleRoot.resolve("input")
            val outputDir = sampleRoot.resolve("output")
            val updateDir = sampleRoot.resolve("update_output")

            resetDirectory(sampleRoot)
            inputDir.createDirectories()
            outputDir.createDirectories()
            updateDir.createDirectories()
            writeGraphRagInputDocs(inputDir, docs)

            val indexStart = System.nanoTime()
            runBlocking {
                runPipeline(
                    pipeline = pipeline,
                    config = GraphRagConfig(sampleRoot, inputDir, outputDir, updateDir),
                    callbacks = NoopWorkflowCallbacks(),
                ).collect { result ->
                    if (!result.errors.isNullOrEmpty()) {
                        error(
                            "GraphRAG indexing failed at '${result.workflow}': ${result.errors.joinToString("; ")}",
                        )
                    }
                }
            }
            val indexMs = elapsedMs(indexStart)

            val queryStart = System.nanoTime()
            val index = QueryIndexLoader(outputDir).load()
            val queryResult =
                runBlocking {
                    BasicQueryEngine(
                        streamingModel = queryModel,
                        embeddingModel = embeddingModel,
                        vectorStore = index.vectorStore,
                        textUnits = index.textUnits,
                        textEmbeddings = index.textEmbeddings,
                        topK = config.topK,
                    ).answer(
                        question = sample.question,
                        responseType = "Answer in one or few words.",
                    )
                }
            val queryMs = elapsedMs(queryStart)

            return RetrievalAndAnswer(
                context = queryResult.context.map { it.text }.take(config.topK),
                prediction = queryResult.answer,
                indexLatencyMs = indexMs,
                queryLatencyMs = queryMs,
                totalLatencyMs = indexMs + queryMs,
            )
        }
    }
}

private fun resetDirectory(path: Path) {
    if (Files.exists(path)) {
        Files.walk(path).use { stream ->
            stream
                .sorted(Comparator.reverseOrder())
                .forEach { Files.deleteIfExists(it) }
        }
    }
    path.createDirectories()
}

private fun writeGraphRagInputDocs(
    inputDir: Path,
    docs: List<String>,
) {
    val docsToWrite =
        docs
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .ifEmpty { listOf("No content provided.") }
    docsToWrite.forEachIndexed { index, doc ->
        Files.writeString(inputDir.resolve("doc_${index + 1}.txt"), doc)
    }
}

private fun sanitizeSampleId(sampleId: String): String = sampleId.replace(Regex("[^A-Za-z0-9._-]"), "_")

private fun elapsedMs(startNanos: Long): Double = (System.nanoTime() - startNanos).toDouble() / 1_000_000.0

private fun parseArgs(args: Array<String>): RunConfig {
    val opts = CliUtils.parseOptions(args.toList())

    val dataPath = Path.of(opts["data"] ?: "data/musique_experiment/musique_dev_balanced_300.jsonl")
    require(Files.exists(dataPath)) { "Missing --data file: $dataPath" }
    val commonConfigPath = opts["config"]?.let { Path.of(it) }
    if (commonConfigPath != null) {
        require(Files.exists(commonConfigPath)) { "Missing --config file: $commonConfigPath" }
    }
    val commonConfig = commonConfigPath?.let { CommonRagConfigLoader.load(it) }
    val sharedModel = commonConfig?.sharedModelSettings()

    val outputDir =
        Path.of(
            opts["output-dir"]
                ?: Path.of("eval_results", "graph_multicondition_${Instant.now().toString().replace(':', '_')}").toString(),
        )

    val conditions = Condition.parse(opts["conditions"] ?: "all")
    val topK = (opts["top-k"] ?: "5").toIntOrNull() ?: 5
    require(topK > 0) { "--top-k must be positive" }

    val llmModel = opts["llm-model"] ?: sharedModel?.llmModelName ?: System.getenv("LLM_MODEL") ?: "gpt-5.4-mini"
    val embeddingModel =
        opts["embedding-model"] ?: sharedModel?.embeddingModelName ?: System.getenv("EMBEDDING_MODEL") ?: "text-embedding-3-small"
    val llmProvider = (opts["provider"] ?: sharedModel?.provider ?: System.getenv("LLM_PROVIDER") ?: "openai").lowercase()
    val llmBaseUrl = opts["llm-base-url"] ?: sharedModel?.baseUrl ?: System.getenv("LLM_BASE_URL")
    val limit = opts["limit"]?.toIntOrNull()
    val parallelism = (opts["parallelism"] ?: "5").toIntOrNull() ?: 5
    require(parallelism > 0) { "--parallelism must be positive" }
    val useUnifiedApi = parseBooleanOption(opts["use-unified-api"]) ?: false
    val useUnifiedPersistence = parseBooleanOption(opts["use-unified-persistence"]) ?: false
    require(!useUnifiedPersistence || useUnifiedApi) {
        "--use-unified-persistence requires --use-unified-api=true"
    }

    val manifestPath =
        opts["manifest"]?.let { Path.of(it) }
            ?: detectManifestNearData(dataPath)

    return RunConfig(
        dataPath = dataPath,
        outputDir = outputDir,
        manifestPath = manifestPath,
        commonConfigPath = commonConfigPath,
        commonConfig = commonConfig,
        conditions = conditions,
        topK = topK,
        llmModel = llmModel,
        embeddingModel = embeddingModel,
        llmProvider = llmProvider,
        llmBaseUrl = llmBaseUrl,
        limit = limit,
        parallelism = parallelism,
        useUnifiedApi = useUnifiedApi,
        useUnifiedPersistence = useUnifiedPersistence,
    )
}

private fun parseBooleanOption(raw: String?): Boolean? =
    when (raw?.trim()?.lowercase()) {
        "1", "true", "yes", "on" -> true
        "0", "false", "no", "off" -> false
        else -> null
    }

private fun detectManifestNearData(dataPath: Path): Path? {
    val parent = dataPath.parent ?: return null
    val candidate = parent.resolve("musique_dev_multihop_manifest.jsonl")
    return if (Files.exists(candidate)) candidate else null
}

private fun loadManifest(path: Path?): Map<String, ManifestRow> {
    if (path == null || !Files.exists(path)) return emptyMap()
    val lines = Files.readAllLines(path).map { it.trim() }.filter { it.isNotBlank() }
    return lines
        .map { json.decodeFromString(ManifestRow.serializer(), it) }
        .associateBy { it.id }
}

private fun writeJsonl(
    path: Path,
    rows: List<PerQuestionResult>,
) {
    Files.newBufferedWriter(path).use { writer ->
        rows.forEach { row ->
            writer.write(json.encodeToString(PerQuestionResult.serializer(), row))
            writer.newLine()
        }
    }
}

private fun supportRecallAtK(
    retrieved: List<String>,
    supporting: List<String>,
    k: Int,
): Double {
    if (supporting.isEmpty()) return 0.0
    val retrievedNorm = retrieved.take(k).map { normalize(it) }.toSet()
    val supportingNorm = supporting.map { normalize(it) }.toSet()
    if (supportingNorm.isEmpty()) return 0.0
    val hits = supportingNorm.count { it in retrievedNorm }
    return hits.toDouble() / supportingNorm.size
}

private fun mrrAtK(
    retrieved: List<String>,
    supporting: List<String>,
    k: Int,
): Double {
    if (supporting.isEmpty()) return 0.0
    val supportingNorm = supporting.map { normalize(it) }.filter { it.isNotBlank() }.toSet()
    if (supportingNorm.isEmpty()) return 0.0

    val retrievedNorm = retrieved.take(k).map { normalize(it) }
    for ((index, doc) in retrievedNorm.withIndex()) {
        if (doc in supportingNorm) {
            return 1.0 / (index + 1).toDouble()
        }
    }
    return 0.0
}

private fun nDCGAtK(
    retrieved: List<String>,
    supporting: List<String>,
    k: Int,
): Double {
    if (supporting.isEmpty()) return 0.0
    val supportingNorm = supporting.map { normalize(it) }.filter { it.isNotBlank() }.toSet()
    if (supportingNorm.isEmpty()) return 0.0

    val retrievedNorm = retrieved.take(k).map { normalize(it) }
    val dcg =
        retrievedNorm.withIndex().sumOf { (index, doc) ->
            if (doc in supportingNorm) {
                1.0 / log2((index + 2).toDouble())
            } else {
                0.0
            }
        }

    val idealRelevant = minOf(supportingNorm.size, k)
    if (idealRelevant == 0) return 0.0
    val idcg =
        (0 until idealRelevant).sumOf { index ->
            1.0 / log2((index + 2).toDouble())
        }
    if (idcg == 0.0) return 0.0

    return dcg / idcg
}

private fun bridgeCoverageAtK(
    retrieved: List<String>,
    supporting: List<String>,
    k: Int,
): Double {
    if (supporting.isEmpty()) return 0.0
    val retrievedNorm = retrieved.take(k).map { normalize(it) }.toSet()
    val supportingNorm = supporting.map { normalize(it) }.toSet()
    if (supportingNorm.isEmpty()) return 0.0
    val hits = supportingNorm.count { it in retrievedNorm }
    return if (hits == supportingNorm.size) 1.0 else 0.0
}

private fun sampleCategory(sample: ExperimentSample): String {
    val category = (sample.metadata["category"] as? JsonPrimitive)?.contentOrNull?.trim()
    return if (category.isNullOrBlank()) "unknown" else category
}

private fun sampleLabelTrue(sample: ExperimentSample): Boolean? {
    val primitive = sample.metadata["label_true"] as? JsonPrimitive ?: return null
    primitive.booleanOrNull?.let { return it }
    return when (primitive.contentOrNull?.trim()?.lowercase()) {
        "true" -> true
        "false" -> false
        else -> null
    }
}

private fun computeGroundingMetrics(
    question: String,
    prediction: String,
    contexts: List<String>,
): Pair<Double, Double> {
    val evalSample =
        SingleTurnSample(
            userInput = question,
            retrievedContexts = contexts,
            response = prediction,
        )
    val faithfulness =
        runBlocking {
            scoreOrZero(faithfulnessMetric.singleTurnAscore(evalSample))
        }
    val responseGroundedness =
        runBlocking {
            scoreOrZero(responseGroundednessMetric.singleTurnAscore(evalSample))
        }
    return faithfulness to responseGroundedness
}

private fun scoreOrZero(value: Any?): Double {
    val numeric = (value as? Number)?.toDouble() ?: return 0.0
    return if (numeric.isFinite()) numeric else 0.0
}

private fun inferHopCount(
    sample: ExperimentSample,
    fallback: Int?,
): Int {
    val fromDecomp = sample.questionDecomposition.size
    return when {
        fromDecomp > 0 -> fromDecomp
        fallback != null -> fallback
        sample.id.startsWith("2hop") -> 2
        sample.id.startsWith("3hop") -> 3
        sample.id.startsWith("4hop") -> 4
        else -> 0
    }
}

private fun bestExactMatch(
    prediction: String,
    golds: List<String>,
): Double = golds.maxOfOrNull { if (normalize(prediction) == normalize(it)) 1.0 else 0.0 } ?: 0.0

private data class QaMetrics(
    val precision: Double,
    val recall: Double,
    val f1: Double,
)

private fun bestQaMetrics(
    prediction: String,
    golds: List<String>,
): QaMetrics = golds.map { qaMetrics(prediction, it) }.maxByOrNull { it.f1 } ?: QaMetrics(0.0, 0.0, 0.0)

private fun bestBertScoreMetrics(
    prediction: String,
    golds: List<String>,
): BertScoreMetrics {
    val candidate = prediction.trim()
    if (candidate.isBlank() || golds.isEmpty()) {
        return BertScoreMetrics(0.0, 0.0, 0.0)
    }

    val scorer = bertScoreThreadLocal.get()
    var best: BertScoreMetrics? = null
    var bestF1 = Double.NEGATIVE_INFINITY

    for (gold in golds) {
        val reference = gold.trim()
        if (reference.isBlank()) continue

        val score =
            runCatching { scorer.score(reference, candidate) }
                .onFailure { ex ->
                    if (bertScoreWarningLogged.compareAndSet(false, true)) {
                        println("Warning: BertScore computation failed, defaulting to 0.0. Cause: ${ex.message ?: ex::class.simpleName}")
                    }
                }.getOrNull() ?: continue

        val metrics =
            BertScoreMetrics(
                precision = sanitizeMetric(score.precision.toDouble()),
                recall = sanitizeMetric(score.recall.toDouble()),
                f1 = sanitizeMetric(score.f1.toDouble()),
            )
        if (metrics.f1 > bestF1) {
            best = metrics
            bestF1 = metrics.f1
        }
    }

    return best ?: BertScoreMetrics(0.0, 0.0, 0.0)
}

private fun sanitizeMetric(value: Double): Double = if (value.isFinite()) value else 0.0

private fun qaMetrics(
    prediction: String,
    gold: String,
): QaMetrics {
    val predTokens = tokenize(normalize(prediction))
    val goldTokens = tokenize(normalize(gold))
    if (predTokens.isEmpty() && goldTokens.isEmpty()) return QaMetrics(1.0, 1.0, 1.0)
    if (predTokens.isEmpty() || goldTokens.isEmpty()) return QaMetrics(0.0, 0.0, 0.0)

    val predCounts = predTokens.groupingBy { it }.eachCount()
    val goldCounts = goldTokens.groupingBy { it }.eachCount()
    var overlap = 0
    for ((token, pCount) in predCounts) {
        val gCount = goldCounts[token] ?: 0
        overlap += minOf(pCount, gCount)
    }
    if (overlap == 0) return QaMetrics(0.0, 0.0, 0.0)
    val precision = overlap.toDouble() / predTokens.size
    val recall = overlap.toDouble() / goldTokens.size
    val f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return QaMetrics(precision = precision, recall = recall, f1 = f1)
}

private fun normalize(text: String): String {
    val lowered = text.lowercase()
    val noPunc = lowered.replace(NON_ALNUM_WHITESPACE_REGEX, " ")
    val noArticles = noPunc.replace(ARTICLES_REGEX, " ")
    return noArticles.replace(MULTISPACE_REGEX, " ").trim()
}

private fun tokenize(text: String): List<String> =
    if (text.isBlank()) {
        emptyList()
    } else {
        text.split(' ')
    }

private fun predictionForQaScoring(prediction: String): String = extractAnswerFieldFromJsonPrediction(prediction) ?: prediction

private fun extractAnswerFieldFromJsonPrediction(prediction: String): String? {
    val trimmed = prediction.trim()
    if (trimmed.isEmpty()) return null

    val candidates = linkedSetOf(trimmed)
    extractJsonObjectFromCodeFence(trimmed)?.let { candidates += it }
    extractFirstJsonObject(trimmed)?.let { candidates += it }

    for (candidate in candidates) {
        val parsed = runCatching { json.parseToJsonElement(candidate) }.getOrNull() ?: continue
        val obj = parsed as? JsonObject ?: continue
        val answer = (obj["answer"] as? JsonPrimitive)?.contentOrNull?.trim()
        if (!answer.isNullOrBlank()) {
            return answer
        }
    }
    return null
}

private fun extractJsonObjectFromCodeFence(text: String): String? {
    val match = JSON_CODE_FENCE_REGEX.matchEntire(text) ?: return null
    return match.groupValues
        .getOrNull(1)
        ?.trim()
        ?.takeIf { it.isNotBlank() }
}

private fun extractFirstJsonObject(text: String): String? {
    val start = text.indexOf('{')
    if (start < 0) return null

    var depth = 0
    var inString = false
    var escaped = false
    for (i in start until text.length) {
        val ch = text[i]
        if (inString) {
            when {
                escaped -> escaped = false
                ch == '\\' -> escaped = true
                ch == '"' -> inString = false
            }
            continue
        }

        when (ch) {
            '"' -> {
                inString = true
            }

            '{' -> {
                depth += 1
            }

            '}' -> {
                depth -= 1
                if (depth == 0) {
                    return text.substring(start, i + 1)
                }
            }
        }
    }
    return null
}
