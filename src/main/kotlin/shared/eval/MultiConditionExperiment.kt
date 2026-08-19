package shared.eval

import causalhippo.CausalHippoQueryParam
import causalhippo.CausalHippoRAG
import causalrag.CausalRAG
import causalrag.examples.CliUtils
import causalrag.generator.llm.LLMInterface
import causalrag.generator.promptbuilder.buildPrompt
import hipporag.HippoRAG
import hipporag.config.BaseConfig
import io.github.ugaikit.bertscore.BertScore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import ragas.metrics.collections.ResponseGroundednessMetric
import ragas.metrics.defaults.FaithfulnessMetric
import ragas.model.SingleTurnSample
import shared.rag.unified.RagId
import shared.rag.unified.UnifiedMode
import shared.rag.unified.UnifiedQuery
import shared.rag.unified.UnifiedRagFactory
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.io.path.createDirectories
import kotlin.math.log2
import kotlin.math.max
import causalrag.QueryParam as CausalQueryParam
import hipporag.QueryParam as HippoQueryParam

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
    val retrievedContexts: List<String> = emptyList(),
    val error: String? = null,
)

private enum class Condition(
    val id: String,
    val description: String,
) {
    CAUSALRAG_FIXED(
        id = "causalrag_fixed",
        description = "CausalRAG fixed weights",
    ),
    CAUSALRAG_ADAPT(
        id = "causalrag_adapt",
        description = "CausalRAG dynamicWeighting + twoPassAdaptive + confidenceSwitch",
    ),
    HIPPORAG_GRAPH(
        id = "hipporag_graph",
        description = "HippoRAG graph retrieval",
    ),
    HIPPORAG_DPR(
        id = "hipporag_dpr",
        description = "HippoRAG DPR retrieval",
    ),
    CAUSALHIPPO_FIXED(
        id = "causalhippo_fixed",
        description = "CausalHippoRAG fixed weights",
    ),
    CAUSALHIPPO_ADAPTIVE(
        id = "causalhippo_adaptive",
        description = "CausalHippoRAG dynamicWeighting + twoPassAdaptive + confidenceSwitch",
    ),
    CAUSALHIPPO_ABLATION_NO_RERANK(
        id = "causalhippo_ablation_no_rerank",
        description = "CausalHippoRAG ablation (no causal reranker)",
    ),
    YOUTURAG(
        id = "youturag",
        description = "YoutuRAG (com/youtu/graphrag, unified adapter)",
    ),
    ;

    companion object {
        fun parse(raw: String): List<Condition> {
            if (raw == "all") return entries

            fun normalizeConditionId(id: String): String =
                when (id.trim().lowercase()) {
                    "youtu", "youtu_rag", "youtu-rag" -> "youturag"
                    else -> id.trim().lowercase()
                }
            val wanted =
                raw
                    .split(',')
                    .map { normalizeConditionId(it) }
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
    val configPath: String?,
    val manifestPath: Path?,
    val conditions: List<Condition>,
    val topK: Int,
    val llmModel: String,
    val embeddingModel: String,
    val llmProvider: String,
    val llmBaseUrl: String?,
    val templateStyle: String,
    val limit: Int?,
    val parallelism: Int,
    val useUnifiedApi: Boolean,
    val useUnifiedPersistence: Boolean,
    // SWO69 E2: root dir of per-sample perturbed snapshots (<root>/<sampleId>/ =
    // HippoRAG snapshot dir with working_dir/); when set, the hipporag_graph
    // condition loads the snapshot instead of building (mirrors the graph
    // runner's --e2-snapshot-root, see MultiConditionGraphExperiment.kt).
    val e2SnapshotRoot: Path?,
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
        printLegacyModeDeprecationWarning("shared.eval.MultiConditionExperimentKt")
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
        retrievedContexts = retrieval.context,
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
        Condition.CAUSALRAG_FIXED -> {
            createCausalRagRunner(
                config = config,
                dynamic = false,
                twoPass = false,
                confidence = false,
            )
        }

        Condition.CAUSALRAG_ADAPT -> {
            createCausalRagRunner(
                config = config,
                dynamic = true,
                twoPass = true,
                confidence = true,
            )
        }

        Condition.HIPPORAG_GRAPH -> {
            createHippoRunner(config, useDpr = false, workdirSuffix = scopedWorkdirSuffix("graph", workerIndex))
        }

        Condition.HIPPORAG_DPR -> {
            createHippoRunner(config, useDpr = true, workdirSuffix = scopedWorkdirSuffix("dpr", workerIndex))
        }

        Condition.CAUSALHIPPO_FIXED -> {
            createCausalHippoRunner(
                config = config,
                dynamic = false,
                twoPass = false,
                confidence = false,
                workdirSuffix = scopedWorkdirSuffix("hippocausal_fixed", workerIndex),
            )
        }

        Condition.CAUSALHIPPO_ADAPTIVE -> {
            createCausalHippoRunner(
                config = config,
                dynamic = true,
                twoPass = true,
                confidence = true,
                workdirSuffix = scopedWorkdirSuffix("hippocausal_adaptive", workerIndex),
            )
        }

        Condition.CAUSALHIPPO_ABLATION_NO_RERANK -> {
            createCausalHippoAblationRunner(config, scopedWorkdirSuffix("ablation", workerIndex))
        }

        Condition.YOUTURAG -> {
            error("Condition 'youturag' requires --use-unified-api=true")
        }
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
    val handle =
        UnifiedRagFactory.create(
            ragId = condition.toUnifiedRagId(),
            configPath = config.configPath,
            overrides = buildUnifiedOverrides(condition, config, workdirRoot),
        )
    val rag = handle.rag
    val ablationLlm =
        if (condition == Condition.CAUSALHIPPO_ABLATION_NO_RERANK) {
            LLMInterface(
                modelName = config.llmModel,
                provider = config.llmProvider,
                baseUrl = config.llmBaseUrl,
            )
        } else {
            null
        }

    return object : ConditionRunner {
        override val id: String = condition.id

        override fun run(
            sample: ExperimentSample,
            docs: List<String>,
        ): RetrievalAndAnswer {
            val indexStart = System.nanoTime()
            rag.drop()
            rag.upsert(docs.filter { it.isNotBlank() })
            val indexMs = elapsedMs(indexStart)

            val queryStart = System.nanoTime()
            val queryResult = rag.query(sample.question, buildUnifiedQuery(condition, config))
            val context =
                queryResult.context
                    .map { it.text.trim() }
                    .filter { it.isNotBlank() }
                    .take(config.topK)
            val prediction =
                if (condition == Condition.CAUSALHIPPO_ABLATION_NO_RERANK) {
                    val llm = requireNotNull(ablationLlm) { "Ablation LLM must be initialized." }
                    val prompt =
                        buildPrompt(
                            sample.question,
                            context,
                            causalPaths = queryResult.graphPaths,
                            causalNodes = emptyList(),
                            templateStyle = config.templateStyle,
                            llmInterface = llm,
                        )
                    llm.generate(prompt, jsonMode = requiresJsonResponseFormat(config.templateStyle))
                } else {
                    queryResult.answer.orEmpty().trim()
                }
            val queryMs = elapsedMs(queryStart)

            return RetrievalAndAnswer(
                context = context,
                prediction = prediction,
                indexLatencyMs = indexMs,
                queryLatencyMs = queryMs,
                totalLatencyMs = indexMs + queryMs,
            )
        }
    }
}

private fun Condition.toUnifiedRagId(): RagId =
    when (this) {
        Condition.CAUSALRAG_FIXED,
        Condition.CAUSALRAG_ADAPT,
        -> RagId.CAUSAL_RAG

        Condition.HIPPORAG_GRAPH,
        Condition.HIPPORAG_DPR,
        -> RagId.HIPPO_RAG

        Condition.CAUSALHIPPO_FIXED,
        Condition.CAUSALHIPPO_ADAPTIVE,
        Condition.CAUSALHIPPO_ABLATION_NO_RERANK,
        -> RagId.CAUSAL_HIPPO_RAG

        Condition.YOUTURAG -> RagId.YOUTU_RAG
    }

private fun buildUnifiedOverrides(
    condition: Condition,
    config: RunConfig,
    workdirRoot: Path,
): Map<String, Any?> {
    val persistenceOverrides: Map<String, Any?> =
        if (config.useUnifiedPersistence) {
            mapOf(
                "useUnifiedPersistence" to true,
                "persistenceBackend" to "filesystem_snapshot",
                "persistenceRootDir" to workdirRoot.resolve("unified_persistence").toString(),
            )
        } else {
            emptyMap()
        }
    val baseHippoOverrides =
        mapOf(
            "saveDir" to workdirRoot.toString(),
            "llmModelName" to config.llmModel,
            "embeddingModelName" to config.embeddingModel,
            "llmProvider" to config.llmProvider,
            "embeddingProvider" to config.llmProvider,
            "llmBaseUrl" to config.llmBaseUrl,
            "embeddingBaseUrl" to config.llmBaseUrl,
            "openAiApiKey" to System.getenv("OPENAI_API_KEY"),
            "retrievalTopK" to config.topK,
            "qaTopK" to config.topK,
        )

    return when (condition) {
        Condition.CAUSALRAG_FIXED -> {
            mapOf(
                "modelName" to config.llmModel,
                "embeddingModel" to config.embeddingModel,
                "templateStyle" to config.templateStyle,
                "dynamicWeightingEnabled" to false,
                "twoPassAdaptiveEnabled" to false,
                "confidenceBasedSwitchEnabled" to false,
            ) + persistenceOverrides
        }

        Condition.CAUSALRAG_ADAPT -> {
            mapOf(
                "modelName" to config.llmModel,
                "embeddingModel" to config.embeddingModel,
                "templateStyle" to config.templateStyle,
                "dynamicWeightingEnabled" to true,
                "twoPassAdaptiveEnabled" to true,
                "confidenceBasedSwitchEnabled" to true,
            ) + persistenceOverrides
        }

        Condition.HIPPORAG_GRAPH,
        Condition.HIPPORAG_DPR,
        -> {
            baseHippoOverrides + persistenceOverrides
        }

        Condition.CAUSALHIPPO_FIXED -> {
            baseHippoOverrides +
                mapOf(
                    "modelName" to config.llmModel,
                    "embeddingModel" to config.embeddingModel,
                    "templateStyle" to config.templateStyle,
                    "dynamicWeightingEnabled" to false,
                    "twoPassAdaptiveEnabled" to false,
                    "confidenceBasedSwitchEnabled" to false,
                ) +
                persistenceOverrides
        }

        Condition.CAUSALHIPPO_ADAPTIVE,
        Condition.CAUSALHIPPO_ABLATION_NO_RERANK,
        -> {
            baseHippoOverrides +
                mapOf(
                    "modelName" to config.llmModel,
                    "embeddingModel" to config.embeddingModel,
                    "templateStyle" to config.templateStyle,
                    "dynamicWeightingEnabled" to true,
                    "twoPassAdaptiveEnabled" to true,
                    "confidenceBasedSwitchEnabled" to true,
                ) +
                persistenceOverrides
        }

        Condition.YOUTURAG -> {
            mapOf(
                "rootDir" to workdirRoot.toString(),
                "datasetName" to "demo",
            ) + persistenceOverrides
        }
    }
}

private fun buildUnifiedQuery(
    condition: Condition,
    config: RunConfig,
): UnifiedQuery =
    when (condition) {
        Condition.CAUSALRAG_FIXED,
        Condition.CAUSALRAG_ADAPT,
        -> {
            UnifiedQuery(mode = UnifiedMode.CAUSAL, topK = config.topK, includeReferences = true)
        }

        Condition.HIPPORAG_GRAPH -> {
            UnifiedQuery(mode = UnifiedMode.GRAPH, topK = config.topK, includeReferences = true)
        }

        Condition.HIPPORAG_DPR -> {
            UnifiedQuery(mode = UnifiedMode.DPR, topK = config.topK, includeReferences = true)
        }

        Condition.CAUSALHIPPO_FIXED,
        Condition.CAUSALHIPPO_ADAPTIVE,
        -> {
            UnifiedQuery(mode = UnifiedMode.CAUSAL, topK = config.topK, includeReferences = true)
        }

        Condition.CAUSALHIPPO_ABLATION_NO_RERANK -> {
            UnifiedQuery(
                mode = UnifiedMode.CAUSAL,
                topK = config.topK,
                includeAnswer = false,
                includeContext = true,
                includeReferences = false,
                includeGraphPaths = true,
                extras = mapOf("maxPaths" to 3),
            )
        }

        Condition.YOUTURAG -> {
            UnifiedQuery(
                mode = UnifiedMode.HYBRID,
                topK = config.topK,
                includeAnswer = true,
                includeContext = true,
                includeReferences = true,
                includeFollowUps = true,
                extras = mapOf("datasetName" to "demo"),
            )
        }
    }

private fun createCausalRagRunner(
    config: RunConfig,
    dynamic: Boolean,
    twoPass: Boolean,
    confidence: Boolean,
): ConditionRunner {
    val rag =
        CausalRAG(
            modelName = config.llmModel,
            embeddingModel = config.embeddingModel,
            configPath = config.configPath,
            templateStyle = config.templateStyle,
            dynamicWeightingEnabled = dynamic,
            twoPassAdaptiveEnabled = twoPass,
            confidenceBasedSwitchEnabled = confidence,
        )
    val conditionId =
        if (dynamic) {
            Condition.CAUSALRAG_ADAPT.id
        } else {
            Condition.CAUSALRAG_FIXED.id
        }
    return object : ConditionRunner {
        override val id: String = conditionId

        override fun run(
            sample: ExperimentSample,
            docs: List<String>,
        ): RetrievalAndAnswer {
            val indexStart = System.nanoTime()
            rag.drop()
            rag.upsert(docs)
            val indexMs = elapsedMs(indexStart)

            val queryStart = System.nanoTime()
            val result = rag.query(sample.question, CausalQueryParam(topK = config.topK))
            val queryMs = elapsedMs(queryStart)

            return RetrievalAndAnswer(
                context = result.context.take(config.topK),
                prediction = result.answer,
                indexLatencyMs = indexMs,
                queryLatencyMs = queryMs,
                totalLatencyMs = indexMs + queryMs,
            )
        }
    }
}

private fun createHippoRunner(
    config: RunConfig,
    useDpr: Boolean,
    workdirSuffix: String,
): ConditionRunner {
    val hippo =
        HippoRAG(
            config =
                BaseConfig(
                    llmName = config.llmModel,
                    embeddingModelName = config.embeddingModel,
                    llmProvider = config.llmProvider,
                    embeddingProvider = config.llmProvider,
                    llmBaseUrl = config.llmBaseUrl,
                    embeddingBaseUrl = config.llmBaseUrl,
                    saveDir =
                        config.outputDir
                            .resolve("workdirs")
                            .resolve(workdirSuffix)
                            .toString(),
                    retrievalTopK = config.topK,
                    qaTopK = config.topK,
                ),
        )
    return object : ConditionRunner {
        override val id: String = if (useDpr) Condition.HIPPORAG_DPR.id else Condition.HIPPORAG_GRAPH.id

        override fun run(
            sample: ExperimentSample,
            docs: List<String>,
        ): RetrievalAndAnswer {
            val indexStart = System.nanoTime()
            val cleanedDocs = docs.filter { it.isNotBlank() }
            val e2Snap = config.e2SnapshotRoot?.resolve(sanitizeSampleId(sample.id))
            if (e2Snap != null) {
                // SWO69 E2 (T7): load the perturbed snapshot instead of building.
                // Layout (scripts/perturb_kg.py): <root>/<sampleId>/working_dir/
                // is a full HippoRAG working dir — only graph.json is perturbed,
                // embeddings/openie artifacts are carried over.
                require(Files.isDirectory(e2Snap)) { "E2 snapshot not found: $e2Snap" }
                hippo.drop()
                hippo.loadGraph(e2Snap.toString())
            } else {
                hippo.drop()
                if (cleanedDocs.isNotEmpty()) {
                    hippo.upsert(cleanedDocs)
                    // SWO69 T1: snapshot the per-sample KG — the shared saveDir is dropped
                    // and reused on the next sample, so only the last graph would survive.
                    val snapshotDir =
                        config.outputDir
                            .resolve("workdirs")
                            .resolve(workdirSuffix)
                            .resolve("snapshots")
                            .resolve(sanitizeSampleId(sample.id))
                    runCatching { hippo.saveGraph(snapshotDir.toString()) }
                }
            }
            val indexMs = elapsedMs(indexStart)

            val queryStart = System.nanoTime()
            val result =
                hippo.query(
                    sample.question,
                    HippoQueryParam(
                        mode = if (useDpr) "dpr" else "graph",
                        topK = config.topK,
                        includeAnswer = true,
                    ),
                )
            val queryMs = elapsedMs(queryStart)

            val context = result.docs.take(config.topK)
            val prediction = result.answer.orEmpty()

            return RetrievalAndAnswer(
                context = context,
                prediction = prediction,
                indexLatencyMs = indexMs,
                queryLatencyMs = queryMs,
                totalLatencyMs = indexMs + queryMs,
            )
        }
    }
}

private fun createCausalHippoRunner(
    config: RunConfig,
    dynamic: Boolean,
    twoPass: Boolean,
    confidence: Boolean,
    workdirSuffix: String,
): ConditionRunner {
    val rag =
        CausalHippoRAG(
            modelName = config.llmModel,
            embeddingModel = config.embeddingModel,
            configPath = config.configPath,
            templateStyle = config.templateStyle,
            hippoConfig =
                BaseConfig(
                    llmName = config.llmModel,
                    embeddingModelName = config.embeddingModel,
                    llmProvider = config.llmProvider,
                    embeddingProvider = config.llmProvider,
                    llmBaseUrl = config.llmBaseUrl,
                    embeddingBaseUrl = config.llmBaseUrl,
                    saveDir =
                        config.outputDir
                            .resolve("workdirs")
                            .resolve(workdirSuffix)
                            .toString(),
                    retrievalTopK = config.topK,
                    qaTopK = config.topK,
                ),
            dynamicWeightingEnabled = dynamic,
            twoPassAdaptiveEnabled = twoPass,
            confidenceBasedSwitchEnabled = confidence,
        )
    val conditionId =
        if (dynamic) {
            Condition.CAUSALHIPPO_ADAPTIVE.id
        } else {
            Condition.CAUSALHIPPO_FIXED.id
        }

    return object : ConditionRunner {
        override val id: String = conditionId

        override fun run(
            sample: ExperimentSample,
            docs: List<String>,
        ): RetrievalAndAnswer {
            val indexStart = System.nanoTime()
            rag.drop()
            rag.upsert(docs)
            val indexMs = elapsedMs(indexStart)

            val queryStart = System.nanoTime()
            val result = rag.query(sample.question, CausalHippoQueryParam(topK = config.topK))
            val queryMs = elapsedMs(queryStart)

            return RetrievalAndAnswer(
                context = result.context.take(config.topK),
                prediction = result.answer,
                indexLatencyMs = indexMs,
                queryLatencyMs = queryMs,
                totalLatencyMs = indexMs + queryMs,
            )
        }
    }
}

private fun createCausalHippoAblationRunner(
    config: RunConfig,
    workdirSuffix: String,
): ConditionRunner {
    val rag =
        CausalHippoRAG(
            modelName = config.llmModel,
            embeddingModel = config.embeddingModel,
            configPath = config.configPath,
            templateStyle = config.templateStyle,
            hippoConfig =
                BaseConfig(
                    llmName = config.llmModel,
                    embeddingModelName = config.embeddingModel,
                    llmProvider = config.llmProvider,
                    embeddingProvider = config.llmProvider,
                    llmBaseUrl = config.llmBaseUrl,
                    embeddingBaseUrl = config.llmBaseUrl,
                    saveDir =
                        config.outputDir
                            .resolve("workdirs")
                            .resolve(workdirSuffix)
                            .toString(),
                    retrievalTopK = config.topK,
                    qaTopK = config.topK,
                ),
            dynamicWeightingEnabled = true,
            twoPassAdaptiveEnabled = true,
            confidenceBasedSwitchEnabled = true,
        )
    val llm =
        LLMInterface(
            modelName = config.llmModel,
            provider = config.llmProvider,
            baseUrl = config.llmBaseUrl,
        )
    return object : ConditionRunner {
        override val id: String = Condition.CAUSALHIPPO_ABLATION_NO_RERANK.id

        override fun run(
            sample: ExperimentSample,
            docs: List<String>,
        ): RetrievalAndAnswer {
            val indexStart = System.nanoTime()
            rag.drop()
            rag.upsert(docs)
            val indexMs = elapsedMs(indexStart)

            val queryStart = System.nanoTime()
            val contextResult =
                rag.query(
                    sample.question,
                    CausalHippoQueryParam(
                        topK = config.topK,
                        onlyNeedContext = true,
                    ),
                )
            val pathsResult =
                rag.query(
                    sample.question,
                    CausalHippoQueryParam(
                        maxPaths = 3,
                        onlyNeedCausalPaths = true,
                    ),
                )
            val context = contextResult.context.take(config.topK)
            val causalPaths = pathsResult.causalPaths
            val causalNodes = emptyList<String>()
            val prompt =
                buildPrompt(
                    sample.question,
                    context,
                    causalPaths = causalPaths,
                    causalNodes = causalNodes,
                    templateStyle = config.templateStyle,
                    llmInterface = llm,
                )
            val prediction = llm.generate(prompt, jsonMode = requiresJsonResponseFormat(config.templateStyle))
            val queryMs = elapsedMs(queryStart)

            return RetrievalAndAnswer(
                context = context,
                prediction = prediction,
                indexLatencyMs = indexMs,
                queryLatencyMs = queryMs,
                totalLatencyMs = indexMs + queryMs,
            )
        }
    }
}

private fun elapsedMs(startNanos: Long): Double = (System.nanoTime() - startNanos).toDouble() / 1_000_000.0

private fun parseArgs(args: Array<String>): RunConfig {
    val opts = CliUtils.parseOptions(args.toList())

    val dataPath = Path.of(opts["data"] ?: "data/musique_experiment/musique_dev_balanced_300.jsonl")
    require(Files.exists(dataPath)) { "Missing --data file: $dataPath" }

    val outputDir =
        Path.of(
            opts["output-dir"]
                ?: Path.of("eval_results", "multicondition_${Instant.now().toString().replace(':', '_')}").toString(),
        )

    val configPath = opts["config"]
    val conditions = Condition.parse(opts["conditions"] ?: "all")
    val topK = (opts["top-k"] ?: "5").toIntOrNull() ?: 5
    require(topK > 0) { "--top-k must be positive" }

    val llmModel = opts["llm-model"] ?: System.getenv("LLM_MODEL") ?: "gpt-5.4-mini"
    val embeddingModel = opts["embedding-model"] ?: System.getenv("EMBEDDING_MODEL") ?: "text-embedding-3-small"
    val llmProvider = (opts["provider"] ?: System.getenv("LLM_PROVIDER") ?: "openai").lowercase()
    val llmBaseUrl = opts["llm-base-url"] ?: System.getenv("LLM_BASE_URL")
    val templateStyle = opts["template-style"] ?: "detailed_musique"
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

    // SWO69 E2 (T7): load per-sample perturbed snapshots instead of building.
    val e2SnapshotRoot =
        opts["e2-snapshot-root"]?.let { raw ->
            val path = Path.of(raw)
            require(Files.isDirectory(path)) { "Missing --e2-snapshot-root directory: $path" }
            path
        }

    return RunConfig(
        dataPath = dataPath,
        outputDir = outputDir,
        configPath = configPath,
        manifestPath = manifestPath,
        conditions = conditions,
        topK = topK,
        llmModel = llmModel,
        embeddingModel = embeddingModel,
        llmProvider = llmProvider,
        llmBaseUrl = llmBaseUrl,
        templateStyle = templateStyle,
        limit = limit,
        parallelism = parallelism,
        useUnifiedApi = useUnifiedApi,
        useUnifiedPersistence = useUnifiedPersistence,
        e2SnapshotRoot = e2SnapshotRoot,
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

private fun sanitizeSampleId(sampleId: String): String = sampleId.replace(Regex("[^A-Za-z0-9._-]"), "_")

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

private fun requiresJsonResponseFormat(style: String): Boolean = style.equals("experiments", ignoreCase = true)
