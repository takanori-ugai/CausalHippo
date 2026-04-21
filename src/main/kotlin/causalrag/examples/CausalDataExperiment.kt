package causalrag.examples

import causalrag.CausalRAGPipeline
import causalrag.HippoCausalRAGPipeline
import causalrag.generator.promptbuilder.buildPrompt
import causalrag.retriever.HippoRagSemanticMode
import hipporag.HippoRag
import hipporag.config.BaseConfig
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.apache.commons.csv.CSVFormat
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant
import kotlin.io.path.createDirectories
import kotlin.io.path.extension
import kotlin.io.path.isRegularFile
import kotlin.io.path.name
import kotlin.math.max

private val json = Json { ignoreUnknownKeys = true }
private val trueWord = Regex("\\b(true|yes)\\b")
private val falseWord = Regex("\\b(false|no)\\b")

@Serializable
private data class CausalCsvSample(
    val id: String,
    val sourceFile: String,
    val category: String,
    val e1: String,
    val e2: String,
    val paragraph: String,
    val question: String,
    val goldLabel: Boolean,
)

@Serializable
private data class CausalPerQuestionResult(
    val condition: String,
    val sampleId: String,
    val sourceFile: String,
    val category: String,
    val goldLabel: Boolean,
    val predictedLabel: Boolean?,
    val correct: Double,
    val indexLatencyMs: Double,
    val queryLatencyMs: Double,
    val totalLatencyMs: Double,
    val prediction: String,
    val error: String? = null,
)

private data class CausalConditionSummary(
    val condition: String,
    val category: String,
    val count: Int,
    val parsed: Int,
    val correct: Int,
    val accuracy: Double,
    val precision: Double,
    val recall: Double,
    val f1: Double,
    val tp: Int,
    val tn: Int,
    val fp: Int,
    val fn: Int,
)

private enum class CausalCondition(
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
    ;

    companion object {
        fun parse(raw: String): List<CausalCondition> {
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

private data class CausalRunConfig(
    val dataDir: Path,
    val dataGlob: String,
    val outputDir: Path,
    val configPath: String?,
    val conditions: List<CausalCondition>,
    val topK: Int,
    val llmModel: String,
    val embeddingModel: String,
    val llmProvider: String,
    val llmBaseUrl: String?,
    val templateStyle: String,
    val limit: Int?,
)

private data class CausalRetrievalAndAnswer(
    val prediction: String,
    val indexLatencyMs: Double,
    val queryLatencyMs: Double,
    val totalLatencyMs: Double,
)

@Suppress("TooGenericExceptionCaught")
fun main(args: Array<String>) {
    val config = parseArgs(args)
    val samples = loadCausalCsvSamples(config.dataDir, config.dataGlob, config.limit)
    require(samples.isNotEmpty()) {
        "No samples found in ${config.dataDir} matching ${config.dataGlob}"
    }

    config.outputDir.createDirectories()
    val perQuestionDir = config.outputDir.resolve("per_question")
    perQuestionDir.createDirectories()

    println("Running ${config.conditions.size} conditions on ${samples.size} causal samples")
    println("Data dir: ${config.dataDir} (glob=${config.dataGlob})")
    println("Output: ${config.outputDir}")

    val allRows = mutableListOf<CausalPerQuestionResult>()

    for (condition in config.conditions) {
        println("\\n=== Condition: ${condition.id} (${condition.description}) ===")
        val rows = mutableListOf<CausalPerQuestionResult>()
        val runner = createConditionRunner(condition, config)

        for ((index, sample) in samples.withIndex()) {
            val row =
                try {
                    runConditionForSample(runner, sample)
                } catch (ex: Exception) {
                    CausalPerQuestionResult(
                        condition = condition.id,
                        sampleId = sample.id,
                        sourceFile = sample.sourceFile,
                        category = sample.category,
                        goldLabel = sample.goldLabel,
                        predictedLabel = null,
                        correct = 0.0,
                        indexLatencyMs = 0.0,
                        queryLatencyMs = 0.0,
                        totalLatencyMs = 0.0,
                        prediction = "",
                        error = ex.message ?: ex::class.simpleName,
                    )
                }

            rows += row
            val done = index + 1
            if (done % 20 == 0 || done == samples.size) {
                println("${condition.id}: processed $done/${samples.size}")
            }
        }

        val jsonlPath = perQuestionDir.resolve("${condition.id}.jsonl")
        writeJsonl(jsonlPath, rows)
        println("Wrote ${rows.size} rows -> $jsonlPath")
        allRows += rows
    }

    val overallSummary = summarizeRows(allRows, byCategory = false)
    val categorySummary = summarizeRows(allRows, byCategory = true)

    val summaryByConditionPath = config.outputDir.resolve("summary_by_condition.csv")
    val summaryByCategoryPath = config.outputDir.resolve("summary_by_condition_category.csv")
    val perQuestionCsvPath = config.outputDir.resolve("per_question_metrics.csv")

    writeSummaryCsv(summaryByConditionPath, overallSummary)
    writeSummaryCsv(summaryByCategoryPath, categorySummary)
    writePerQuestionCsv(perQuestionCsvPath, allRows)

    println("\\nFinished causal evaluation")
    println("Per-question JSONL: $perQuestionDir")
    println("Summary by condition: $summaryByConditionPath")
    println("Summary by condition/category: $summaryByCategoryPath")
    println("Per-question CSV: $perQuestionCsvPath")
}

private fun runConditionForSample(
    runner: CausalConditionRunner,
    sample: CausalCsvSample,
): CausalPerQuestionResult {
    val retrieval = runner.run(sample)
    val predictedLabel = parseBooleanPrediction(retrieval.prediction)
    val correct = if (predictedLabel != null && predictedLabel == sample.goldLabel) 1.0 else 0.0

    return CausalPerQuestionResult(
        condition = runner.id,
        sampleId = sample.id,
        sourceFile = sample.sourceFile,
        category = sample.category,
        goldLabel = sample.goldLabel,
        predictedLabel = predictedLabel,
        correct = correct,
        indexLatencyMs = retrieval.indexLatencyMs,
        queryLatencyMs = retrieval.queryLatencyMs,
        totalLatencyMs = retrieval.totalLatencyMs,
        prediction = retrieval.prediction,
        error = null,
    )
}

private interface CausalConditionRunner {
    val id: String

    fun run(sample: CausalCsvSample): CausalRetrievalAndAnswer
}

private fun createConditionRunner(
    condition: CausalCondition,
    config: CausalRunConfig,
): CausalConditionRunner =
    when (condition) {
        CausalCondition.CAUSALRAG_FIXED -> {
            createCausalRagRunner(
                config = config,
                dynamic = false,
                twoPass = false,
                confidence = false,
            )
        }

        CausalCondition.CAUSALRAG_ADAPT -> {
            createCausalRagRunner(
                config = config,
                dynamic = true,
                twoPass = true,
                confidence = true,
            )
        }

        CausalCondition.HIPPORAG_GRAPH -> {
            createHippoRunner(config, useDpr = false, workdirSuffix = "graph")
        }

        CausalCondition.HIPPORAG_DPR -> {
            createHippoRunner(config, useDpr = true, workdirSuffix = "dpr")
        }

        CausalCondition.CAUSALHIPPO_FIXED -> {
            createCausalHippoRunner(
                config = config,
                dynamic = false,
                twoPass = false,
                confidence = false,
                workdirSuffix = "hippocausal_fixed",
            )
        }

        CausalCondition.CAUSALHIPPO_ADAPTIVE -> {
            createCausalHippoRunner(
                config = config,
                dynamic = true,
                twoPass = true,
                confidence = true,
                workdirSuffix = "hippocausal_adaptive",
            )
        }

        CausalCondition.CAUSALHIPPO_ABLATION_NO_RERANK -> {
            createCausalHippoAblationRunner(config)
        }
    }

private fun createCausalRagRunner(
    config: CausalRunConfig,
    dynamic: Boolean,
    twoPass: Boolean,
    confidence: Boolean,
): CausalConditionRunner {
    val pipeline =
        CausalRAGPipeline(
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
            CausalCondition.CAUSALRAG_ADAPT.id
        } else {
            CausalCondition.CAUSALRAG_FIXED.id
        }

    return object : CausalConditionRunner {
        override val id: String = conditionId

        override fun run(sample: CausalCsvSample): CausalRetrievalAndAnswer {
            val indexStart = System.nanoTime()
            pipeline.reindex(listOf(sample.paragraph))
            val indexMs = elapsedMs(indexStart)

            val queryStart = System.nanoTime()
            val result = pipeline.runWithContext(sample.question, topK = config.topK)
            val queryMs = elapsedMs(queryStart)

            return CausalRetrievalAndAnswer(
                prediction = result.answer,
                indexLatencyMs = indexMs,
                queryLatencyMs = queryMs,
                totalLatencyMs = indexMs + queryMs,
            )
        }
    }
}

private fun createHippoRunner(
    config: CausalRunConfig,
    useDpr: Boolean,
    workdirSuffix: String,
): CausalConditionRunner {
    val hippo =
        HippoRag(
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

    var previousDocs: List<String> = emptyList()

    return object : CausalConditionRunner {
        override val id: String =
            if (useDpr) {
                CausalCondition.HIPPORAG_DPR.id
            } else {
                CausalCondition.HIPPORAG_GRAPH.id
            }

        override fun run(sample: CausalCsvSample): CausalRetrievalAndAnswer {
            val indexStart = System.nanoTime()
            if (previousDocs.isNotEmpty()) {
                hippo.delete(previousDocs)
            }
            hippo.index(listOf(sample.paragraph))
            previousDocs = listOf(sample.paragraph).filter { it.isNotBlank() }
            val indexMs = elapsedMs(indexStart)

            val queryStart = System.nanoTime()
            val result =
                if (useDpr) {
                    hippo.ragQaDpr(queries = listOf(sample.question), goldDocs = null, goldAnswers = null)
                } else {
                    hippo.ragQa(queries = listOf(sample.question), goldDocs = null, goldAnswers = null)
                }
            val queryMs = elapsedMs(queryStart)
            val prediction =
                result.solutions
                    .firstOrNull()
                    ?.answer
                    .orEmpty()

            return CausalRetrievalAndAnswer(
                prediction = prediction,
                indexLatencyMs = indexMs,
                queryLatencyMs = queryMs,
                totalLatencyMs = indexMs + queryMs,
            )
        }
    }
}

private fun createCausalHippoRunner(
    config: CausalRunConfig,
    dynamic: Boolean,
    twoPass: Boolean,
    confidence: Boolean,
    workdirSuffix: String,
): CausalConditionRunner {
    val pipeline =
        HippoCausalRAGPipeline(
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
            hippoSemanticMode = HippoRagSemanticMode.GRAPH,
            dynamicWeightingEnabled = dynamic,
            twoPassAdaptiveEnabled = twoPass,
            confidenceBasedSwitchEnabled = confidence,
        )

    val conditionId =
        if (dynamic) {
            CausalCondition.CAUSALHIPPO_ADAPTIVE.id
        } else {
            CausalCondition.CAUSALHIPPO_FIXED.id
        }

    return object : CausalConditionRunner {
        override val id: String = conditionId

        override fun run(sample: CausalCsvSample): CausalRetrievalAndAnswer {
            val indexStart = System.nanoTime()
            pipeline.reindex(listOf(sample.paragraph))
            val indexMs = elapsedMs(indexStart)

            val queryStart = System.nanoTime()
            val result = pipeline.runWithContext(sample.question, topK = config.topK)
            val queryMs = elapsedMs(queryStart)

            return CausalRetrievalAndAnswer(
                prediction = result.answer,
                indexLatencyMs = indexMs,
                queryLatencyMs = queryMs,
                totalLatencyMs = indexMs + queryMs,
            )
        }
    }
}

private fun createCausalHippoAblationRunner(config: CausalRunConfig): CausalConditionRunner {
    val pipeline =
        HippoCausalRAGPipeline(
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
                            .resolve("ablation")
                            .toString(),
                    retrievalTopK = config.topK,
                    qaTopK = config.topK,
                ),
            hippoSemanticMode = HippoRagSemanticMode.GRAPH,
            dynamicWeightingEnabled = true,
            twoPassAdaptiveEnabled = true,
            confidenceBasedSwitchEnabled = true,
        )

    return object : CausalConditionRunner {
        override val id: String = CausalCondition.CAUSALHIPPO_ABLATION_NO_RERANK.id

        override fun run(sample: CausalCsvSample): CausalRetrievalAndAnswer {
            val indexStart = System.nanoTime()
            pipeline.reindex(listOf(sample.paragraph))
            val indexMs = elapsedMs(indexStart)

            val queryStart = System.nanoTime()
            val candidateDetails = pipeline.hybridRetriever.retrieveWithDetails(sample.question, topK = config.topK)
            val context = candidateDetails.map { it["passage"] as String }.take(config.topK)
            val causalNodes = pipeline.graphRetriever.retrievePathNodes(sample.question)
            val causalPaths = pipeline.graphRetriever.retrievePaths(sample.question, maxPaths = 3)
            val prompt =
                buildPrompt(
                    sample.question,
                    context,
                    causalPaths = causalPaths,
                    causalNodes = causalNodes,
                    templateStyle = config.templateStyle,
                    llmInterface = pipeline.llm,
                )
            val prediction = pipeline.llm.generate(prompt)
            val queryMs = elapsedMs(queryStart)

            return CausalRetrievalAndAnswer(
                prediction = prediction,
                indexLatencyMs = indexMs,
                queryLatencyMs = queryMs,
                totalLatencyMs = indexMs + queryMs,
            )
        }
    }
}

private fun parseArgs(args: Array<String>): CausalRunConfig {
    val opts = CliUtils.parseOptions(args.toList())

    val dataDir = Path.of(opts["data-dir"] ?: "eval/causal")
    require(Files.exists(dataDir) && Files.isDirectory(dataDir)) {
        "Missing --data-dir directory: $dataDir"
    }

    val dataGlob = opts["data-glob"] ?: "Data-*.csv"

    val defaultOutputDir =
        Path.of(
            "eval_results",
            "causal_multicondition_${Instant.now().toString().replace(':', '_')}",
        )
    val outputDirRaw =
        opts["output-dir"] ?: defaultOutputDir.toString()
    val outputDir = Path.of(outputDirRaw)

    val configPath = opts["config"]
    val conditions = CausalCondition.parse(opts["conditions"] ?: "all")
    val topK = (opts["top-k"] ?: "5").toIntOrNull() ?: 5
    require(topK > 0) { "--top-k must be positive" }

    val llmModel = opts["llm-model"] ?: System.getenv("LLM_MODEL") ?: "gpt-5.4-mini"
    val embeddingModel = opts["embedding-model"] ?: System.getenv("EMBEDDING_MODEL") ?: "text-embedding-3-small"
    val llmProvider = (opts["provider"] ?: System.getenv("LLM_PROVIDER") ?: "openai").lowercase()
    val llmBaseUrl = opts["llm-base-url"] ?: System.getenv("LLM_BASE_URL")
    val templateStyle = opts["template-style"] ?: "detailed"
    val limit = opts["limit"]?.toIntOrNull()

    return CausalRunConfig(
        dataDir = dataDir,
        dataGlob = dataGlob,
        outputDir = outputDir,
        configPath = configPath,
        conditions = conditions,
        topK = topK,
        llmModel = llmModel,
        embeddingModel = embeddingModel,
        llmProvider = llmProvider,
        llmBaseUrl = llmBaseUrl,
        templateStyle = templateStyle,
        limit = limit,
    )
}

private fun loadCausalCsvSamples(
    dataDir: Path,
    dataGlob: String,
    limit: Int?,
): List<CausalCsvSample> {
    val matcher = dataDir.fileSystem.getPathMatcher("glob:$dataGlob")
    val files =
        Files.list(dataDir).use { stream ->
            val filtered =
                stream.filter {
                    it.isRegularFile() &&
                        it.extension.lowercase() == "csv" &&
                        matcher.matches(it.fileName)
                }
            val sorted = filtered.sorted { a, b -> a.name.compareTo(b.name) }
            sorted.toList()
        }

    val samples = mutableListOf<CausalCsvSample>()

    for (csvFile in files) {
        csvFile.toFile().bufferedReader().use { reader ->
            CSVFormat.DEFAULT.parse(reader).use { records ->
                var rowIndex = 0
                for (fields in records) {
                    if ((0 until fields.size()).none { i -> fields[i].isNotBlank() }) {
                        continue
                    }

                    require(fields.size() >= 4) {
                        "${csvFile.name} row $rowIndex expected at least 4 fields (e1,e2,paragraph,answer[,category]), got ${fields.size()}"
                    }

                    val e1 = fields[0].trim()
                    val e2 = fields[1].trim()
                    val paragraph = stripEntityTags(fields[2].trim())
                    val goldLabel = parseCsvBoolean(fields[3])
                    val category = if (fields.size() > 4) fields[4].trim() else ""

                    samples +=
                        CausalCsvSample(
                            id = "${csvFile.fileName.toString().removeSuffix(".csv")}_$rowIndex",
                            sourceFile = csvFile.fileName.toString(),
                            category = category.ifBlank { "unknown" },
                            e1 = e1,
                            e2 = e2,
                            paragraph = paragraph,
                            question = "Does $e1 cause $e2? Answer only true or false.",
                            goldLabel = goldLabel,
                        )
                    rowIndex++

                    if (limit != null && samples.size >= limit) {
                        return samples
                    }
                }
            }
        }
    }

    return samples
}

private fun stripEntityTags(text: String): String = text.replace(Regex("</?e[12]>"), "")

private fun parseCsvBoolean(raw: String): Boolean = raw.trim().equals("true", ignoreCase = true)

private fun parseBooleanPrediction(text: String): Boolean? {
    val normalized = text.lowercase()
    val trueMatch = trueWord.find(normalized)
    val falseMatch = falseWord.find(normalized)

    return when {
        trueMatch != null && falseMatch == null -> {
            true
        }

        trueMatch == null && falseMatch != null -> {
            false
        }

        trueMatch != null && falseMatch != null -> {
            if (trueMatch.range.first < falseMatch.range.first) {
                true
            } else {
                false
            }
        }

        else -> {
            null
        }
    }
}

private fun writeJsonl(
    path: Path,
    rows: List<CausalPerQuestionResult>,
) {
    Files.newBufferedWriter(path).use { writer ->
        rows.forEach { row ->
            writer.write(json.encodeToString(CausalPerQuestionResult.serializer(), row))
            writer.newLine()
        }
    }
}

private fun summarizeRows(
    rows: List<CausalPerQuestionResult>,
    byCategory: Boolean,
): List<CausalConditionSummary> {
    if (rows.isEmpty()) return emptyList()

    val grouped =
        if (byCategory) {
            rows.groupBy { it.condition to it.category }
        } else {
            rows.groupBy { it.condition to "all" }
        }

    return grouped
        .map { (key, groupRows) ->
            val (condition, category) = key
            var tp = 0
            var tn = 0
            var fp = 0
            var fn = 0
            var parsed = 0

            for (row in groupRows) {
                val pred = row.predictedLabel
                if (pred == null) continue
                parsed += 1
                when {
                    row.goldLabel && pred -> tp += 1
                    !row.goldLabel && !pred -> tn += 1
                    !row.goldLabel && pred -> fp += 1
                    row.goldLabel && !pred -> fn += 1
                }
            }

            val count = groupRows.size
            val correct = groupRows.count { it.correct >= 0.5 }
            val accuracy = correct.toDouble() / max(count, 1)
            val precision = tp.toDouble() / max(tp + fp, 1)
            val recall = tp.toDouble() / max(tp + fn, 1)
            val f1 = if (precision + recall == 0.0) 0.0 else 2 * precision * recall / (precision + recall)

            CausalConditionSummary(
                condition = condition,
                category = category,
                count = count,
                parsed = parsed,
                correct = correct,
                accuracy = accuracy,
                precision = precision,
                recall = recall,
                f1 = f1,
                tp = tp,
                tn = tn,
                fp = fp,
                fn = fn,
            )
        }.sortedWith(compareBy<CausalConditionSummary> { it.condition }.thenBy { it.category })
}

private fun writeSummaryCsv(
    path: Path,
    rows: List<CausalConditionSummary>,
) {
    Files.newBufferedWriter(path).use { writer ->
        writer.write("condition,category,count,parsed,correct,accuracy,precision,recall,f1,tp,tn,fp,fn")
        writer.newLine()
        rows.forEach { row ->
            writer.write(
                listOf(
                    row.condition,
                    row.category,
                    row.count.toString(),
                    row.parsed.toString(),
                    row.correct.toString(),
                    "%.6f".format(row.accuracy),
                    "%.6f".format(row.precision),
                    "%.6f".format(row.recall),
                    "%.6f".format(row.f1),
                    row.tp.toString(),
                    row.tn.toString(),
                    row.fp.toString(),
                    row.fn.toString(),
                ).joinToString(",") { csvEscape(it) },
            )
            writer.newLine()
        }
    }
}

private fun writePerQuestionCsv(
    path: Path,
    rows: List<CausalPerQuestionResult>,
) {
    Files.newBufferedWriter(path).use { writer ->
        writer.write(
            "condition,sample_id,source_file,category,gold_label,predicted_label," +
                "correct,index_latency_ms,query_latency_ms,total_latency_ms,prediction,error",
        )
        writer.newLine()
        rows.forEach { row ->
            writer.write(
                listOf(
                    row.condition,
                    row.sampleId,
                    row.sourceFile,
                    row.category,
                    row.goldLabel.toString(),
                    row.predictedLabel?.toString().orEmpty(),
                    "%.1f".format(row.correct),
                    "%.3f".format(row.indexLatencyMs),
                    "%.3f".format(row.queryLatencyMs),
                    "%.3f".format(row.totalLatencyMs),
                    row.prediction,
                    row.error.orEmpty(),
                ).joinToString(",") { csvEscape(it) },
            )
            writer.newLine()
        }
    }
}

private fun csvEscape(raw: String): String {
    val escaped = raw.replace("\"", "\"\"")
    return "\"$escaped\""
}

private fun elapsedMs(startNanos: Long): Double = (System.nanoTime() - startNanos).toDouble() / 1_000_000.0
