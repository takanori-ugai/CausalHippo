package causalrag.examples

import causalrag.retriever.HippoRagSemanticMode
import dev.langchain4j.model.openai.OpenAiChatModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import ragas.evaluate
import ragas.llms.LangChain4jLlm
import ragas.metrics.collections.AnswerCorrectnessMetric
import ragas.metrics.collections.FactualCorrectnessMetric
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import java.nio.file.Files
import java.nio.file.Path
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.DoubleAdder
import kotlin.io.path.createDirectories

/**
 * Runs MusiQue evaluation using HybridRAG for generation and ragas for scoring.
 */
object MusiQueRagas {
    private val inputJson = Json { ignoreUnknownKeys = true }
    private val prettyJson = Json { prettyPrint = true }

    private data class GeneratedSample(
        val id: String,
        val question: String,
        val prediction: String,
        val reference: String,
        val primaryGold: String,
        val aliases: List<String>,
        val retrievedContexts: List<String>,
        val causalPaths: List<List<String>>,
        val exactMatch: Double,
    )

    /**
     * Environment variables:
     * - LLM_PROVIDER (default: openai)
     * - OPENAI_API_KEY (required when provider=openai)
     * - LLM_MODEL (default: gpt-5.4-mini)
     * - MUSIQUE_EVAL_MODEL (default: same as LLM_MODEL)
     * - EMBEDDING_MODEL (default: text-embedding-3-small)
     * - LLM_BASE_URL (optional)
     * - MUSIQUE_SEMANTIC_MODE (graph|dpr, default: graph)
     * - MUSIQUE_LIMIT (optional)
     * - MUSIQUE_PARALLELISM (default: 5)
     */
    @Suppress("TooGenericExceptionCaught")
    @JvmStatic
    fun main(args: Array<String>) {
        val dataPath = resolveInputPath(args)
        if (!Files.exists(dataPath)) {
            System.err.println("Missing MusiQue data file at $dataPath")
            kotlin.system.exitProcess(1)
        }

        val provider = (System.getenv("LLM_PROVIDER") ?: "openai").lowercase()
        val apiKey = System.getenv("OPENAI_API_KEY")
        if (provider == "openai" && apiKey.isNullOrBlank()) {
            System.err.println("OPENAI_API_KEY is required for provider=openai")
            kotlin.system.exitProcess(1)
        }
        if (apiKey.isNullOrBlank()) {
            System.err.println("OPENAI_API_KEY is required for ragas evaluation")
            kotlin.system.exitProcess(1)
        }

        val modelName = System.getenv("LLM_MODEL") ?: "gpt-5.4-mini"
        val evalModelName = System.getenv("MUSIQUE_EVAL_MODEL") ?: modelName
        val embeddingModel = System.getenv("EMBEDDING_MODEL") ?: "text-embedding-3-small"
        val baseUrl = System.getenv("LLM_BASE_URL")
        val semanticMode = parseSemanticMode(System.getenv("MUSIQUE_SEMANTIC_MODE"))
        val limit = (System.getenv("MUSIQUE_LIMIT") ?: "").toIntOrNull()
        val parallelism = (System.getenv("MUSIQUE_PARALLELISM") ?: "5").toIntOrNull() ?: 5
        require(parallelism > 0) { "MUSIQUE_PARALLELISM must be positive." }

        var sawNonBlankSample = false
        val examples =
            Files.newBufferedReader(dataPath).useLines { lines ->
                if (limit == 0) {
                    sawNonBlankSample = lines.any { it.isNotBlank() }
                    emptyList()
                } else {
                    lines
                        .map { it.trim() }
                        .filter { it.isNotBlank() }
                        .onEach { sawNonBlankSample = true }
                        .map { inputJson.decodeFromString(MusiqueExample.serializer(), it) }
                        .filter { it.answerable }
                        .let { list -> if (limit != null) list.take(limit) else list }
                        .toList()
                }
            }
        if (!sawNonBlankSample) {
            System.err.println("No MusiQue samples found in $dataPath")
            kotlin.system.exitProcess(1)
        }

        if (examples.isEmpty()) {
            System.err.println("No answerable MusiQue samples were processed.")
            kotlin.system.exitProcess(1)
        }

        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
        val resultsDir = Path.of("eval_results").resolve("musique_ragas_$timestamp")
        Files.createDirectories(resultsDir)
        val configPath = writeTempConfig(modelName, embeddingModel, provider, apiKey, baseUrl)

        val generated = mutableListOf<GeneratedSample>()
        val processed = AtomicInteger(0)
        val semaphore = Semaphore(parallelism)

        runBlocking {
            val jobs =
                examples.map { example ->
                    async(Dispatchers.IO) {
                        semaphore.withPermit {
                            try {
                                val documents = example.paragraphs.map { it.paragraphText }
                                val workingDir = sampleWorkingDir(resultsDir, example.id)
                                workingDir.createDirectories()
                                val pipeline =
                                    CliUtils.createHybridPipeline(
                                        configPath = configPath.toString(),
                                        workingDir = workingDir.toString(),
                                        modelName = modelName,
                                        embeddingModel = embeddingModel,
                                        semanticMode = semanticMode,
                                    )

                                pipeline.index(documents)
                                val result = pipeline.runWithContext(example.question, topK = 10)
                                val prediction = result.answer
                                val golds = listOf(example.answer) + example.answerAliases
                                val reference = EvalUtils.pickBestReferenceForPrediction(prediction, golds)
                                val em = EvalUtils.bestExactMatch(prediction, golds)

                                synchronized(generated) {
                                    generated +=
                                        GeneratedSample(
                                            id = example.id,
                                            question = example.question,
                                            prediction = prediction,
                                            reference = reference,
                                            primaryGold = example.answer,
                                            aliases = example.answerAliases,
                                            retrievedContexts = result.context,
                                            causalPaths = result.causalPaths,
                                            exactMatch = em,
                                        )
                                }

                                val count = processed.incrementAndGet()
                                if (count % 10 == 0) {
                                    println("Generated $count / ${examples.size} samples")
                                }
                            } catch (ex: Exception) {
                                System.err.println("Error processing sample ${example.id}: ${ex.message}")
                            }
                        }
                    }
                }
            jobs.awaitAll()
        }

        if (generated.isEmpty()) {
            System.err.println("No MusiQue samples were successfully generated.")
            kotlin.system.exitProcess(1)
        }

        val dataset =
            EvaluationDataset(
                generated.map { sample ->
                    SingleTurnSample(
                        userInput = sample.question,
                        retrievedContexts = sample.retrievedContexts,
                        response = sample.prediction,
                        reference = sample.reference,
                    )
                },
            )

        val chatModel =
            OpenAiChatModel
                .builder()
                .apiKey(apiKey)
                .modelName(evalModelName)
                .temperature(0.0)
                .build()
        val ragasLlm =
            LangChain4jLlm(
                model = chatModel,
                runConfig = RunConfig(timeoutSeconds = 90),
            )

        val correctnessMetric = AnswerCorrectnessMetric(name = "correctness", weights = listOf(1.0, 0.0))
        val precisionMetric = FactualCorrectnessMetric(name = "precision", mode = FactualCorrectnessMetric.Mode.PRECISION)
        val recallMetric = FactualCorrectnessMetric(name = "recall", mode = FactualCorrectnessMetric.Mode.RECALL)
        val f1Metric = FactualCorrectnessMetric(name = "f1", mode = FactualCorrectnessMetric.Mode.F1)

        val ragasResult =
            evaluate(
                dataset = dataset,
                metrics = listOf(correctnessMetric, precisionMetric, recallMetric, f1Metric),
                llm = ragasLlm,
            )

        val exactMatchTotal = DoubleAdder()
        val correctnessTotal = DoubleAdder()
        val precisionTotal = DoubleAdder()
        val recallTotal = DoubleAdder()
        val f1Total = DoubleAdder()

        generated.forEachIndexed { index, sample ->
            val row = ragasResult.scores.getOrNull(index).orEmpty()
            val correctness = (row["correctness"] as? Number)?.toDouble() ?: Double.NaN
            val precision = (row["precision"] as? Number)?.toDouble() ?: Double.NaN
            val recall = (row["recall"] as? Number)?.toDouble() ?: Double.NaN
            val f1 = (row["f1"] as? Number)?.toDouble() ?: Double.NaN

            exactMatchTotal.add(sample.exactMatch)
            if (!correctness.isNaN()) correctnessTotal.add(correctness)
            if (!precision.isNaN()) precisionTotal.add(precision)
            if (!recall.isNaN()) recallTotal.add(recall)
            if (!f1.isNaN()) f1Total.add(f1)

            val perSample =
                buildString {
                    appendLine("id: ${sample.id}")
                    appendLine("question: ${sample.question}")
                    appendLine("prediction: ${sample.prediction}")
                    appendLine("gold: ${sample.primaryGold}")
                    if (sample.aliases.isNotEmpty()) {
                        appendLine("aliases: ${sample.aliases.joinToString(", ")}")
                    }
                    appendLine("reference_used_for_ragas: ${sample.reference}")
                    appendLine("exact_match: ${sample.exactMatch}")
                    appendLine("correctness: ${"%.4f".format(correctness)}")
                    appendLine("precision: ${"%.4f".format(precision)}")
                    appendLine("recall: ${"%.4f".format(recall)}")
                    appendLine("f1: ${"%.4f".format(f1)}")
                    if (sample.retrievedContexts.isNotEmpty()) {
                        appendLine("retrieved_contexts:")
                        sample.retrievedContexts.forEachIndexed { idx, ctx ->
                            appendLine("  [${idx + 1}] $ctx")
                        }
                    }
                    if (sample.causalPaths.isNotEmpty()) {
                        appendLine("causal_paths:")
                        sample.causalPaths.forEach { path ->
                            appendLine("  - ${path.joinToString(" -> ")}")
                        }
                    }
                }
            Files.writeString(resultsDir.resolve("${sanitizeSampleId(sample.id)}.txt"), perSample)
        }

        val count = generated.size
        val exactMatch = exactMatchTotal.sum() / count
        val correctness = correctnessTotal.sum() / count
        val precision = precisionTotal.sum() / count
        val recall = recallTotal.sum() / count
        val f1 = f1Total.sum() / count

        println("MusiQue + ragas evaluation completed for $count samples")
        println("Generation Model: $modelName")
        println("Evaluation Model: $evalModelName")
        println("Semantic Mode: ${semanticMode.name.lowercase()}")
        println("ExactMatch: ${"%.4f".format(exactMatch)}")
        println("Correctness: ${"%.4f".format(correctness)}")
        println("Precision: ${"%.4f".format(precision)}")
        println("Recall: ${"%.4f".format(recall)}")
        println("F1: ${"%.4f".format(f1)}")
        println("Per-sample outputs written to $resultsDir")
    }

    private fun resolveInputPath(args: Array<String>): Path {
        val defaultPath = "data/musique_ans_v1.0_train-200.jsonl"
        if (args.isEmpty()) return Path.of(defaultPath)

        if (args[0] == "--input") {
            val value = args.getOrNull(1)
            if (value.isNullOrBlank()) {
                System.err.println("Usage: MusiQueRagas [--input <path>] or MusiQueRagas [<path>]")
                kotlin.system.exitProcess(1)
            }
            return Path.of(value)
        }

        return Path.of(args[0])
    }

    private fun parseSemanticMode(raw: String?): HippoRagSemanticMode =
        when (raw?.lowercase()) {
            null, "", "graph" -> {
                HippoRagSemanticMode.GRAPH
            }

            "dpr" -> {
                HippoRagSemanticMode.DPR
            }

            else -> {
                System.err.println("Unknown MUSIQUE_SEMANTIC_MODE '$raw'; defaulting to graph")
                HippoRagSemanticMode.GRAPH
            }
        }

    private fun sampleWorkingDir(
        resultsDir: Path,
        sampleId: String,
    ): Path = resultsDir.resolve("workdirs").resolve(sanitizeSampleId(sampleId))

    private fun writeTempConfig(
        modelName: String,
        embeddingModel: String,
        provider: String,
        apiKey: String?,
        baseUrl: String?,
    ): Path {
        val tempDir = Files.createTempDirectory("hybridrag-musique-ragas")
        tempDir.toFile().deleteOnExit()
        val configPath = tempDir.resolve("musique-ragas-config.json")
        configPath.toFile().deleteOnExit()
        val json =
            JsonObject(
                mapOf(
                    "modelName" to JsonPrimitive(modelName),
                    "embeddingModel" to JsonPrimitive(embeddingModel),
                    "llmProvider" to JsonPrimitive(provider),
                    "llmApiKey" to (apiKey?.let { JsonPrimitive(it) } ?: JsonNull),
                    "llmBaseUrl" to (baseUrl?.let { JsonPrimitive(it) } ?: JsonNull),
                    "embeddingApiKey" to (apiKey?.let { JsonPrimitive(it) } ?: JsonNull),
                    "graphPath" to JsonNull,
                    "indexPath" to JsonNull,
                    "templateStyle" to JsonPrimitive("detailed"),
                ),
            )
        val content = prettyJson.encodeToString(JsonElement.serializer(), json)
        Files.writeString(configPath, content)
        return configPath
    }

    private fun sanitizeSampleId(sampleId: String): String = sampleId.replace(Regex("[^A-Za-z0-9._-]"), "_")
}
