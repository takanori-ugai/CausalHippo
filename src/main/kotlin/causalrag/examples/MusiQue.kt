package causalrag.examples

import causalrag.retriever.HippoRagSemanticMode
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.nio.file.Files
import java.nio.file.Path
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.DoubleAdder
import kotlin.io.path.createDirectories
import kotlin.math.max

/**
 * Paragraph entry from the MusiQue dataset.
 *
 * @property idx Paragraph index within the example.
 * @property title Source title.
 * @property paragraphText Paragraph content.
 * @property isSupporting Whether the paragraph is annotated as supporting evidence.
 */
@Serializable
data class MusiqueParagraph(
    val idx: Int,
    val title: String,
    @SerialName("paragraph_text")
    val paragraphText: String,
    @SerialName("is_supporting")
    val isSupporting: Boolean,
)

/**
 * Single MusiQue question-answer example.
 *
 * @property id Example identifier.
 * @property paragraphs Paragraphs associated with the example.
 * @property question Question text.
 * @property answer Gold answer text.
 * @property answerAliases Alternate gold answers.
 * @property answerable Whether the example is answerable.
 */
@Serializable
data class MusiqueExample(
    val id: String,
    val paragraphs: List<MusiqueParagraph>,
    val question: String,
    val answer: String,
    @SerialName("answer_aliases")
    val answerAliases: List<String> = emptyList(),
    val answerable: Boolean = true,
)

/**
 * Runs batch evaluation of the HybridRAG pipeline on the MusiQue dataset.
 */
object MusiQue {
    private val inputJson = Json { ignoreUnknownKeys = true }
    private val prettyJson = Json { prettyPrint = true }

    private data class QaMetrics(
        val precision: Double,
        val recall: Double,
        val f1: Double,
    )

    /**
     * Entry point for MusiQue evaluation using HybridRAG.
     *
     * Configuration is read from environment variables:
     * - `LLM_PROVIDER`: LLM provider name (default: `"openai"`)
     * - `OPENAI_API_KEY`: API key for the OpenAI provider
     * - `LLM_MODEL`: Model name (default: `"gpt-4o-mini"`)
     * - `EMBEDDING_MODEL`: Embedding model (default: `"text-embedding-3-small"`)
     * - `LLM_BASE_URL`: Optional base URL override
     * - `MUSIQUE_SEMANTIC_MODE`: HippoRAG semantic mode (`"graph"` or `"dpr"`, default: `"graph"`)
     * - `MUSIQUE_LIMIT`: Optional limit on samples to process
     * - `MUSIQUE_PARALLELISM`: Concurrent processing limit (default: `5`)
     *
     * @param args Command-line arguments. Optional forms:
     * - positional: `<input-path>`
     * - named: `--input <input-path>`
     */
    @Suppress("TooGenericExceptionCaught")
    @Deprecated(
        message =
            "Legacy evaluator entrypoint. Prefer shared.eval.MultiConditionExperimentKt " +
                "with --use-unified-api=true for unified adapter-based runs.",
        level = DeprecationLevel.WARNING,
    )
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

        val modelName = System.getenv("LLM_MODEL") ?: "gpt-5.4-mini"
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

        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
        val resultsDir = Path.of("eval_results").resolve("musique_$timestamp")
        Files.createDirectories(resultsDir)
        val configPath = writeTempConfig(modelName, embeddingModel, provider, apiKey, baseUrl)

        if (examples.isEmpty()) {
            System.err.println("No answerable MusiQue samples were processed.")
            kotlin.system.exitProcess(1)
        }

        val exactMatchTotal = DoubleAdder()
        val precisionTotal = DoubleAdder()
        val recallTotal = DoubleAdder()
        val f1Total = DoubleAdder()
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
                                println("Answer: $prediction")
                                println("Gold: $golds")
                                val em = bestExactMatch(prediction, golds)
                                val metrics = bestQaMetrics(prediction, golds)
                                exactMatchTotal.add(em)
                                precisionTotal.add(metrics.precision)
                                recallTotal.add(metrics.recall)
                                f1Total.add(metrics.f1)

                                val count = processed.incrementAndGet()
                                if ((count % 10) == 0) {
                                    println("Processed $count / ${examples.size} samples")
                                }

                                val perSample =
                                    buildString {
                                        appendLine("id: ${example.id}")
                                        appendLine("question: ${example.question}")
                                        appendLine("prediction: $prediction")
                                        appendLine("gold: ${example.answer}")
                                        if (example.answerAliases.isNotEmpty()) {
                                            appendLine("aliases: ${example.answerAliases.joinToString(", ")}")
                                        }
                                        appendLine("semantic_mode: ${semanticMode.name.lowercase()}")
                                        appendLine("exact_match: $em")
                                        appendLine("precision: ${"%.4f".format(metrics.precision)}")
                                        appendLine("recall: ${"%.4f".format(metrics.recall)}")
                                        appendLine("f1: ${"%.4f".format(metrics.f1)}")
                                        if (result.causalPaths.isNotEmpty()) {
                                            appendLine("causal_paths:")
                                            result.causalPaths.forEach { path ->
                                                appendLine("  - ${path.joinToString(" -> ")}")
                                            }
                                        }
                                    }
                                Files.writeString(resultsDir.resolve("${sanitizeSampleId(example.id)}.txt"), perSample)
                            } catch (ex: Exception) {
                                if (ex is CancellationException) throw ex
                                System.err.println("Error processing sample ${example.id}: ${ex.message}")
                            }
                        }
                    }
                }
            jobs.awaitAll()
        }

        val processedCount = processed.get()
        if (processedCount == 0) {
            System.err.println("No answerable MusiQue samples were processed.")
            kotlin.system.exitProcess(1)
        }

        val exactMatch = exactMatchTotal.sum() / processedCount
        val precision = precisionTotal.sum() / processedCount
        val recall = recallTotal.sum() / processedCount
        val f1 = f1Total.sum() / processedCount
        println("MusiQue evaluation completed for $processedCount samples")
        println("Semantic Mode: ${semanticMode.name.lowercase()}")
        println("Exact Match: ${"%.4f".format(exactMatch)}")
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
                System.err.println("Usage: MusiQue [--input <path>] or MusiQue [<path>]")
                kotlin.system.exitProcess(1)
            }
            return Path.of(value)
        }

        return Path.of(args[0])
    }

    private fun writeTempConfig(
        modelName: String,
        embeddingModel: String,
        provider: String,
        apiKey: String?,
        baseUrl: String?,
    ): Path {
        val tempDir = Files.createTempDirectory("hybridrag-musique")
        tempDir.toFile().deleteOnExit()
        val configPath = tempDir.resolve("musique-config.json")
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

    private fun sampleWorkingDir(
        resultsDir: Path,
        sampleId: String,
    ): Path = resultsDir.resolve("workdirs").resolve(sanitizeSampleId(sampleId))

    private fun sanitizeSampleId(sampleId: String): String = sampleId.replace(Regex("[^A-Za-z0-9._-]"), "_")

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

    private fun bestExactMatch(
        prediction: String,
        golds: List<String>,
    ): Double = golds.maxOfOrNull { if (normalize(prediction) == normalize(it)) 1.0 else 0.0 } ?: 0.0

    private fun bestQaMetrics(
        prediction: String,
        golds: List<String>,
    ): QaMetrics = golds.map { qaMetrics(prediction, it) }.maxByOrNull { it.f1 } ?: QaMetrics(0.0, 0.0, 0.0)

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
        return QaMetrics(precision, recall, f1)
    }

    private fun normalize(text: String): String {
        val lowered = text.lowercase()
        val noPunc = lowered.replace(Regex("[^a-z0-9\\s]"), " ")
        val noArticles = noPunc.replace(Regex("\\b(a|an|the)\\b"), " ")
        return noArticles.replace(Regex("\\s+"), " ").trim()
    }

    private fun tokenize(text: String): List<String> =
        if (text.isBlank()) {
            emptyList()
        } else {
            text.split(' ')
        }
}
