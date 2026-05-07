package causalrag.examples

import causalrag.CausalRAG
import causalrag.QueryParam
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
import ragas.metrics.collections.ContextRelevanceMetric
import ragas.metrics.collections.ResponseGroundednessMetric
import ragas.metrics.defaults.ContextPrecisionMetric
import ragas.metrics.defaults.FaithfulnessMetric
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import java.nio.file.Files
import java.nio.file.Path
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.DoubleAdder

/**
 * Runs OpenAlex-introduction MusiQue evaluation using CausalRAG generation and RAGAS metrics.
 *
 * Metrics:
 * - Faithfulness
 * - Context precision
 * - Response groundedness
 * - Context relevance
 */
object OpenAlexIntroMusiQueCausalRagas {
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
    )

    /**
     * Environment variables:
     * - LLM_PROVIDER (default: openai)
     * - OPENAI_API_KEY (required when provider=openai and for ragas evaluation)
     * - LLM_MODEL (default: gpt-5.4-mini)
     * - MUSIQUE_EVAL_MODEL (default: same as LLM_MODEL)
     * - EMBEDDING_MODEL (default: text-embedding-3-small)
     * - LLM_BASE_URL (optional)
     * - MUSIQUE_LIMIT (optional)
     * - MUSIQUE_PARALLELISM (default: 5)
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
        if (apiKey.isNullOrBlank()) {
            System.err.println("OPENAI_API_KEY is required for ragas evaluation")
            kotlin.system.exitProcess(1)
        }

        val modelName = System.getenv("LLM_MODEL") ?: "gpt-5.4-mini"
        val evalModelName = System.getenv("MUSIQUE_EVAL_MODEL") ?: modelName
        val embeddingModel = System.getenv("EMBEDDING_MODEL") ?: "text-embedding-3-small"
        val baseUrl = System.getenv("LLM_BASE_URL")
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
        val resultsDir = Path.of("eval_results").resolve("openalex_intro_musique_causal_ragas_$timestamp")
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
                                // Each sample has its own passages. CausalRAG mutates in-memory index state,
                                // so keep one instance per sample to avoid cross-sample contamination.
                                val rag =
                                    CausalRAG(
                                        modelName = modelName,
                                        embeddingModel = embeddingModel,
                                        configPath = configPath.toString(),
                                    )

                                rag.upsert(documents)
                                val result = rag.query(example.question, QueryParam(topK = 10))
                                val prediction = result.answer
                                val golds = listOf(example.answer) + example.answerAliases
                                val reference = EvalUtils.pickBestReferenceForPrediction(prediction, golds)

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

        val faithfulnessMetric = FaithfulnessMetric()
        val contextPrecisionMetric = ContextPrecisionMetric()
        val responseGroundednessMetric = ResponseGroundednessMetric()
        val contextRelevanceMetric = ContextRelevanceMetric()

        val ragasResult =
            evaluate(
                dataset = dataset,
                metrics =
                    listOf(
                        faithfulnessMetric,
                        contextPrecisionMetric,
                        responseGroundednessMetric,
                        contextRelevanceMetric,
                    ),
                llm = ragasLlm,
            )

        val faithfulnessTotal = DoubleAdder()
        val contextPrecisionTotal = DoubleAdder()
        val responseGroundednessTotal = DoubleAdder()
        val contextRelevanceTotal = DoubleAdder()

        generated.forEachIndexed { index, sample ->
            val row = ragasResult.scores.getOrNull(index).orEmpty()
            val faithfulness = (row[faithfulnessMetric.name] as? Number)?.toDouble() ?: Double.NaN
            val contextPrecision = (row[contextPrecisionMetric.name] as? Number)?.toDouble() ?: Double.NaN
            val responseGroundedness = (row[responseGroundednessMetric.name] as? Number)?.toDouble() ?: Double.NaN
            val contextRelevance = (row[contextRelevanceMetric.name] as? Number)?.toDouble() ?: Double.NaN

            if (!faithfulness.isNaN()) faithfulnessTotal.add(faithfulness)
            if (!contextPrecision.isNaN()) contextPrecisionTotal.add(contextPrecision)
            if (!responseGroundedness.isNaN()) responseGroundednessTotal.add(responseGroundedness)
            if (!contextRelevance.isNaN()) contextRelevanceTotal.add(contextRelevance)

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
                    appendLine("faithfulness: ${"%.4f".format(faithfulness)}")
                    appendLine("context_precision: ${"%.4f".format(contextPrecision)}")
                    appendLine("response_groundedness: ${"%.4f".format(responseGroundedness)}")
                    appendLine("context_relevance: ${"%.4f".format(contextRelevance)}")
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
        val faithfulnessAvg = faithfulnessTotal.sum() / count
        val contextPrecisionAvg = contextPrecisionTotal.sum() / count
        val responseGroundednessAvg = responseGroundednessTotal.sum() / count
        val contextRelevanceAvg = contextRelevanceTotal.sum() / count

        println("OpenAlex intro MusiQue + ragas evaluation completed for $count samples")
        println("Input: $dataPath")
        println("Generation Model: $modelName")
        println("Evaluation Model: $evalModelName")
        println("Pipeline: CausalRAG")
        println("Answer Faithfulness: ${"%.4f".format(faithfulnessAvg)}")
        println("Context Precision: ${"%.4f".format(contextPrecisionAvg)}")
        println("Response Groundedness: ${"%.4f".format(responseGroundednessAvg)}")
        println("Context Relevance: ${"%.4f".format(contextRelevanceAvg)}")
        println("Per-sample outputs written to $resultsDir")
    }

    private fun resolveInputPath(args: Array<String>): Path {
        val defaultPath = "data/openalex_eval_dataset_intro_musique.jsonl"
        if (args.isEmpty()) return Path.of(defaultPath)

        if (args[0] == "--input") {
            val value = args.getOrNull(1)
            if (value.isNullOrBlank()) {
                System.err.println("Usage: OpenAlexIntroMusiQueCausalRagas [--input <path>] or OpenAlexIntroMusiQueCausalRagas [<path>]")
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
        val tempDir = Files.createTempDirectory("causalrag-openalex-intro-ragas")
        tempDir.toFile().deleteOnExit()
        val configPath = tempDir.resolve("openalex-intro-causal-ragas-config.json")
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
