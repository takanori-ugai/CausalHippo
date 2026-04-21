package lightrag.examples

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiChatModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.llm.LLMFactory
import lightrag.operate.removeThinkTags
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager
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
import kotlin.math.min

@Serializable
data class MusiqueParagraph(
    val idx: Int,
    val title: String,
    @SerialName("paragraph_text")
    val paragraphText: String,
    @SerialName("is_supporting")
    val isSupporting: Boolean,
)

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
 * Runs MusiQue evaluation using LightRAG for generation and ragas for scoring.
 *
 * Environment variables:
 * - LLM_PROVIDER (openai|ollama, default: openai)
 * - OPENAI_API_KEY (required for provider=openai, and always required for ragas LLM metrics)
 * - LLM_MODEL (default: gpt-5.4-mini)
 * - MUSIQUE_EVAL_MODEL (default: same as LLM_MODEL)
 * - EMBEDDING_MODEL (default: text-embedding-3-small)
 * - LLM_BASE_URL (optional; used for generation and evaluation model)
 * - MUSIQUE_QUERY_MODE (naive|local|global|hybrid, default: hybrid)
 * - MUSIQUE_LIMIT (optional)
 * - MUSIQUE_PARALLELISM (default: 2)
 * - MUSIQUE_TOP_K (default: 10)
 * - MUSIQUE_CHUNK_TOP_K (default: 10)
 */
object MusiQueLightRagRagas {
    private val inputJson = Json { ignoreUnknownKeys = true }

    private data class GeneratedSample(
        val id: String,
        val question: String,
        val prediction: String,
        val reference: String,
        val primaryGold: String,
        val aliases: List<String>,
        val retrievedContexts: List<String>,
        val exactMatch: Double,
    )

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
        val queryMode = parseQueryMode(System.getenv("MUSIQUE_QUERY_MODE"))
        val limit = (System.getenv("MUSIQUE_LIMIT") ?: "").toIntOrNull()
        val parallelism = (System.getenv("MUSIQUE_PARALLELISM") ?: "2").toIntOrNull() ?: 2
        val topK = (System.getenv("MUSIQUE_TOP_K") ?: "10").toIntOrNull() ?: 10
        val chunkTopK = (System.getenv("MUSIQUE_CHUNK_TOP_K") ?: "10").toIntOrNull() ?: 10
        require(parallelism > 0) { "MUSIQUE_PARALLELISM must be positive." }
        require(topK > 0) { "MUSIQUE_TOP_K must be positive." }
        require(chunkTopK > 0) { "MUSIQUE_CHUNK_TOP_K must be positive." }

        val lines =
            Files
                .readAllLines(dataPath)
                .map { it.trim() }
                .filter { it.isNotBlank() }
        if (lines.isEmpty()) {
            System.err.println("No MusiQue samples found in $dataPath")
            kotlin.system.exitProcess(1)
        }

        val examples =
            lines
                .map { inputJson.decodeFromString(MusiqueExample.serializer(), it) }
                .filter { it.answerable }
                .let { list -> if (limit != null) list.take(limit) else list }
        if (examples.isEmpty()) {
            System.err.println("No answerable MusiQue samples were processed.")
            kotlin.system.exitProcess(1)
        }

        val chatModel =
            LLMFactory.createChatModel(
                binding = provider,
                modelName = modelName,
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
                modelName = embeddingModel,
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

        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
        val resultsDir = Path.of("eval_results").resolve("musique_lightrag_ragas_$timestamp")
        Files.createDirectories(resultsDir)

        val generated = mutableListOf<GeneratedSample>()
        val processed = AtomicInteger(0)
        val semaphore = Semaphore(parallelism)

        runBlocking {
            val jobs =
                examples.map { example ->
                    async(Dispatchers.IO) {
                        semaphore.withPermit {
                            try {
                                val sampleDir = sampleWorkingDir(resultsDir, example.id)
                                sampleDir.createDirectories()
                                val rag =
                                    createLightRagForSample(
                                        workingDir = sampleDir.toString(),
                                        chatModel = chatModel,
                                        embeddingModel = retrievalEmbeddingModel,
                                        tokenizer = tokenizer,
                                        decoder = decoder,
                                    )
                                rag.storageManager.initialize()
                                rag.insert(example.paragraphs.map { it.paragraphText })
                                rag.rebuildDerivedStorageIfEmpty()

                                val queryResult =
                                    rag.query(
                                        example.question,
                                        QueryParam(
                                            mode = queryMode,
                                            includeReferences = true,
                                            topK = topK,
                                            chunkTopK = chunkTopK,
                                        ),
                                    )
                                val prediction = removeThinkTags(queryResult?.content.orEmpty()).trim()
                                val retrievedContexts = extractChunkContents(queryResult?.rawData).take(topK)
                                val golds = listOf(example.answer) + example.answerAliases
                                val reference = pickBestReferenceForPrediction(prediction, golds)
                                val em = bestExactMatch(prediction, golds)

                                synchronized(generated) {
                                    generated +=
                                        GeneratedSample(
                                            id = example.id,
                                            question = example.question,
                                            prediction = prediction,
                                            reference = reference,
                                            primaryGold = example.answer,
                                            aliases = example.answerAliases,
                                            retrievedContexts = retrievedContexts,
                                            exactMatch = em,
                                        )
                                }

                                val count = processed.incrementAndGet()
                                if (count % 10 == 0 || count == examples.size) {
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

        val evalChatModelBuilder =
            OpenAiChatModel
                .builder()
                .apiKey(apiKey)
                .modelName(evalModelName)
                .temperature(0.0)
        if (!baseUrl.isNullOrBlank()) {
            evalChatModelBuilder.baseUrl(baseUrl)
        }
        val ragasLlm =
            LangChain4jLlm(
                model = evalChatModelBuilder.build(),
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
                }
            Files.writeString(resultsDir.resolve("${sanitizeSampleId(sample.id)}.txt"), perSample)
        }

        val count = generated.size.toDouble()
        val exactMatch = exactMatchTotal.sum() / count
        val correctness = correctnessTotal.sum() / count
        val precision = precisionTotal.sum() / count
        val recall = recallTotal.sum() / count
        val f1 = f1Total.sum() / count

        println("MusiQue + ragas evaluation completed for ${generated.size} samples")
        println("Generation Provider: $provider")
        println("Generation Model: $modelName")
        println("Evaluation Model: $evalModelName")
        println("Pipeline: LightRAG")
        println("Query Mode: $queryMode")
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
                System.err.println("Usage: MusiQueLightRagRagas [--input <path>] or MusiQueLightRagRagas [<path>]")
                kotlin.system.exitProcess(1)
            }
            return Path.of(value)
        }

        return Path.of(args[0])
    }

    private fun parseQueryMode(raw: String?): String {
        val mode = raw?.lowercase()?.trim().orEmpty()
        return when (mode) {
            "", "naive", "local", "global", "hybrid" -> {
                if (mode.isEmpty()) "hybrid" else mode
            }

            else -> {
                System.err.println("Unknown MUSIQUE_QUERY_MODE '$raw'; defaulting to hybrid")
                "hybrid"
            }
        }
    }

    private fun createLightRagForSample(
        workingDir: String,
        chatModel: ChatModel,
        embeddingModel: EmbeddingModel,
        tokenizer: (String) -> List<Int>,
        decoder: (List<Int>) -> String,
    ): LightRAG {
        val globalConfig =
            mapOf(
                "llm_model_func" to chatModel,
                "embedding_func" to embeddingModel,
                "chunk_token_size" to 1200,
                "chunk_overlap_token_size" to 100,
                "entity_types" to listOf("Person", "Organization", "Location", "Event", "Concept"),
                "language" to "English",
                "working_dir" to workingDir,
                "enable_llm_cache" to false,
            )

        val storageManager =
            StorageManager(
                workingDir = workingDir,
                embeddingModel = embeddingModel,
                graphStorageName = "InMemoryGraphStorage",
                vectorStorageName = "InMemoryVectorStorage",
                addonConfig = AddonConfig(),
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

    private fun extractChunkContents(rawData: Map<String, Any?>?): List<String> {
        if (rawData == null) return emptyList()
        val fromRoot = extractChunkContentsFromMap(rawData)
        if (fromRoot.isNotEmpty()) return fromRoot

        val dataMap = rawData["data"] as? Map<*, *> ?: return emptyList()
        return extractChunkContentsFromMap(dataMap)
    }

    private fun extractChunkContentsFromMap(map: Map<*, *>): List<String> {
        val chunks = map["chunks"] as? List<*> ?: return emptyList()
        return chunks.mapNotNull { chunk ->
            val chunkMap = chunk as? Map<*, *> ?: return@mapNotNull null
            val value = chunkMap["content"] ?: chunkMap["text"] ?: return@mapNotNull null
            value.toString().trim().takeIf { it.isNotBlank() }
        }
    }

    private fun sampleWorkingDir(
        resultsDir: Path,
        sampleId: String,
    ): Path = resultsDir.resolve("workdirs").resolve(sanitizeSampleId(sampleId))

    private fun sanitizeSampleId(sampleId: String): String = sampleId.replace(Regex("[^A-Za-z0-9._-]"), "_")

    private fun pickBestReferenceForPrediction(
        prediction: String,
        golds: List<String>,
    ): String =
        golds
            .maxByOrNull { candidate ->
                if (normalize(prediction) == normalize(candidate)) {
                    10_000
                } else {
                    tokenOverlapScore(prediction, candidate)
                }
            } ?: ""

    private fun tokenOverlapScore(
        prediction: String,
        gold: String,
    ): Int {
        val predTokens = tokenize(normalize(prediction))
        val goldTokens = tokenize(normalize(gold))
        if (predTokens.isEmpty() || goldTokens.isEmpty()) return 0
        val predCounts = predTokens.groupingBy { it }.eachCount()
        val goldCounts = goldTokens.groupingBy { it }.eachCount()
        var overlap = 0
        for ((token, pCount) in predCounts) {
            val gCount = goldCounts[token] ?: 0
            overlap += min(pCount, gCount)
        }
        return overlap
    }

    private fun bestExactMatch(
        prediction: String,
        golds: List<String>,
    ): Double = golds.maxOfOrNull { if (normalize(prediction) == normalize(it)) 1.0 else 0.0 } ?: 0.0

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
