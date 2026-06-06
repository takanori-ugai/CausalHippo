package com.youtu.graphrag.cli

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.youtu.graphrag.server.api.QuestionAnsweringService
import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.config.ConfigProvider
import com.youtu.graphrag.shared.constructor.KTBuilder
import com.youtu.graphrag.shared.llm.LlmClient
import com.youtu.graphrag.shared.llm.LlmClientFactory
import com.youtu.graphrag.shared.retriever.IrcotPromptSource
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.runBlocking
import picocli.CommandLine
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.nio.file.Path
import java.util.Locale
import kotlin.io.path.createDirectories
import kotlin.io.path.deleteIfExists
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.io.path.listDirectoryEntries
import kotlin.system.exitProcess
import kotlin.system.measureTimeMillis

data class QaItem(
    val question: String,
    val referenceAnswer: String? = null,
)

private data class EvalOutcome(
    val result: String?,
    val method: String,
)

@Command(
    name = "youtu-graphrag",
    description = ["Youtu-GraphRAG Kotlin CLI"],
    mixinStandardHelpOptions = true,
)
class MainCommand(
    private val llmClient: LlmClient = LlmClientFactory.fromEnvironment(),
) : Runnable {
    private val logger = KotlinLogging.logger {}
    private val objectMapper = ObjectMapper().registerKotlinModule()

    @Option(
        names = ["--config"],
        description = ["Path to configuration file"],
        defaultValue = "config/base_config.json",
    )
    lateinit var configPath: String

    @Option(
        names = ["--datasets"],
        arity = "1..*",
        description = ["List of datasets to process"],
        defaultValue = "demo",
    )
    var datasets: List<String> = listOf("demo")

    @Option(
        names = ["--override"],
        description = ["JSON string with configuration overrides"],
    )
    var overrideJson: String? = null

    override fun run() {
        val config =
            try {
                ConfigProvider.reloadConfig(configPath)
            } catch (error: Exception) {
                throw CommandLine.ParameterException(
                    CommandLine(this),
                    "Failed to load configuration: ${error.message}",
                    error,
                )
            }

        applyOverridesIfPresent(config)
        setupEnvironment(config)

        runConstructorWorkflow(config)
        runRetrievalWorkflow(config)
    }

    private fun applyOverridesIfPresent(config: ConfigManager) {
        val overrideInput = overrideJson ?: return

        try {
            val overrides: Map<String, Any?> =
                objectMapper.readValue(overrideInput, object : TypeReference<Map<String, Any?>>() {})
            config.overrideConfig(overrides)
            logger.info { "Applied configuration overrides" }
        } catch (error: Exception) {
            throw CommandLine.ParameterException(
                CommandLine(this),
                "Invalid JSON in --override argument: ${error.message}",
                error,
            )
        }
    }

    private fun setupEnvironment(config: ConfigManager) {
        config.createOutputDirectories()

        logger.info { "Youtu-GraphRAG Kotlin initialized" }
        logger.info { "Mode: ${config.triggers.mode}" }
        logger.info { "Constructor enabled: ${config.triggers.constructorTrigger}" }
        logger.info { "Retriever enabled: ${config.triggers.retrieveTrigger}" }
        logger.info { "Datasets: ${datasets.joinToString(", ")}" }
    }

    private fun runConstructorWorkflow(config: ConfigManager) {
        if (!config.triggers.constructorTrigger) {
            logger.info { "Constructor workflow disabled by triggers.constructor_trigger=false" }
            return
        }

        logger.info { "Starting knowledge graph construction..." }
        datasets.forEach { datasetName ->
            try {
                val datasetConfig = config.getDatasetConfig(datasetName)
                logger.info { "Building knowledge graph for dataset: $datasetName" }
                clearCacheFiles(datasetName, config)

                val builder =
                    KTBuilder(
                        datasetName = datasetName,
                        schemaPath = datasetConfig.schemaPath,
                        mode = config.construction.mode,
                        config = config,
                    )
                builder.buildKnowledgeGraph(datasetConfig.corpusPath)
                logger.info { "Successfully built knowledge graph for dataset: $datasetName" }
            } catch (error: Exception) {
                logger.error(error) { "Failed to build knowledge graph for dataset: $datasetName" }
            }
        }
    }

    private fun runRetrievalWorkflow(config: ConfigManager) {
        if (!config.triggers.retrieveTrigger) {
            logger.info { "Retrieval workflow disabled by triggers.retrieve_trigger=false" }
            return
        }

        logger.info { "Starting retrieval and QA..." }
        datasets.forEach { datasetName ->
            try {
                val datasetConfig = config.getDatasetConfig(datasetName)
                val qaItems = loadQaItems(datasetConfig.qaPath)
                if (qaItems.isEmpty()) {
                    logger.warn { "No questions found for dataset '$datasetName' at ${datasetConfig.qaPath}" }
                    return@forEach
                }

                val qaService =
                    QuestionAnsweringService(
                        config = config,
                        llmClient = llmClient,
                        ircotPromptSource = IrcotPromptSource.MAIN,
                    )
                var evaluatedCount = 0
                var correctCount = 0
                var totalAnswerTimeSeconds = 0.0

                val qaResults =
                    runBlocking {
                        val results = mutableListOf<Map<String, Any?>>()
                        qaItems.forEachIndexed { index, qaItem ->
                            try {
                                var responseAnswer = ""
                                lateinit var answered: com.youtu.graphrag.server.api.contracts.QuestionResponse
                                val elapsedMs =
                                    measureTimeMillis {
                                        answered =
                                            qaService.answerQuestion(
                                                datasetName = datasetName,
                                                question = qaItem.question,
                                            )
                                        responseAnswer = answered.answer
                                    }
                                val elapsedSeconds = elapsedMs / 1000.0
                                totalAnswerTimeSeconds += elapsedSeconds
                                val evalOutcome =
                                    computeEvalResult(
                                        question = qaItem.question,
                                        referenceAnswer = qaItem.referenceAnswer,
                                        generatedAnswer = responseAnswer,
                                    )
                                val evalResult = evalOutcome.result
                                if (evalResult != null) {
                                    evaluatedCount += 1
                                    if (evalResult == "1") {
                                        correctCount += 1
                                    }
                                }

                                val result =
                                    mapOf(
                                        "question" to qaItem.question,
                                        "reference_answer" to qaItem.referenceAnswer,
                                        "answer" to answered.answer,
                                        "eval_result" to evalResult,
                                        "eval_method" to evalOutcome.method,
                                        "answer_time_seconds" to elapsedSeconds,
                                        "sub_questions" to answered.subQuestions,
                                        "retrieved_triples" to answered.retrievedTriples,
                                        "retrieved_chunks" to answered.retrievedChunks,
                                        "reasoning_steps" to
                                            answered.reasoningSteps.map { step ->
                                                mapOf(
                                                    "type" to step.type,
                                                    "question" to step.question,
                                                    "triples" to step.triples,
                                                    "triples_count" to step.triplesCount,
                                                    "chunk_contents" to step.chunkContents,
                                                    "chunks_count" to step.chunksCount,
                                                    "processing_time" to step.processingTime,
                                                    "thought" to step.thought,
                                                )
                                            },
                                        "visualization_data" to objectMapper.readTree(answered.visualizationData.toString()),
                                        "triples_count" to answered.retrievedTriples.size,
                                        "chunks_count" to answered.retrievedChunks.size,
                                    )
                                val initialAnswerThought =
                                    answered.reasoningSteps
                                        .firstOrNull { step -> step.type == "initial_answer" }
                                        ?.thought
                                        .orEmpty()
                                val ircotThoughts =
                                    answered.reasoningSteps
                                        .filter { step -> step.type == "ircot_step" }
                                        .mapNotNull { step -> step.thought?.trim()?.takeIf { value -> value.isNotBlank() } }
                                val thoughtProcess =
                                    buildList {
                                        initialAnswerThought
                                            .takeIf { value -> value.isNotBlank() }
                                            ?.let { value -> add("Initial analysis (noagent mode): $value") }
                                        addAll(ircotThoughts)
                                    }.joinToString(" | ")

                                logger.info { "========== Original Question: ${qaItem.question} ==========" }
                                if (config.triggers.mode == "agent" && initialAnswerThought.isNotBlank()) {
                                    logger.info { "Noagent Initial Answer: $initialAnswerThought" }
                                    logger.info { "IRCoT Steps: ${1 + ircotThoughts.size}" }
                                    logger.info { "Final Triples: ${answered.retrievedTriples.size}" }
                                    logger.info { "Final Chunks: ${answered.retrievedChunks.size}" }
                                }
                                logger.info { "Gold Answer: ${qaItem.referenceAnswer.orEmpty()}" }
                                logger.info { "Generated Answer: ${answered.answer}" }
                                if (config.triggers.mode == "agent" && thoughtProcess.isNotBlank()) {
                                    logger.info { "Thought Process: $thoughtProcess" }
                                }
                                if (evalResult != null) {
                                    val modeTag = if (config.triggers.mode == "agent") "Agent" else "No agent"
                                    logger.info { "$modeTag mode eval result: $evalResult (${evalOutcome.method})" }
                                }
                                logger.info {
                                    "Answered question ${index + 1}/${qaItems.size} for dataset '$datasetName'"
                                }
                                results.add(result)
                            } catch (error: Exception) {
                                logger.error(
                                    error,
                                ) {
                                    "Failed to answer question ${index + 1}/${qaItems.size} for dataset '$datasetName': ${qaItem.question}"
                                }
                            }
                        }
                        results
                    }

                val answeredCount = qaResults.size
                if (evaluatedCount > 0) {
                    val accuracy = correctCount.toDouble() / evaluatedCount.toDouble() * 100.0
                    logger.info {
                        "Evaluation summary for '$datasetName': $correctCount/$evaluatedCount correct (${String.format(
                            Locale.ROOT,
                            "%.2f",
                            accuracy,
                        )}%)"
                    }
                }
                if (answeredCount > 0) {
                    val averageSeconds = totalAnswerTimeSeconds / answeredCount.toDouble()
                    logger.info {
                        "Average time taken for '$datasetName': ${String.format(Locale.ROOT, "%.3f", averageSeconds)} seconds"
                    }
                }

                val sanitizedName = sanitizeDatasetName(datasetName)
                val logsDir = resolvePath(config.output.logsDir).also { it.createDirectories() }
                val outputFile = logsDir.resolve("${sanitizedName}_qa_results.json")
                objectMapper.writerWithDefaultPrettyPrinter().writeValue(outputFile.toFile(), qaResults)
                logger.info { "Saved QA results for '$datasetName' to $outputFile" }
                val summaryFile = logsDir.resolve("${sanitizedName}_qa_summary.json")
                objectMapper.writerWithDefaultPrettyPrinter().writeValue(
                    summaryFile.toFile(),
                    mapOf(
                        "dataset" to datasetName,
                        "mode" to config.triggers.mode,
                        "total_questions" to qaItems.size,
                        "answered_questions" to answeredCount,
                        "evaluated_questions" to evaluatedCount,
                        "correct_answers" to correctCount,
                        "accuracy" to
                            if (evaluatedCount > 0) {
                                correctCount.toDouble() / evaluatedCount.toDouble()
                            } else {
                                null
                            },
                        "total_answer_time_seconds" to totalAnswerTimeSeconds,
                        "average_answer_time_seconds" to
                            if (answeredCount > 0) {
                                totalAnswerTimeSeconds / answeredCount.toDouble()
                            } else {
                                null
                            },
                    ),
                )
                logger.info { "Saved QA summary for '$datasetName' to $summaryFile" }
            } catch (error: Exception) {
                logger.error(error) { "Retrieval workflow failed for dataset '$datasetName'" }
            }
        }
    }

    private fun clearCacheFiles(
        datasetName: String,
        config: ConfigManager,
    ) {
        val sanitizedName = sanitizeDatasetName(datasetName)
        deleteRecursivelyIfExists(resolvePath(config.retrieval.cacheDir).resolve(sanitizedName))
        deleteRecursivelyIfExists(resolvePath(config.output.chunksDir).resolve("$sanitizedName.txt"))
        deleteRecursivelyIfExists(resolvePath(config.output.graphsDir).resolve("${sanitizedName}_new.json"))

        deleteDatasetPrefixEntries(resolvePath(config.output.logsDir), datasetName)
        deleteDatasetPrefixEntries(resolvePath(config.output.chunksDir), datasetName)
        deleteDatasetPrefixEntries(resolvePath(config.output.graphsDir), datasetName)
    }

    private fun deleteDatasetPrefixEntries(
        directory: Path,
        datasetName: String,
    ) {
        if (!directory.exists() || !directory.isDirectory()) {
            return
        }

        val prefix = "${sanitizeDatasetName(datasetName)}_"
        directory.listDirectoryEntries().forEach { entry ->
            if (entry.fileName.toString().startsWith(prefix)) {
                deleteRecursivelyIfExists(entry)
            }
        }
    }

    private fun sanitizeDatasetName(name: String): String =
        name
            .replace(Regex("[^a-zA-Z0-9._-]"), "_")
            .replace(Regex("\\.+"), ".")
            .trim { it == '.' || it == '_' || it == '-' }
            .ifBlank { "default_dataset" }

    private fun deleteRecursivelyIfExists(path: Path) {
        if (!path.exists()) {
            return
        }

        if (path.isDirectory()) {
            path.listDirectoryEntries().forEach { child ->
                deleteRecursivelyIfExists(child)
            }
        }

        path.deleteIfExists()
    }

    private fun resolvePath(path: String): Path {
        val candidate = Path.of(path)
        return if (candidate.isAbsolute) candidate else Path.of(".").resolve(candidate)
    }

    private fun computeEvalResult(
        question: String,
        referenceAnswer: String?,
        generatedAnswer: String,
    ): EvalOutcome {
        val reference = referenceAnswer?.trim().orEmpty()
        if (reference.isBlank()) {
            return EvalOutcome(result = null, method = "skipped_no_reference")
        }

        val llmEval = evaluateWithLlm(question = question, referenceAnswer = reference, generatedAnswer = generatedAnswer)
        if (llmEval != null) {
            return EvalOutcome(result = llmEval, method = "llm")
        }

        val normalizedReference = normalizeForEval(reference)
        val normalizedGenerated = normalizeForEval(generatedAnswer)
        if (normalizedReference.isBlank() || normalizedGenerated.isBlank()) {
            return EvalOutcome(result = "0", method = "heuristic")
        }

        val result =
            if (
                normalizedReference == normalizedGenerated ||
                normalizedGenerated.contains(normalizedReference) ||
                normalizedReference.contains(normalizedGenerated)
            ) {
                "1"
            } else {
                "0"
            }
        return EvalOutcome(result = result, method = "heuristic")
    }

    private fun evaluateWithLlm(
        question: String,
        referenceAnswer: String,
        generatedAnswer: String,
    ): String? {
        val prompt =
            """
            You are an expert evaluator. Your task is to determine if the predicted answer is correct based on the question and gold answer.
            The criteria should be reasonable, not too strict or too lenient.

            Question: $question
            Gold Answer: $referenceAnswer
            Predicted Answer: $generatedAnswer

            Return only "1" (correct) or "0" (incorrect):
            """.trimIndent()

        val llmOutput =
            runCatching { llmClient.complete(prompt).trim() }.getOrElse { error ->
                logger.warn(error) { "LLM evaluator call failed; falling back to heuristic evaluation." }
                return null
            }
        if (llmOutput.isBlank()) {
            return null
        }

        val exactToken = Regex("\\b[01]\\b").find(llmOutput)?.value
        if (exactToken != null) {
            return exactToken
        }
        return if ("1" in llmOutput && "0" !in llmOutput) {
            "1"
        } else if ("0" in llmOutput && "1" !in llmOutput) {
            "0"
        } else {
            null
        }
    }

    private fun normalizeForEval(text: String): String =
        text
            .lowercase()
            .replace(Regex("[^a-z0-9\\s]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
}

internal fun loadQaItems(
    qaPath: String,
    objectMapper: ObjectMapper = ObjectMapper().registerKotlinModule(),
): List<QaItem> {
    val qaFile = Path.of(qaPath)
    if (!qaFile.exists()) {
        return emptyList()
    }

    val root = runCatching { objectMapper.readTree(qaFile.toFile()) }.getOrNull() ?: return emptyList()
    val collected = mutableListOf<QaItem>()
    collectQaItems(root, collected)
    return collected
        .map { item -> item.copy(question = item.question.trim(), referenceAnswer = item.referenceAnswer?.trim()) }
        .filter { item -> item.question.isNotBlank() }
        .distinctBy { item -> item.question }
}

private fun collectQaItems(
    node: JsonNode,
    destination: MutableList<QaItem>,
) {
    when {
        node.isArray -> {
            node.forEach { element -> collectQaItems(element, destination) }
        }

        node.isObject -> {
            val question = extractText(node, QUESTION_KEYS)
            if (!question.isNullOrBlank()) {
                destination.add(
                    QaItem(
                        question = question,
                        referenceAnswer = extractText(node, ANSWER_KEYS),
                    ),
                )
            }

            COMMON_CONTAINER_KEYS.forEach { key ->
                val child = node.get(key)
                if (child != null) {
                    collectQaItems(child, destination)
                }
            }
        }
    }
}

private fun extractText(
    node: JsonNode,
    keys: Set<String>,
): String? {
    keys.forEach { key ->
        val value = node.get(key) ?: return@forEach
        when {
            value.isTextual -> {
                return value.asText()
            }

            value.isArray -> {
                val joined = value.filter { it.isTextual }.joinToString(" | ") { it.asText() }
                if (joined.isNotBlank()) {
                    return joined
                }
            }
        }
    }
    return null
}

private val QUESTION_KEYS = setOf("question", "query", "q", "question_text", "input")
private val ANSWER_KEYS = setOf("answer", "gold_answer", "reference_answer", "output")
private val COMMON_CONTAINER_KEYS = setOf("qa_pairs", "questions", "items", "data", "examples")

fun main(args: Array<String>) {
    val exitCode = CommandLine(MainCommand()).execute(*args)
    if (exitCode != 0) {
        exitProcess(exitCode)
    }
}
