package ragas.metrics.defaults

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import ragas.llms.BaseRagasLlm
import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.MetricWithLlm
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.metrics.tokenSet
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig

/**
 * Implements [FaithfulnessMetric].
 *
 * @property allowHeuristicFallback Whether heuristic fallback is allowed.
 */
class FaithfulnessMetric(
    private val allowHeuristicFallback: Boolean = false,
) : BaseMetric(
        name = "faithfulness",
        requiredColumns =
            mapOf(
                MetricType.SINGLE_TURN to
                    if (allowHeuristicFallback) {
                        setOf("response", "retrieved_contexts")
                    } else {
                        setOf("user_input", "response", "retrieved_contexts")
                    },
            ),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric,
    MetricWithLlm {
    override var llm: BaseRagasLlm? = null

    private val minSentenceCoveragePerContext = 0.5

    /**
     * Executes init.
     * @param runConfig Runtime configuration for model calls and execution behavior.
     */
    override suspend fun init(runConfig: RunConfig) {
        validateRequiredColumns()
        llm?.runConfig = runConfig
    }

    /**
     * Executes singleTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val llmInstance = llm
        if (llmInstance != null) {
            return llmFaithfulnessScore(sample, llmInstance)
        }
        if (!allowHeuristicFallback) {
            throw IllegalStateException(
                "FaithfulnessMetric requires an LLM for parity semantics. " +
                    "Set llm on the metric or use FaithfulnessMetric(allowHeuristicFallback = true).",
            )
        }
        return fallbackFaithfulnessScore(sample)
    }

    private suspend fun llmFaithfulnessScore(
        sample: SingleTurnSample,
        llmInstance: BaseRagasLlm,
    ): Double {
        val userInput = sample.userInput.orEmpty().trim()
        val response = sample.response.orEmpty().trim()
        val contexts = sample.retrievedContexts.orEmpty().filter { it.isNotBlank() }
        require(userInput.isNotBlank()) { "user_input is missing. Please add user_input to the test sample." }
        require(response.isNotBlank()) { "response is missing. Please add response to the test sample." }
        require(contexts.isNotEmpty()) { "retrieved_contexts is missing. Please add retrieved_contexts to the test sample." }

        val statements = generateStatements(userInput, response, llmInstance)
        if (statements.isEmpty()) {
            return Double.NaN
        }

        val verdicts = generateVerdicts(statements, contexts.joinToString("\n"), llmInstance)
        if (verdicts.isEmpty()) {
            return Double.NaN
        }
        val normalizedVerdicts = verdicts.filter { verdict -> verdict == 0 || verdict == 1 }
        if (normalizedVerdicts.size != statements.size) {
            return Double.NaN
        }

        val faithfulCount = normalizedVerdicts.count { verdict -> verdict == 1 }
        return faithfulCount.toDouble() / normalizedVerdicts.size.toDouble()
    }

    private suspend fun generateStatements(
        question: String,
        response: String,
        llmInstance: BaseRagasLlm,
    ): List<String> {
        val prompt =
            buildString {
                appendLine(
                    "Given a question and an answer, analyze the complexity of each sentence in the answer. " +
                        "Break down each sentence into one or more fully understandable statements. " +
                        "Ensure that no pronouns are used in any statement.",
                )
                appendLine("Return JSON only with this shape: {\"statements\": [\"...\"]}")
                appendLine()
                appendLine("Example:")
                appendLine("Question: Who was Albert Einstein and what is he best known for?")
                appendLine(
                    "Answer: He was a German-born theoretical physicist, widely acknowledged to be one " +
                        "of the greatest and most influential physicists of all time. " +
                        "He was best known for developing the theory of relativity, he also made important " +
                        "contributions to the development of the theory of quantum mechanics.",
                )
                appendLine(
                    "Output: {\"statements\": [" +
                        "\"Albert Einstein was a German-born theoretical physicist.\"," +
                        "\"Albert Einstein is recognized as one of the greatest and most influential physicists of all time.\"," +
                        "\"Albert Einstein was best known for developing the theory of relativity.\"," +
                        "\"Albert Einstein made important contributions to the development of the theory of quantum mechanics.\"" +
                        "]}",
                )
                appendLine()
                appendLine("Question: ${JsonPrimitive(question)}")
                appendLine("Answer: ${JsonPrimitive(response)}")
                append("Output:")
            }
        val raw =
            llmInstance
                .generateText(prompt = prompt)
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return emptyList()
        return LlmJsonSupport.readStringArray(parsed, "statements")
    }

    private suspend fun generateVerdicts(
        statements: List<String>,
        context: String,
        llmInstance: BaseRagasLlm,
    ): List<Int> {
        val statementsJson =
            statements.joinToString(
                separator = ",",
                prefix = "[",
                postfix = "]",
            ) { statement -> JsonPrimitive(statement).toString() }
        val prompt =
            buildString {
                appendLine(
                    "Your task is to judge the faithfulness of a series of statements based on a given context. " +
                        "For each statement you must return verdict as 1 if the statement can be directly inferred " +
                        "based on the context or 0 if the statement can not be directly inferred based on the context.",
                )
                appendLine("Return JSON only with this shape:")
                appendLine("{\"statements\":[{\"statement\":\"...\",\"reason\":\"...\",\"verdict\":0}]}")
                appendLine()
                appendLine("Example:")
                appendLine(
                    "Context: John is a student at XYZ University. He is pursuing a degree in Computer Science. " +
                        "He is enrolled in several courses this semester, including Data Structures, Algorithms, " +
                        "and Database Management. John is a diligent student and spends a significant amount of " +
                        "time studying and completing assignments. He often stays late in the library to work on his projects.",
                )
                appendLine(
                    "Statements: [\"John is majoring in Biology.\"," +
                        "\"John is taking a course on Artificial Intelligence.\"," +
                        "\"John is a dedicated student.\"," +
                        "\"John has a part-time job.\"]",
                )
                appendLine(
                    "Output: {\"statements\":[" +
                        "{\"statement\":\"John is majoring in Biology.\"," +
                        "\"reason\":\"John's major is explicitly stated as Computer Science, not Biology.\",\"verdict\":0}," +
                        "{\"statement\":\"John is taking a course on Artificial Intelligence.\"," +
                        "\"reason\":\"The context lists Data Structures, Algorithms, " +
                        "and Database Management but not AI.\",\"verdict\":0}," +
                        "{\"statement\":\"John is a dedicated student.\"," +
                        "\"reason\":\"The context states John is diligent and spends significant time studying.\",\"verdict\":1}," +
                        "{\"statement\":\"John has a part-time job.\"," +
                        "\"reason\":\"There is no information in the context about a part-time job.\",\"verdict\":0}" +
                        "]}",
                )
                appendLine()
                appendLine("Context: ${JsonPrimitive(context)}")
                appendLine("Statements: $statementsJson")
                append("Output:")
            }
        val raw =
            llmInstance
                .generateText(prompt = prompt)
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return emptyList()
        val items = (parsed["statements"] as? JsonArray).orEmpty()
        return items.mapNotNull { element ->
            val obj = element as? JsonObject ?: return@mapNotNull null
            LlmJsonSupport.readIntLike(obj, "verdict")
        }
    }

    private fun fallbackFaithfulnessScore(sample: SingleTurnSample): Double {
        val contexts = sample.retrievedContexts.orEmpty()
        val response = sample.response.orEmpty()
        if (contexts.isEmpty() || response.isBlank()) {
            return 0.0
        }

        val contextTokenSets = contexts.map { context -> tokenSet(context) }
        val responseSentences =
            response
                .split(Regex("[.!?]"))
                .map { sentence -> sentence.trim() }
                .filter { sentence -> sentence.isNotBlank() }

        if (responseSentences.isEmpty()) {
            return 0.0
        }

        val supported =
            responseSentences.count { sentence ->
                val sentenceTokens = tokenSet(sentence)
                sentenceTokens.isNotEmpty() &&
                    contextTokenSets.any { contextTokens ->
                        val overlap = sentenceTokens.intersect(contextTokens).size.toDouble()
                        val coverage = overlap / sentenceTokens.size.toDouble()
                        coverage >= minSentenceCoveragePerContext
                    }
            }

        return clamp01(supported.toDouble() / responseSentences.size.toDouble())
    }
}
