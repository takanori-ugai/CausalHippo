package ragas.metrics.defaults

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import ragas.embeddings.BaseRagasEmbedding
import ragas.llms.BaseRagasLlm
import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.MetricWithEmbeddings
import ragas.metrics.MetricWithLlm
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.metrics.jaccardSimilarity
import ragas.metrics.tokenSet
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import kotlin.math.sqrt

/**
 * Measures how relevant the response is to the user input.
 *
 * Uses LLM-generated question reconstruction plus embedding similarity when available,
 * with an optional lexical heuristic fallback.
 *
 * @property allowHeuristicFallback Whether heuristic fallback is allowed.
 */
class AnswerRelevancyMetric(
    private val allowHeuristicFallback: Boolean = false,
) : BaseMetric(
        name = "answer_relevancy",
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("user_input", "response")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric,
    MetricWithLlm,
    MetricWithEmbeddings {
    override var llm: BaseRagasLlm? = null
    override var embeddings: BaseRagasEmbedding? = null
    var strictness: Int = 3

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
        val embeddingInstance = embeddings
        if (llmInstance != null && embeddingInstance != null) {
            return llmAnswerRelevancyScore(sample, llmInstance, embeddingInstance)
        }
        if (!allowHeuristicFallback) {
            throw IllegalStateException(
                "AnswerRelevancyMetric requires both LLM and embeddings for parity semantics. " +
                    "Set llm+embeddings on the metric or use AnswerRelevancyMetric(allowHeuristicFallback = true).",
            )
        }
        return fallbackAnswerRelevancyScore(sample)
    }

    private suspend fun llmAnswerRelevancyScore(
        sample: SingleTurnSample,
        llmInstance: BaseRagasLlm,
        embeddingInstance: BaseRagasEmbedding,
    ): Double {
        val userInput = sample.userInput.orEmpty().trim()
        val response = sample.response.orEmpty().trim()
        require(userInput.isNotEmpty()) { "user_input cannot be empty" }
        require(response.isNotEmpty()) { "response cannot be empty" }

        val generatedQuestions = mutableListOf<String>()
        val noncommittalFlags = mutableListOf<Int>()
        repeat(strictness.coerceAtLeast(1)) {
            val raw =
                llmInstance
                    .generateText(prompt = answerRelevancePrompt(response))
                    .generations
                    .firstOrNull()
                    ?.text
                    .orEmpty()
            val parsed = LlmJsonSupport.parseFirstJsonObject(raw)
            val generatedQuestion = parsed?.readString("question").orEmpty()
            if (generatedQuestion.isNotBlank()) {
                generatedQuestions += generatedQuestion
                noncommittalFlags += parsed?.let { LlmJsonSupport.readIntLike(it, "noncommittal") } ?: 0
            }
        }
        if (generatedQuestions.isEmpty()) {
            return 0.0
        }

        val questionEmbedding = embeddingInstance.embedText(userInput)
        val generatedEmbeddings = embeddingInstance.embedTexts(generatedQuestions)
        val cosineScores =
            generatedEmbeddings.map { generated ->
                cosineSimilarity(questionEmbedding, generated)
            }
        val meanCosine = cosineScores.average()
        val allNoncommittal = noncommittalFlags.all { flag -> flag != 0 }
        return if (allNoncommittal) 0.0 else meanCosine
    }

    private fun fallbackAnswerRelevancyScore(sample: SingleTurnSample): Double {
        val questionTokens = tokenSet(sample.userInput.orEmpty())
        val answerTokens = tokenSet(sample.response.orEmpty())
        return clamp01(jaccardSimilarity(questionTokens, answerTokens))
    }
}

private fun answerRelevancePrompt(response: String): String =
    buildString {
        appendLine("Generate a question for the given answer and identify if the answer is noncommittal.")
        appendLine(
            "Give noncommittal as 1 if the answer is noncommittal (evasive, vague, or ambiguous) and 0 if the answer is substantive.",
        )
        appendLine("Examples of noncommittal answers: \"I don't know\", \"I'm not sure\", \"It depends\".")
        appendLine("Return your response as JSON with this format: {\"question\":\"...\",\"noncommittal\":0}")
        appendLine()
        appendLine("Input:")
        appendLine("{\"response\":${JsonPrimitive(response)}}")
        append("Output:")
    }

private fun cosineSimilarity(
    left: List<Float>,
    right: List<Float>,
): Double {
    if (left.isEmpty() || right.isEmpty()) {
        return 0.0
    }
    val size = minOf(left.size, right.size)
    if (size == 0) {
        return 0.0
    }
    var dot = 0.0
    var leftNorm = 0.0
    var rightNorm = 0.0
    repeat(size) { index ->
        val l = left[index].toDouble()
        val r = right[index].toDouble()
        dot += l * r
        leftNorm += l * l
        rightNorm += r * r
    }
    if (leftNorm == 0.0 || rightNorm == 0.0) {
        return 0.0
    }
    return dot / (sqrt(leftNorm) * sqrt(rightNorm))
}

private fun JsonObject.readString(key: String): String? {
    val primitive = this[key] as? JsonPrimitive ?: return null
    return primitive.contentOrNull?.trim()
}
