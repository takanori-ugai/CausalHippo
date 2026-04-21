package ragas.metrics.collections

import kotlinx.coroutines.CancellationException
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import ragas.embeddings.BaseRagasEmbedding
import ragas.llms.BaseRagasLlm
import ragas.metrics.BaseMetric
import ragas.metrics.COMMON_STOP_WORDS
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.MetricWithEmbeddings
import ragas.metrics.MetricWithLlm
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.metrics.defaults.LlmJsonSupport
import ragas.metrics.jaccardSimilarity
import ragas.metrics.tokenSet
import ragas.metrics.tokenize
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import kotlin.math.pow

/**
 * Computes answer accuracy by comparing the response against the reference answer.
 *
 * Uses LLM judge ratings when available, with a lexical/number-aware fallback heuristic.
 *
 * @property maxRetries Maximum retries.
 */
class AnswerAccuracyMetric(
    name: String = "answer_accuracy",
    private val maxRetries: Int = 5,
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("user_input", "response", "reference")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric,
    MetricWithLlm {
    override var llm: BaseRagasLlm? = null

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
            return llmAnswerAccuracyScore(sample, llmInstance)
        }
        return fallbackAnswerAccuracyScore(sample)
    }

    private suspend fun llmAnswerAccuracyScore(
        sample: SingleTurnSample,
        llmInstance: BaseRagasLlm,
    ): Double {
        val question = sample.userInput.orEmpty().trim()
        val response = sample.response.orEmpty().trim()
        val reference = sample.reference.orEmpty().trim()
        require(question.isNotEmpty()) { "user_input is missing. Please add user_input to the test sample." }
        require(response.isNotEmpty()) { "response is missing. Please add response to the test sample." }
        require(reference.isNotEmpty()) { "reference is missing. Please add reference to the test sample." }

        val judge1 = getJudgeRating(llmInstance, answerAccuracyJudge1Prompt(question, response, reference))
        val judge2 = getJudgeRating(llmInstance, answerAccuracyJudge2Prompt(question, reference, response))
        val score1 = judge1 / 4.0
        val score2 = judge2 / 4.0
        return averageScores(score1, score2)
    }

    private suspend fun getJudgeRating(
        llmInstance: BaseRagasLlm,
        prompt: String,
    ): Double {
        repeat(maxRetries.coerceAtLeast(1)) { index ->
            try {
                val raw =
                    llmInstance
                        .generateText(prompt = prompt)
                        .generations
                        .firstOrNull()
                        ?.text
                        .orEmpty()
                val parsed = LlmJsonSupport.parseFirstJsonObject(raw)
                val rating = parsed?.let { root -> LlmJsonSupport.readIntLike(root, "rating") }
                val parsedRating = rating ?: return@repeat
                if (parsedRating in setOf(0, 2, 4)) {
                    return parsedRating.toDouble()
                }
            } catch (e: CancellationException) {
                throw e
            } catch (_: Exception) {
                // Mirror Python retry behavior by trying again up to maxRetries.
            }
            if (index == maxRetries.coerceAtLeast(1) - 1) {
                return Double.NaN
            }
        }
        return Double.NaN
    }

    private fun averageScores(
        score1: Double,
        score2: Double,
    ): Double =
        if (!score1.isNaN() && !score2.isNaN()) {
            (score1 + score2) / 2.0
        } else {
            Double.NaN
        }

    private fun fallbackAnswerAccuracyScore(sample: SingleTurnSample): Double {
        val question = sample.userInput.orEmpty().trim()
        val response = sample.response.orEmpty().trim()
        val reference = sample.reference.orEmpty().trim()
        if (question.isBlank() || response.isBlank() || reference.isBlank()) {
            return 0.0
        }
        val normalizedResponse = normalizeText(response)
        val normalizedReference = normalizeText(reference)
        if (normalizedResponse == normalizedReference) {
            return 1.0
        }

        val responseTokens = meaningfulTokenSet(response)
        val referenceTokens = meaningfulTokenSet(reference)
        if (responseTokens.isEmpty() || referenceTokens.isEmpty()) {
            return 0.0
        }

        val overlap = responseTokens.intersect(referenceTokens).size.toDouble()
        val precision = overlap / responseTokens.size.toDouble()
        val recall = overlap / referenceTokens.size.toDouble()
        val lexicalF1 = harmonicMean(precision, recall)

        val referenceNumbers = numericTokenSet(reference)
        val responseNumbers = numericTokenSet(response)
        val numberAlignment = numberAlignmentScore(responseNumbers, referenceNumbers)

        val questionTokens = meaningfulTokenSet(question)
        val questionCoverage =
            if (questionTokens.isEmpty()) {
                1.0
            } else {
                responseTokens.intersect(questionTokens).size.toDouble() / questionTokens.size.toDouble()
            }

        var score =
            (LEXICAL_F1_WEIGHT * lexicalF1) +
                (NUMBER_ALIGNMENT_WEIGHT * numberAlignment) +
                (QUESTION_COVERAGE_WEIGHT * questionCoverage)
        if (referenceNumbers.isNotEmpty() && responseNumbers.isNotEmpty() && responseNumbers.intersect(referenceNumbers).isEmpty()) {
            score *= DISJOINT_NUMBER_PENALTY
        }

        return clamp01(score)
    }

    private companion object {
        const val LEXICAL_F1_WEIGHT = 0.65
        const val NUMBER_ALIGNMENT_WEIGHT = 0.25
        const val QUESTION_COVERAGE_WEIGHT = 0.10
        const val DISJOINT_NUMBER_PENALTY = 0.2
    }
}

private fun answerAccuracyJudge1Prompt(
    query: String,
    userAnswer: String,
    referenceAnswer: String,
): String =
    buildString {
        appendLine(
            "Instruction: You are a world class state of the art assistant for rating a User Answer " +
                "given a Question. The Question is completely answered by the Reference Answer.",
        )
        appendLine(
            "Say 4, if User Answer is full contained and equivalent to Reference Answer in all terms, " +
                "topics, numbers, metrics, dates and units.",
        )
        appendLine(
            "Say 2, if User Answer is partially contained and almost equivalent to Reference Answer in all terms, " +
                "topics, numbers, metrics, dates and units.",
        )
        appendLine(
            "Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, " +
                "topics, numbers, metrics, dates and units or the User Answer do not answer the question.",
        )
        appendLine(
            "Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to " +
                "the instructions above.",
        )
        appendLine("Return your response as JSON in this format: {\"rating\": X} where X is 0, 2, or 4.")
        appendLine()
        appendLine("### Question: ${kotlinx.serialization.json.JsonPrimitive(query)}")
        appendLine("### User Answer: ${kotlinx.serialization.json.JsonPrimitive(userAnswer)}")
        appendLine("### Reference Answer: ${kotlinx.serialization.json.JsonPrimitive(referenceAnswer)}")
        append("The rating is:")
    }

private fun answerAccuracyJudge2Prompt(
    query: String,
    userAnswer: String,
    referenceAnswer: String,
): String =
    buildString {
        appendLine("I will rate the User Answer in comparison to the Reference Answer for a given Question.")
        appendLine(
            "A rating of 4 indicates that the User Answer is entirely consistent with the Reference Answer, " +
                "covering all aspects, topics, numbers, metrics, dates, and units.",
        )
        appendLine(
            "A rating of 2 signifies that the User Answer is mostly aligned with the Reference Answer, " +
                "with minor discrepancies in some areas.",
        )
        appendLine(
            "A rating of 0 means that the User Answer is either inaccurate, incomplete, or unrelated to " +
                "the Reference Answer, or it fails to address the Question.",
        )
        appendLine(
            "I will provide the rating without any explanation or justification, adhering to the following " +
                "scale: 0 (no match), 2 (partial match), 4 (exact match).",
        )
        appendLine("Do not explain or justify my rating. My rating must be only 4, 2 or 0 only.")
        appendLine("Return your response as JSON in this format: {\"rating\": X} where X is 0, 2, or 4.")
        appendLine()
        appendLine("Question: ${kotlinx.serialization.json.JsonPrimitive(query)}")
        appendLine()
        appendLine("Reference Answer: ${kotlinx.serialization.json.JsonPrimitive(referenceAnswer)}")
        appendLine()
        appendLine("User Answer: ${kotlinx.serialization.json.JsonPrimitive(userAnswer)}")
        appendLine()
        append("Rating: ")
    }

/**
 * Computes answer correctness from factual consistency and semantic similarity.
 *
 * Combines factuality classification with embedding-based similarity using configurable weights.
 *
 * @property weights Metric component weights.
 * @property beta F-score beta parameter.
 */
class AnswerCorrectnessMetric(
    name: String = "answer_correctness",
    private val weights: List<Double> = listOf(0.75, 0.25),
    private val beta: Double = 1.0,
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("user_input", "response", "reference")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric,
    MetricWithLlm,
    MetricWithEmbeddings {
    override var llm: BaseRagasLlm? = null
    override var embeddings: BaseRagasEmbedding? = null

    init {
        require(weights.size == 2) {
            "Expects a list of two weights. First for factuality, second for semantic similarity"
        }
        require(weights.any { it != 0.0 }) { "At least one weight must be non-zero" }
        require(weights.all { it >= 0.0 }) { "Weights must be non-negative" }
        require(beta.isFinite() && beta > 0.0) { "Beta must be a positive finite value." }
    }

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
            return llmAnswerCorrectnessScore(sample, llmInstance)
        }
        return fallbackAnswerCorrectnessScore(sample)
    }

    private suspend fun llmAnswerCorrectnessScore(
        sample: SingleTurnSample,
        llmInstance: BaseRagasLlm,
    ): Double {
        val question = sample.userInput.orEmpty().trim()
        val response = sample.response.orEmpty().trim()
        val reference = sample.reference.orEmpty().trim()
        if (question.isEmpty() || response.isEmpty() || reference.isEmpty()) {
            return 0.0
        }

        val responseStatements = generateStatements(llmInstance, question, response)
        val referenceStatements = generateStatements(llmInstance, question, reference)
        if (responseStatements.isEmpty() || referenceStatements.isEmpty()) {
            return Double.NaN
        }
        val classification =
            classifyStatements(llmInstance, question, responseStatements, referenceStatements)
                ?: return Double.NaN
        val factuality = fBetaFromClassification(classification)

        val similarity =
            if (weights[1] == 0.0) {
                0.0
            } else {
                val embeddingInstance =
                    requireNotNull(embeddings) {
                        "Embeddings are required for semantic similarity scoring. Either provide embeddings " +
                            "or set similarity weight to 0 (weights=[1.0, 0.0]) for pure factuality-only evaluation."
                    }
                semanticSimilarityScore(embeddingInstance, response, reference)
            }

        return weightedAverage(listOf(factuality, similarity), weights)
    }

    private suspend fun generateStatements(
        llmInstance: BaseRagasLlm,
        question: String,
        answer: String,
    ): List<String> {
        val raw =
            llmInstance
                .generateText(prompt = answerCorrectnessStatementGeneratorPrompt(question, answer))
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return emptyList()
        return LlmJsonSupport.readStringArray(parsed, "statements")
    }

    private suspend fun classifyStatements(
        llmInstance: BaseRagasLlm,
        question: String,
        answerStatements: List<String>,
        groundTruthStatements: List<String>,
    ): ClassificationCounts? {
        val raw =
            llmInstance
                .generateText(
                    prompt =
                        answerCorrectnessClassifierPrompt(
                            question = question,
                            answerStatements = answerStatements,
                            groundTruthStatements = groundTruthStatements,
                        ),
                ).generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return null
        val tp = parsed.countArrayEntries("TP")
        val fp = parsed.countArrayEntries("FP")
        val fn = parsed.countArrayEntries("FN")
        if (tp == 0 && fp == 0 && fn == 0) {
            return null
        }
        return ClassificationCounts(tp = tp, fp = fp, fn = fn)
    }

    private fun fBetaFromClassification(classification: ClassificationCounts): Double {
        val tp = classification.tp
        val fp = classification.fp
        val fn = classification.fn
        val precision =
            if (tp + fp == 0) {
                if (fn == 0) 1.0 else 0.0
            } else {
                tp.toDouble() / (tp + fp).toDouble()
            }
        val recall =
            if (tp + fn == 0) {
                if (fp == 0) 1.0 else 0.0
            } else {
                tp.toDouble() / (tp + fn).toDouble()
            }
        return fBetaScore(precision, recall)
    }

    private suspend fun semanticSimilarityScore(
        embeddingInstance: BaseRagasEmbedding,
        response: String,
        reference: String,
    ): Double {
        val responseEmbedding = embeddingInstance.embedText(response)
        val referenceEmbedding = embeddingInstance.embedText(reference)
        val cosine = cosineSimilarityDouble(responseEmbedding, referenceEmbedding)
        return clamp01((cosine + 1.0) / 2.0)
    }

    private fun weightedAverage(
        values: List<Double>,
        weights: List<Double>,
    ): Double {
        val weightSum = weights.sum()
        if (weightSum <= 0.0) {
            return 0.0
        }
        val weightedSum = values.zip(weights).sumOf { (value, weight) -> value * weight }
        return weightedSum / weightSum
    }

    private fun fallbackAnswerCorrectnessScore(sample: SingleTurnSample): Double {
        val question = sample.userInput.orEmpty().trim()
        val response = sample.response.orEmpty().trim()
        val reference = sample.reference.orEmpty().trim()
        if (question.isBlank() || response.isBlank() || reference.isBlank()) {
            return 0.0
        }

        val factuality = factualityScore(response, reference)
        val similarity =
            if (weights[1] == 0.0) {
                0.0
            } else {
                lexicalSimilarityScore(response, reference)
            }

        val weightSum = weights[0] + weights[1]
        val combined = ((weights[0] * factuality) + (weights[1] * similarity)) / weightSum
        return clamp01(combined)
    }

    private fun factualityScore(
        response: String,
        reference: String,
    ): Double {
        val responseStatements = toStatements(response)
        val referenceStatements = toStatements(reference)
        if (responseStatements.isEmpty() && referenceStatements.isEmpty()) {
            return 1.0
        }
        if (responseStatements.isEmpty() || referenceStatements.isEmpty()) {
            return 0.0
        }

        val tp =
            responseStatements.count { answerStmt ->
                referenceStatements.any { refStmt ->
                    statementMatch(
                        answerStmt,
                        refStmt,
                    ) >= 0.55
                }
            }
        val fp = responseStatements.size - tp
        val fn =
            referenceStatements.count { refStmt ->
                responseStatements.none { answerStmt -> statementMatch(answerStmt, refStmt) >= 0.55 }
            }

        val precision =
            if (tp + fp == 0) {
                if (fn == 0) 1.0 else 0.0
            } else {
                tp.toDouble() / (tp + fp).toDouble()
            }
        val recall =
            if (tp + fn == 0) {
                if (fp == 0) 1.0 else 0.0
            } else {
                tp.toDouble() / (tp + fn).toDouble()
            }

        val fbeta = fBetaScore(precision, recall)
        val contradictionPenalty = contradictionPenalty(response, reference)
        return clamp01(fbeta * contradictionPenalty)
    }

    private fun lexicalSimilarityScore(
        response: String,
        reference: String,
    ): Double {
        val responseTokens = meaningfulTokenSet(response)
        val referenceTokens = meaningfulTokenSet(reference)
        if (responseTokens.isEmpty() || referenceTokens.isEmpty()) {
            return 0.0
        }

        val jaccard = jaccardSimilarity(responseTokens, referenceTokens)
        val coverage = responseTokens.intersect(referenceTokens).size.toDouble() / referenceTokens.size.toDouble()
        return clamp01((0.7 * jaccard) + (0.3 * coverage))
    }

    private fun statementMatch(
        answerStatement: String,
        referenceStatement: String,
    ): Double {
        val answerTokens = meaningfulTokenSet(answerStatement)
        val referenceTokens = meaningfulTokenSet(referenceStatement)
        if (answerTokens.isEmpty() || referenceTokens.isEmpty()) {
            return 0.0
        }

        var score = jaccardSimilarity(answerTokens, referenceTokens)
        val answerNums = numericTokenSet(answerStatement)
        val referenceNums = numericTokenSet(referenceStatement)
        if (referenceNums.isNotEmpty() && answerNums.isNotEmpty() && answerNums.intersect(referenceNums).isEmpty()) {
            score *= 0.5
        }
        if (referenceNums.isNotEmpty() && answerNums.isEmpty()) {
            score *= 0.7
        }

        return clamp01(score)
    }

    private fun contradictionPenalty(
        response: String,
        reference: String,
    ): Double {
        var penalty = 1.0
        val responseNums = numericTokenSet(response)
        val referenceNums = numericTokenSet(reference)
        if (referenceNums.isNotEmpty() && responseNums.isNotEmpty() && responseNums.intersect(referenceNums).isEmpty()) {
            penalty *= 0.45
        }

        val responseHasNegation = NEGATION_REGEX.containsMatchIn(response.lowercase())
        val referenceHasNegation = NEGATION_REGEX.containsMatchIn(reference.lowercase())
        if (responseHasNegation != referenceHasNegation) {
            penalty *= 0.8
        }

        return penalty
    }

    private fun toStatements(text: String): List<String> =
        text
            .split(STATEMENT_SPLIT_REGEX)
            .map { it.trim() }
            .filter { it.isNotBlank() }

    private fun fBetaScore(
        precision: Double,
        recall: Double,
    ): Double {
        if (precision + recall == 0.0) {
            return 0.0
        }
        val betaSq = beta.pow(2)
        return ((1.0 + betaSq) * precision * recall) / ((betaSq * precision) + recall)
    }

    private companion object {
        val STATEMENT_SPLIT_REGEX = Regex("[.!?;]+")
        val NEGATION_REGEX = Regex("\\b(no|not|never|none|without|cannot|can't|won't|isn't|aren't|didn't|doesn't|don't)\\b")
    }
}

private data class ClassificationCounts(
    val tp: Int,
    val fp: Int,
    val fn: Int,
)

private fun answerCorrectnessStatementGeneratorPrompt(
    question: String,
    answer: String,
): String =
    buildString {
        appendLine("Given a question and an answer, analyze the complexity of each sentence in the answer.")
        appendLine("Break down each sentence into one or more fully understandable statements.")
        appendLine("Ensure that no pronouns are used in any statement.")
        appendLine("Return JSON only with this shape: {\"statements\": [\"...\"]}")
        appendLine()
        appendLine("Input:")
        appendLine("{\"question\":${JsonPrimitive(question)},\"answer\":${JsonPrimitive(answer)}}")
        append("Output:")
    }

private fun answerCorrectnessClassifierPrompt(
    question: String,
    answerStatements: List<String>,
    groundTruthStatements: List<String>,
): String {
    val answerJson =
        answerStatements.joinToString(
            separator = ",",
            prefix = "[",
            postfix = "]",
        ) { statement -> JsonPrimitive(statement).toString() }
    val groundTruthJson =
        groundTruthStatements.joinToString(
            separator = ",",
            prefix = "[",
            postfix = "]",
        ) { statement -> JsonPrimitive(statement).toString() }
    return buildString {
        appendLine("Given a ground truth and answer statements, classify into TP, FP, and FN.")
        appendLine("TP: answer statements directly supported by one or more ground truth statements.")
        appendLine("FP: answer statements not directly supported by any ground truth statement.")
        appendLine("FN: ground truth statements not present in answer.")
        appendLine("Each statement can only belong to one category.")
        appendLine("Return JSON only with this shape:")
        appendLine("{\"TP\":[{\"statement\":\"...\",\"reason\":\"...\"}],\"FP\":[...],\"FN\":[...]}")
        appendLine()
        appendLine("Input:")
        appendLine("{\"question\":${JsonPrimitive(question)},\"answer\":$answerJson,\"ground_truth\":$groundTruthJson}")
        append("Output:")
    }
}

private fun JsonObject.countArrayEntries(key: String): Int = (this[key] as? JsonArray)?.size ?: 0

private fun cosineSimilarityDouble(
    left: List<Float>,
    right: List<Float>,
): Double {
    if (left.isEmpty() || right.isEmpty()) {
        return 0.0
    }
    require(left.size == right.size) {
        "Embedding dimension mismatch: left=${left.size}, right=${right.size}"
    }
    val size = left.size
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
    return dot / (kotlin.math.sqrt(leftNorm) * kotlin.math.sqrt(rightNorm))
}

private fun normalizeText(text: String): String =
    text
        .lowercase()
        .replace(Regex("\\s+"), " ")
        .trim()

private fun harmonicMean(
    precision: Double,
    recall: Double,
): Double =
    if (precision + recall == 0.0) {
        0.0
    } else {
        (2.0 * precision * recall) / (precision + recall)
    }

private fun meaningfulTokenSet(text: String): Set<String> =
    tokenize(text)
        .filter { token -> token.length > 2 && token !in ANSWER_QUALITY_STOP_WORDS }
        .toSet()

private fun numericTokenSet(text: String): Set<String> =
    NUMERIC_TOKEN_REGEX
        .findAll(text.lowercase())
        .map { match -> match.value }
        .toSet()

private fun numberAlignmentScore(
    responseNumbers: Set<String>,
    referenceNumbers: Set<String>,
): Double {
    if (referenceNumbers.isEmpty() && responseNumbers.isEmpty()) {
        return 1.0
    }
    if (referenceNumbers.isEmpty()) {
        return 0.9
    }
    if (responseNumbers.isEmpty()) {
        return 0.4
    }

    val hits = responseNumbers.intersect(referenceNumbers).size.toDouble()
    val precision = hits / responseNumbers.size.toDouble()
    val recall = hits / referenceNumbers.size.toDouble()
    return 0.5 * precision + 0.5 * recall
}

private val NUMERIC_TOKEN_REGEX = Regex("\\b\\d+(?:[.,]\\d+)?\\b")

private val ANSWER_QUALITY_STOP_WORDS =
    COMMON_STOP_WORDS +
        setOf(
            "what",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "whose",
            "why",
            "how",
        )
