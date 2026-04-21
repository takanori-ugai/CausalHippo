package ragas.metrics.collections

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import ragas.llms.BaseRagasLlm
import ragas.metrics.BaseMetric
import ragas.metrics.COMMON_STOP_WORDS
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.MetricWithLlm
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.metrics.defaults.LlmJsonSupport
import ragas.metrics.tokenize
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig

/**
 * Measures how sensitive an answer is to noisy or irrelevant retrieved context.
 *
 * @property mode Controls whether the score tracks relevant-context noise effects or
 * irrelevant-context noise effects.
 */
class NoiseSensitivityMetric(
    name: String = "noise_sensitivity",
    private val mode: Mode = Mode.RELEVANT,
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("user_input", "response", "reference", "retrieved_contexts")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric,
    MetricWithLlm {
    override var llm: BaseRagasLlm? = null

    /**
     * Enumerates Mode values.
     */
    enum class Mode {
        RELEVANT,
        IRRELEVANT,
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
            return llmNoiseSensitivityScore(sample, llmInstance)
        }
        return fallbackNoiseSensitivityScore(sample)
    }

    private suspend fun llmNoiseSensitivityScore(
        sample: SingleTurnSample,
        llmInstance: BaseRagasLlm,
    ): Double {
        val userInput = sample.userInput.orEmpty().trim()
        val response = sample.response.orEmpty().trim()
        val reference = sample.reference.orEmpty().trim()
        val contexts = sample.retrievedContexts.orEmpty().filter { it.isNotBlank() }
        require(reference.isNotEmpty()) { "reference is missing. Please add reference to the test sample." }
        require(userInput.isNotEmpty()) { "user_input is missing. Please add user_input to the test sample." }
        require(response.isNotEmpty()) { "response is missing. Please add response to the test sample." }
        require(contexts.isNotEmpty()) { "retrieved_contexts is missing. Please add retrieved_contexts to the test sample." }

        val gtStatements = decomposeStatementsWithLlm(llmInstance, reference, userInput)
        val ansStatements = decomposeStatementsWithLlm(llmInstance, response, userInput)
        if (ansStatements.isEmpty()) {
            return Double.NaN
        }

        val retrievedToGroundTruth = matrixFaithfulnessWithLlm(llmInstance, gtStatements, contexts)
        val retrievedToAnswer = matrixFaithfulnessWithLlm(llmInstance, ansStatements, contexts)
        val groundTruthToAnswer = evaluateFaithfulnessWithLlm(llmInstance, ansStatements, reference)

        val relevantContexts = contexts.indices.map { ctxIdx -> retrievedToGroundTruth.any { row -> row[ctxIdx] } }
        val relevantFaithful =
            ansStatements.indices.map { stmtIdx ->
                contexts.indices.any { ctxIdx -> relevantContexts[ctxIdx] && retrievedToAnswer[stmtIdx][ctxIdx] }
            }

        val incorrect = groundTruthToAnswer.map { faithful -> !faithful }

        val signal =
            if (mode == Mode.RELEVANT) {
                relevantFaithful.zip(incorrect).map { (faithful, wrong) -> faithful && wrong }
            } else {
                val irrelevantFaithfulRaw =
                    ansStatements.indices.map { stmtIdx ->
                        contexts.indices.any { ctxIdx -> !relevantContexts[ctxIdx] && retrievedToAnswer[stmtIdx][ctxIdx] }
                    }
                val irrelevantFaithful =
                    irrelevantFaithfulRaw.zip(relevantFaithful).map { (irr, rel) -> irr && !rel }
                irrelevantFaithful.zip(incorrect).map { (faithful, wrong) -> faithful && wrong }
            }

        if (signal.isEmpty()) {
            return Double.NaN
        }
        return signal.count { it }.toDouble() / signal.size.toDouble()
    }

    private suspend fun decomposeStatementsWithLlm(
        llmInstance: BaseRagasLlm,
        text: String,
        question: String,
    ): List<String> {
        val raw =
            llmInstance
                .generateText(prompt = noiseStatementGeneratorPrompt(question, text))
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return emptyList()
        return LlmJsonSupport.readStringArray(parsed, "statements")
    }

    private suspend fun evaluateFaithfulnessWithLlm(
        llmInstance: BaseRagasLlm,
        statements: List<String>,
        context: String,
    ): List<Boolean> {
        val raw =
            llmInstance
                .generateText(prompt = noiseFaithfulnessPrompt(context, statements))
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return emptyList()
        val verdicts = (parsed["statements"] as? JsonArray).orEmpty()
        return verdicts.mapNotNull { element ->
            val obj = element as? JsonObject ?: return@mapNotNull null
            val verdict = LlmJsonSupport.readIntLike(obj, "verdict") ?: return@mapNotNull null
            verdict != 0
        }
    }

    private suspend fun matrixFaithfulnessWithLlm(
        llmInstance: BaseRagasLlm,
        statements: List<String>,
        contexts: List<String>,
    ): List<List<Boolean>> {
        if (statements.isEmpty()) {
            return emptyList()
        }
        val resultsPerContext =
            contexts.map { ctx ->
                evaluateFaithfulnessWithLlm(llmInstance, statements, ctx)
            }
        return statements.indices.map { stmtIdx ->
            contexts.indices.map { ctxIdx ->
                resultsPerContext.getOrNull(ctxIdx)?.getOrNull(stmtIdx) ?: false
            }
        }
    }

    private fun fallbackNoiseSensitivityScore(sample: SingleTurnSample): Double {
        val userInput = sample.userInput.orEmpty().trim()
        val response = sample.response.orEmpty().trim()
        val reference = sample.reference.orEmpty().trim()
        val contexts = sample.retrievedContexts.orEmpty().filter { it.isNotBlank() }
        if (userInput.isBlank() || response.isBlank() || reference.isBlank() || contexts.isEmpty()) {
            return 0.0
        }

        val gtStatements = decomposeStatements(reference)
        val ansStatements = decomposeStatements(response)
        if (gtStatements.isEmpty() || ansStatements.isEmpty()) {
            return 0.0
        }

        val retrievedToGroundTruth = matrixFaithfulness(gtStatements, contexts)
        val retrievedToAnswer = matrixFaithfulness(ansStatements, contexts)
        val groundTruthToAnswer = ansStatements.map { stmt -> statementFaithful(stmt, reference) }

        val relevantContexts = contexts.indices.map { ctxIdx -> retrievedToGroundTruth.any { row -> row[ctxIdx] } }
        val relevantFaithful =
            ansStatements.indices.map { stmtIdx ->
                contexts.indices.any { ctxIdx -> relevantContexts[ctxIdx] && retrievedToAnswer[stmtIdx][ctxIdx] }
            }

        val incorrect = groundTruthToAnswer.map { faithful -> !faithful }

        val signal =
            if (mode == Mode.RELEVANT) {
                relevantFaithful.zip(incorrect).map { (faithful, wrong) -> faithful && wrong }
            } else {
                val irrelevantFaithfulRaw =
                    ansStatements.indices.map { stmtIdx ->
                        contexts.indices.any { ctxIdx -> !relevantContexts[ctxIdx] && retrievedToAnswer[stmtIdx][ctxIdx] }
                    }
                val irrelevantFaithful =
                    irrelevantFaithfulRaw.zip(relevantFaithful).map { (irr, rel) -> irr && !rel }
                irrelevantFaithful.zip(incorrect).map { (faithful, wrong) -> faithful && wrong }
            }

        return clamp01(signal.count { it }.toDouble() / signal.size.toDouble())
    }

    private fun matrixFaithfulness(
        statements: List<String>,
        contexts: List<String>,
    ): List<List<Boolean>> = statements.map { statement -> contexts.map { ctx -> statementFaithful(statement, ctx) } }

    private fun decomposeStatements(text: String): List<String> =
        text
            .split(SENTENCE_SPLIT_REGEX)
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .flatMap { sentence ->
                sentence
                    .split(CLAUSE_SPLIT_REGEX)
                    .map { it.trim() }
                    .filter { it.isNotBlank() }
                    .map { clause ->
                        val tokens = tokenize(clause).filter { it !in COMMON_STOP_WORDS }
                        tokens.joinToString(" ").ifBlank { clause }
                    }
            }

    private fun statementFaithful(
        statement: String,
        context: String,
    ): Boolean {
        val stmtTokens = tokenize(statement).filter { it.length > 2 && it !in COMMON_STOP_WORDS }.toSet()
        val ctxTokens = tokenize(context).filter { it.length > 2 && it !in COMMON_STOP_WORDS }.toSet()
        if (stmtTokens.isEmpty() || ctxTokens.isEmpty()) {
            return false
        }

        val overlap = stmtTokens.intersect(ctxTokens).size.toDouble()
        val coverage = overlap / stmtTokens.size.toDouble()
        var score = coverage

        val stmtNums = NUMBER_REGEX.findAll(statement.lowercase()).map { it.value }.toSet()
        val ctxNums = NUMBER_REGEX.findAll(context.lowercase()).map { it.value }.toSet()
        if (stmtNums.isNotEmpty() && ctxNums.isNotEmpty() && stmtNums.intersect(ctxNums).isEmpty()) {
            score *= 0.2
        }

        return score >= 0.45
    }

    private companion object {
        val SENTENCE_SPLIT_REGEX = Regex("[.!?]+")
        val CLAUSE_SPLIT_REGEX = Regex("\\b(?:and|but|while|whereas|although)\\b|,")
        val NUMBER_REGEX = Regex("\\b\\d+(?:[.,]\\d+)?\\b")
    }
}

private fun noiseStatementGeneratorPrompt(
    question: String,
    text: String,
): String =
    buildString {
        appendLine("Given a question and an answer, analyze the complexity of each sentence in the answer.")
        appendLine("Break down each sentence into one or more fully understandable statements.")
        appendLine("Ensure that no pronouns are used in any statement.")
        appendLine("Return JSON only with this shape: {\"statements\":[\"...\"]}")
        appendLine()
        appendLine("Input:")
        appendLine("{\"question\":${JsonPrimitive(question)},\"answer\":${JsonPrimitive(text)}}")
        append("Output:")
    }

private fun noiseFaithfulnessPrompt(
    context: String,
    statements: List<String>,
): String {
    val statementsJson =
        statements.joinToString(
            separator = ",",
            prefix = "[",
            postfix = "]",
        ) { statement -> JsonPrimitive(statement).toString() }
    return buildString {
        appendLine("Your task is to judge the faithfulness of statements based on a given context.")
        appendLine("For each statement, return verdict as 1 if directly inferable from context, else 0.")
        appendLine("Return JSON only with this shape:")
        appendLine("{\"statements\":[{\"statement\":\"...\",\"reason\":\"...\",\"verdict\":0}]}")
        appendLine()
        appendLine("Input:")
        appendLine("{\"context\":${JsonPrimitive(context)},\"statements\":$statementsJson}")
        append("Output:")
    }
}

/**
 * Scores summary quality from coverage and optional conciseness.
 *
 * @property lengthPenalty Whether to blend in a conciseness penalty.
 * @property coeff Mixing coefficient for conciseness when [lengthPenalty] is enabled.
 */
class SummaryScoreMetric(
    name: String = "summary_score",
    private val lengthPenalty: Boolean = true,
    private val coeff: Double = 0.5,
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("reference_contexts", "response")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric,
    MetricWithLlm {
    override var llm: BaseRagasLlm? = null

    init {
        require(coeff in 0.0..1.0) {
            "Coefficient must be between 0.0 and 1.0, got $coeff"
        }
    }

    /**
     * Initializes the metric by validating required columns and configuring the LLM.
     *
     * @param runConfig Runtime configuration for model calls and execution behavior.
     */
    override suspend fun init(runConfig: RunConfig) {
        validateRequiredColumns()
        llm?.runConfig = runConfig
    }

    /**
     * Computes summary quality score by evaluating QA-style content coverage and,
     * when enabled, a conciseness component.
     *
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val llmInstance = llm
        if (llmInstance != null) {
            return llmSummaryScore(sample, llmInstance)
        }
        return fallbackSummaryScore(sample)
    }

    private suspend fun llmSummaryScore(
        sample: SingleTurnSample,
        llmInstance: BaseRagasLlm,
    ): Double {
        val referenceContexts = sample.referenceContexts.orEmpty().filter { it.isNotBlank() }
        val response = sample.response.orEmpty().trim()
        require(referenceContexts.isNotEmpty()) { "reference_contexts cannot be empty or contain only whitespace" }
        require(response.isNotEmpty()) { "response cannot be empty or whitespace only" }

        val text = referenceContexts.joinToString("\n")
        var keyphrases = extractKeyphrasesWithLlm(llmInstance, text)
        if (keyphrases.isEmpty()) {
            keyphrases = emptyList()
        }

        var questions = generateQuestionsWithLlm(llmInstance, text, keyphrases)
        if (questions.isEmpty()) {
            questions = emptyList()
        }

        val answers = generateAnswersWithLlm(llmInstance, response, questions)
        val qaScore = computeQaScore(answers)

        if (!lengthPenalty) {
            return qaScore
        }

        val conciseness = computeConcisenessScoreLlm(text, response)
        return (qaScore * (1.0 - coeff)) + (conciseness * coeff)
    }

    private suspend fun extractKeyphrasesWithLlm(
        llmInstance: BaseRagasLlm,
        text: String,
    ): List<String> {
        val raw =
            llmInstance
                .generateText(prompt = summaryExtractKeyphrasesPrompt(text))
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return emptyList()
        return LlmJsonSupport.readStringArray(parsed, "keyphrases")
    }

    private suspend fun generateQuestionsWithLlm(
        llmInstance: BaseRagasLlm,
        text: String,
        keyphrases: List<String>,
    ): List<String> {
        val raw =
            llmInstance
                .generateText(prompt = summaryGenerateQuestionsPrompt(text, keyphrases))
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return emptyList()
        return LlmJsonSupport.readStringArray(parsed, "questions")
    }

    private suspend fun generateAnswersWithLlm(
        llmInstance: BaseRagasLlm,
        summary: String,
        questions: List<String>,
    ): List<String> {
        val raw =
            llmInstance
                .generateText(prompt = summaryGenerateAnswersPrompt(summary, questions))
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return emptyList()
        return LlmJsonSupport.readStringArray(parsed, "answers")
    }

    private fun computeQaScore(answers: List<String>): Double {
        val correct = answers.count { answer -> answer.lowercase() == "1" }
        return correct.toDouble() / answers.size.toDouble()
    }

    private fun computeConcisenessScoreLlm(
        text: String,
        summary: String,
    ): Double = 1.0 - (minOf(summary.length, text.length).toDouble() / (text.length.toDouble() + 1e-10))

    private fun fallbackSummaryScore(sample: SingleTurnSample): Double {
        val referenceContexts = sample.referenceContexts.orEmpty().filter { it.isNotBlank() }
        val response = sample.response.orEmpty().trim()
        if (referenceContexts.isEmpty() || response.isBlank()) {
            return 0.0
        }
        val text = referenceContexts.joinToString("\n")
        val keyphrases = extractKeyphrases(text)
        val qaScore = computeQaScore(response, keyphrases)

        if (!lengthPenalty) {
            return clamp01(qaScore)
        }

        val conciseness = computeConcisenessScore(text, response)
        val finalScore = (qaScore * (1.0 - coeff)) + (conciseness * coeff)
        return clamp01(finalScore)
    }

    private fun extractKeyphrases(text: String): List<String> {
        val phrases = linkedSetOf<String>()
        KEYPHRASE_REGEX.findAll(text).forEach { match ->
            val phrase = match.value.trim()
            if (phrase.length >= 4) {
                phrases += phrase
            }
        }

        val tokenCounts =
            tokenize(text)
                .filter { token -> token.length >= 5 && token !in COMMON_STOP_WORDS }
                .groupingBy { it }
                .eachCount()
                .toList()
                .sortedByDescending { (_, count) -> count }
                .take(12)
                .map { (token, _) -> token }
        phrases += tokenCounts

        return phrases.toList()
    }

    private fun computeQaScore(
        summary: String,
        keyphrases: List<String>,
    ): Double {
        if (keyphrases.isEmpty()) {
            return 0.0
        }
        val summaryTokens = tokenize(summary).toSet()
        if (summaryTokens.isEmpty()) {
            return 0.0
        }

        val answers =
            keyphrases.map { phrase ->
                val phraseTokens = tokenize(phrase).toSet()
                if (phraseTokens.isEmpty()) {
                    false
                } else {
                    val overlap = phraseTokens.intersect(summaryTokens).size.toDouble()
                    val coverage = overlap / phraseTokens.size.toDouble()
                    coverage >= 0.6
                }
            }

        return answers.count { it }.toDouble() / answers.size.toDouble()
    }

    private fun computeConcisenessScore(
        text: String,
        summary: String,
    ): Double = 1.0 - (minOf(summary.length, text.length).toDouble() / text.length.toDouble())

    private companion object {
        val KEYPHRASE_REGEX =
            Regex("\\b(?:[A-Z][a-zA-Z0-9-]*(?:\\s+[A-Z][a-zA-Z0-9-]*)*|\\d{4}|\\$\\d+(?:[.,]\\d+)?\\s*(?:trillion|billion|million)?)\\b")
    }
}

private fun summaryExtractKeyphrasesPrompt(text: String): String =
    buildString {
        appendLine("Extract keyphrases of type: Person, Organization, Location, Date/Time, Monetary Values, and Percentages.")
        appendLine("Return JSON only with this shape: {\"keyphrases\":[\"...\"]}")
        appendLine()
        appendLine("Input:")
        appendLine("{\"text\":${JsonPrimitive(text)}}")
        append("Output:")
    }

private fun summaryGenerateQuestionsPrompt(
    text: String,
    keyphrases: List<String>,
): String {
    val keyphrasesJson =
        keyphrases.joinToString(
            separator = ",",
            prefix = "[",
            postfix = "]",
        ) { keyphrase -> JsonPrimitive(keyphrase).toString() }
    return buildString {
        appendLine("Based on the given text and keyphrases, generate closed-ended questions.")
        appendLine("Questions should ALWAYS be answerable as '1' from the given text.")
        appendLine("Return JSON only with this shape: {\"questions\":[\"...\"]}")
        appendLine()
        appendLine("Input:")
        appendLine("{\"text\":${JsonPrimitive(text)},\"keyphrases\":$keyphrasesJson}")
        append("Output:")
    }
}

private fun summaryGenerateAnswersPrompt(
    summary: String,
    questions: List<String>,
): String {
    val questionsJson =
        questions.joinToString(
            separator = ",",
            prefix = "[",
            postfix = "]",
        ) { question -> JsonPrimitive(question).toString() }
    return buildString {
        appendLine("Given a summary and close-ended questions, return '1' or '0' for each question.")
        appendLine("Return JSON only with this shape: {\"answers\":[\"1\",\"0\"]}")
        appendLine()
        appendLine("Input:")
        appendLine("{\"summary\":${JsonPrimitive(summary)},\"questions\":$questionsJson}")
        append("Output:")
    }
}
