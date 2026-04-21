package ragas.metrics.collections

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.intOrNull
import ragas.llms.BaseRagasLlm
import ragas.metrics.BaseMetric
import ragas.metrics.COMMON_STOP_WORDS
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.MetricWithLlm
import ragas.metrics.MultiTurnMetric
import ragas.metrics.SingleTurnMetric
import ragas.metrics.clamp01
import ragas.metrics.defaults.LlmJsonSupport
import ragas.metrics.tokenize
import ragas.model.AiMessage
import ragas.model.ConversationMessage
import ragas.model.HumanMessage
import ragas.model.MultiTurnSample
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import kotlin.math.pow
import kotlin.math.round

/**
 * Implements [FactualCorrectnessMetric].
 *
 * @property mode Scoring mode.
 * @property beta F-score beta parameter.
 * @property atomicity Property `atomicity`.
 * @property coverage Property `coverage`.
 */
class FactualCorrectnessMetric(
    name: String = "factual_correctness",
    private val mode: Mode = Mode.F1,
    private val beta: Double = 1.0,
    private val atomicity: DecompositionLevel = DecompositionLevel.LOW,
    private val coverage: DecompositionLevel = DecompositionLevel.LOW,
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response", "reference")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    SingleTurnMetric,
    MetricWithLlm {
    override var llm: BaseRagasLlm? = null

    /**
     * Enumerates Mode values.
     */
    enum class Mode {
        PRECISION,
        RECALL,
        F1,
    }

    /**
     * Enumerates DecompositionLevel values.
     */
    enum class DecompositionLevel {
        LOW,
        HIGH,
    }

    init {
        require(beta.isFinite() && beta > 0.0) {
            "Beta must be a positive finite value."
        }
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
            return llmFactualCorrectnessScore(sample, llmInstance)
        }
        return fallbackFactualCorrectnessScore(sample)
    }

    private suspend fun llmFactualCorrectnessScore(
        sample: SingleTurnSample,
        llmInstance: BaseRagasLlm,
    ): Double {
        val response = sample.response.orEmpty().trim()
        val reference = sample.reference.orEmpty().trim()
        require(response.isNotEmpty()) { "response is missing. Please add response to the test sample." }
        require(reference.isNotEmpty()) { "reference is missing. Please add reference to the test sample." }

        val referenceResponse = decomposeAndVerifyClaims(llmInstance, response, reference)
        val responseReference =
            if (mode == Mode.PRECISION) {
                BooleanArray(0)
            } else {
                decomposeAndVerifyClaims(llmInstance, reference, response)
            }

        val tp = referenceResponse.count { it }
        val fp = referenceResponse.count { !it }
        val fn =
            if (mode == Mode.PRECISION) {
                0
            } else {
                responseReference.count { !it }
            }

        val score =
            when (mode) {
                Mode.PRECISION -> tp.toDouble() / (tp + fp + 1e-8)
                Mode.RECALL -> tp.toDouble() / (tp + fn + 1e-8)
                Mode.F1 -> fBetaFromCounts(tp, fp, fn, beta)
            }
        return round(score * 100.0) / 100.0
    }

    private suspend fun decomposeAndVerifyClaims(
        llmInstance: BaseRagasLlm,
        textToDecompose: String,
        referenceText: String,
    ): BooleanArray {
        val claims = decomposeClaimsWithLlm(llmInstance, textToDecompose)
        if (claims.isEmpty()) {
            return BooleanArray(0)
        }
        val verdicts = verifyClaimsWithLlm(llmInstance, claims, referenceText)
        if (verdicts.isEmpty()) {
            return BooleanArray(0)
        }
        return verdicts.toBooleanArray()
    }

    private suspend fun decomposeClaimsWithLlm(
        llmInstance: BaseRagasLlm,
        text: String,
    ): List<String> {
        val raw =
            llmInstance
                .generateText(prompt = factualClaimDecompositionPrompt(text, atomicity, coverage))
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return emptyList()
        return LlmJsonSupport.readStringArray(parsed, "claims")
    }

    private suspend fun verifyClaimsWithLlm(
        llmInstance: BaseRagasLlm,
        claims: List<String>,
        reference: String,
    ): List<Boolean> {
        val raw =
            llmInstance
                .generateText(prompt = factualNliPrompt(reference, claims))
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return emptyList()
        val statements = (parsed["statements"] as? JsonArray).orEmpty()
        return statements.mapNotNull { element ->
            val obj = element as? JsonObject ?: return@mapNotNull null
            val verdict = LlmJsonSupport.readIntLike(obj, "verdict") ?: return@mapNotNull null
            verdict != 0
        }
    }

    private fun fallbackFactualCorrectnessScore(sample: SingleTurnSample): Double {
        val response = sample.response.orEmpty().trim()
        val reference = sample.reference.orEmpty().trim()
        if (response.isBlank() || reference.isBlank()) {
            return 0.0
        }

        val responseClaims = decomposeClaims(response)
        val referenceClaims = decomposeClaims(reference)
        if (responseClaims.isEmpty() && referenceClaims.isEmpty()) {
            return 1.0
        }
        if (responseClaims.isEmpty()) {
            return 0.0
        }

        val responseSupported = responseClaims.map { claim -> isClaimSupported(claim, referenceClaims) }.toBooleanArray()
        val referenceSupported =
            if (mode == Mode.PRECISION) {
                BooleanArray(0)
            } else {
                referenceClaims.map { claim -> isClaimSupported(claim, responseClaims) }.toBooleanArray()
            }

        val precision = safeDivide(responseSupported.count { it }.toDouble(), responseSupported.size.toDouble())
        val recall =
            if (mode == Mode.PRECISION) {
                0.0
            } else {
                safeDivide(referenceSupported.count { it }.toDouble(), referenceSupported.size.toDouble())
            }

        val score =
            when (mode) {
                Mode.PRECISION -> precision
                Mode.RECALL -> recall
                Mode.F1 -> fBeta(precision = precision, recall = recall, beta = beta)
            }

        return round(clamp01(score) * 100.0) / 100.0
    }

    private fun decomposeClaims(text: String): List<String> {
        val sentenceParts =
            text
                .split(SENTENCE_SPLIT_REGEX)
                .map { it.trim() }
                .filter { it.isNotBlank() }
        if (sentenceParts.isEmpty()) {
            return emptyList()
        }

        val atomicClaims =
            if (atomicity == DecompositionLevel.HIGH) {
                sentenceParts.flatMap { sentence ->
                    sentence
                        .split(CLAIM_SPLIT_REGEX)
                        .map { it.trim() }
                        .filter { it.isNotBlank() }
                }
            } else {
                sentenceParts
            }

        if (coverage == DecompositionLevel.HIGH) {
            return atomicClaims
        }

        return atomicClaims.map { claim ->
            val tokens = meaningfulTokens(claim)
            tokens.take(12).joinToString(" ").ifBlank { claim }
        }
    }

    private fun isClaimSupported(
        claim: String,
        referenceClaims: List<String>,
    ): Boolean {
        if (referenceClaims.isEmpty()) {
            return false
        }
        val claimTokens = meaningfulTokens(claim).toSet()
        if (claimTokens.isEmpty()) {
            return false
        }

        val claimNumbers = numberTokens(claim)
        val bestScore =
            referenceClaims.maxOf { ref ->
                val refTokens = meaningfulTokens(ref).toSet()
                val overlap = claimTokens.intersect(refTokens).size.toDouble()
                val precision = safeDivide(overlap, claimTokens.size.toDouble())
                val recall = safeDivide(overlap, refTokens.size.toDouble())
                var score = (0.6 * precision) + (0.4 * recall)

                val unmatchedCritical =
                    claimTokens
                        .minus(refTokens)
                        .count { token -> token.length >= 5 && token !in GENERIC_FACT_WORDS }
                if (unmatchedCritical > 0) {
                    score *= (1.0 - (0.6 * unmatchedCritical.coerceAtMost(2)))
                }

                val refNumbers = numberTokens(ref)
                if (claimNumbers.isNotEmpty() && refNumbers.isNotEmpty() && claimNumbers.intersect(refNumbers).isEmpty()) {
                    score *= 0.25
                }
                if (claimNumbers.isNotEmpty() && refNumbers.isEmpty()) {
                    score *= 0.6
                }
                score
            }

        return bestScore >= 0.5
    }

    private fun fBeta(
        precision: Double,
        recall: Double,
        beta: Double,
    ): Double {
        if (precision + recall == 0.0) {
            return 0.0
        }
        val betaSq = beta.pow(2.0)
        return ((1.0 + betaSq) * precision * recall) / ((betaSq * precision) + recall)
    }

    private fun fBetaFromCounts(
        tp: Int,
        fp: Int,
        fn: Int,
        beta: Double,
    ): Double {
        val precision = tp.toDouble() / (tp + fp + 1e-8)
        val recall = tp.toDouble() / (tp + fn + 1e-8)
        val betaSq = beta.pow(2.0)
        return ((1.0 + betaSq) * precision * recall) / ((betaSq * precision) + recall + 1e-8)
    }

    private fun safeDivide(
        numerator: Double,
        denominator: Double,
    ): Double =
        if (denominator <= 0.0) {
            0.0
        } else {
            numerator / denominator
        }

    private fun meaningfulTokens(text: String): List<String> =
        tokenize(text).filter { token ->
            token.length > 2 &&
                token !in COMMON_STOP_WORDS
        }

    private fun numberTokens(text: String): Set<String> = NUMBER_TOKEN_REGEX.findAll(text.lowercase()).map { it.value }.toSet()

    private companion object {
        val SENTENCE_SPLIT_REGEX = Regex("[.!?]+")
        val CLAIM_SPLIT_REGEX = Regex("\\b(?:and|but|while|whereas|although)\\b|,")
        val NUMBER_TOKEN_REGEX = Regex("\\b\\d+(?:[.,]\\d+)?\\b")
        val GENERIC_FACT_WORDS =
            setOf(
                "answer",
                "question",
                "response",
                "information",
                "process",
                "system",
                "theory",
                "model",
            )
    }
}

private fun factualClaimDecompositionPrompt(
    response: String,
    atomicity: FactualCorrectnessMetric.DecompositionLevel,
    coverage: FactualCorrectnessMetric.DecompositionLevel,
): String =
    buildString {
        appendLine("Decompose and break down each of the input sentences into one or more standalone statements.")
        appendLine("Each statement should be a standalone claim that can be independently verified.")
        appendLine("Follow atomicity and coverage settings from input.")
        appendLine("Return JSON only with this shape: {\"claims\":[\"...\"]}")
        appendLine()
        appendLine("Input:")
        appendLine(
            "{\"response\":${JsonPrimitive(
                response,
            )},\"atomicity\":${JsonPrimitive(atomicity.name.lowercase())},\"coverage\":${JsonPrimitive(coverage.name.lowercase())}}",
        )
        append("Output:")
    }

private fun factualNliPrompt(
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

private fun topicExtractionPrompt(conversation: String): String =
    buildString {
        appendLine("Given an interaction between Human, Tool and AI, extract the topics from Human's input.")
        appendLine("Return JSON only with this shape: {\"topics\":[\"...\"]}")
        appendLine()
        appendLine("Input:")
        appendLine("{\"user_input\":${JsonPrimitive(conversation)}}")
        append("Output:")
    }

private fun topicRefusedPrompt(
    conversation: String,
    topic: String,
): String =
    buildString {
        appendLine("Given a topic, classify if the AI refused to answer the question about the topic.")
        appendLine("Return JSON only with this shape: {\"refused_to_answer\": true}")
        appendLine()
        appendLine("Input:")
        appendLine("{\"user_input\":${JsonPrimitive(conversation)},\"topic\":${JsonPrimitive(topic)}}")
        append("Output:")
    }

private fun topicClassificationPrompt(
    referenceTopics: List<String>,
    topics: List<String>,
): String {
    val referenceJson =
        referenceTopics.joinToString(
            separator = ",",
            prefix = "[",
            postfix = "]",
        ) { topic -> JsonPrimitive(topic).toString() }
    val topicsJson = topics.joinToString(separator = ",", prefix = "[", postfix = "]") { topic -> JsonPrimitive(topic).toString() }
    return buildString {
        appendLine("Given a set of topics classify if each topic falls into any of the given reference topics.")
        appendLine("Return JSON only with this shape: {\"classifications\":[true,false]}")
        appendLine()
        appendLine("Input:")
        appendLine("{\"reference_topics\":$referenceJson,\"topics\":$topicsJson}")
        append("Output:")
    }
}

/**
 * Implements [TopicAdherenceMetric].
 *
 * @property mode Scoring mode.
 */
class TopicAdherenceMetric(
    name: String = "topic_adherence",
    private val mode: Mode = Mode.F1,
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.MULTI_TURN to setOf("user_input", "reference_topics")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    MultiTurnMetric,
    MetricWithLlm {
    override var llm: BaseRagasLlm? = null

    /**
     * Enumerates Mode values.
     */
    enum class Mode {
        PRECISION,
        RECALL,
        F1,
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
     * Executes multiTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun multiTurnAscore(sample: MultiTurnSample): Any {
        val llmInstance = llm
        if (llmInstance != null) {
            return llmTopicAdherenceScore(sample, llmInstance)
        }
        return fallbackTopicAdherenceScore(sample)
    }

    private suspend fun llmTopicAdherenceScore(
        sample: MultiTurnSample,
        llmInstance: BaseRagasLlm,
    ): Double {
        val referenceTopics =
            sample.referenceTopics
                .orEmpty()
                .map { it.trim() }
                .filter { it.isNotBlank() }
        require(referenceTopics.isNotEmpty()) { "reference_topics must be a non-empty list of topics" }

        val conversation = sample.userInput.joinToString(separator = "\n") { message -> message.prettyRepr() }
        val topics = extractTopicsWithLlm(llmInstance, conversation)
        if (topics.isEmpty()) {
            return Double.NaN
        }

        val topicAnswered = checkTopicsAnsweredWithLlm(llmInstance, conversation, topics)
        val topicClassifications = classifyTopicsWithLlm(llmInstance, referenceTopics, topics)
        return computeScore(topicAnswered, topicClassifications)
    }

    private suspend fun extractTopicsWithLlm(
        llmInstance: BaseRagasLlm,
        conversation: String,
    ): List<String> {
        val raw =
            llmInstance
                .generateText(prompt = topicExtractionPrompt(conversation))
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw) ?: return emptyList()
        return LlmJsonSupport.readStringArray(parsed, "topics")
    }

    private suspend fun checkTopicsAnsweredWithLlm(
        llmInstance: BaseRagasLlm,
        conversation: String,
        topics: List<String>,
    ): BooleanArray {
        val answered = BooleanArray(topics.size)
        topics.forEachIndexed { index, topic ->
            val raw =
                llmInstance
                    .generateText(prompt = topicRefusedPrompt(conversation, topic))
                    .generations
                    .firstOrNull()
                    ?.text
                    .orEmpty()
            val parsed = LlmJsonSupport.parseFirstJsonObject(raw)
            val refused = parsed?.let { parseRefusedFlag(it) } ?: true
            answered[index] = !refused
        }
        return answered
    }

    private suspend fun classifyTopicsWithLlm(
        llmInstance: BaseRagasLlm,
        referenceTopics: List<String>,
        topics: List<String>,
    ): BooleanArray {
        val raw =
            llmInstance
                .generateText(prompt = topicClassificationPrompt(referenceTopics, topics))
                .generations
                .firstOrNull()
                ?.text
                .orEmpty()
        val parsed = LlmJsonSupport.parseFirstJsonObject(raw)
        val classifications = parseClassifications(parsed).toMutableList()
        val expected = topics.size
        return when {
            classifications.size < expected -> {
                repeat(expected - classifications.size) { classifications += false }
                classifications.toBooleanArray()
            }

            classifications.size > expected -> {
                classifications.take(expected).toBooleanArray()
            }

            else -> {
                classifications.toBooleanArray()
            }
        }
    }

    private fun computeScore(
        topicAnswered: BooleanArray,
        topicClassifications: BooleanArray,
    ): Double {
        val truePositives = topicAnswered.indices.count { index -> topicAnswered[index] && topicClassifications[index] }.toDouble()
        val falsePositives = topicAnswered.indices.count { index -> topicAnswered[index] && !topicClassifications[index] }.toDouble()
        val falseNegatives = topicAnswered.indices.count { index -> !topicAnswered[index] && topicClassifications[index] }.toDouble()
        val eps = 1e-10

        return when (mode) {
            Mode.PRECISION -> {
                truePositives / (truePositives + falsePositives + eps)
            }

            Mode.RECALL -> {
                truePositives / (truePositives + falseNegatives + eps)
            }

            Mode.F1 -> {
                val precision = truePositives / (truePositives + falsePositives + eps)
                val recall = truePositives / (truePositives + falseNegatives + eps)
                (2.0 * precision * recall) / (precision + recall + eps)
            }
        }
    }

    private fun parseRefusedFlag(root: JsonObject): Boolean {
        val primitive = root["refused_to_answer"] as? JsonPrimitive ?: return true
        primitive.booleanOrNull?.let { return it }
        primitive.intOrNull?.let { return it != 0 }
        return primitive.content.trim().lowercase() in setOf("true", "1", "yes")
    }

    private fun parseClassifications(root: JsonObject?): List<Boolean> {
        val raw = (root?.get("classifications") as? JsonArray).orEmpty()
        return raw.mapNotNull { element ->
            val primitive = element as? JsonPrimitive ?: return@mapNotNull null
            primitive.booleanOrNull
                ?: primitive.intOrNull?.let { number -> number != 0 }
                ?: run {
                    val text = primitive.content.trim().lowercase()
                    when (text) {
                        "true", "1", "yes" -> true
                        "false", "0", "no" -> false
                        else -> null
                    }
                }
        }
    }

    private fun fallbackTopicAdherenceScore(sample: MultiTurnSample): Double {
        val referenceTopics =
            sample.referenceTopics
                .orEmpty()
                .map { normalizeTopic(it) }
                .filter { it.isNotBlank() }
        if (referenceTopics.isEmpty()) {
            return 0.0
        }

        val extracted = extractTopics(sample.userInput, referenceTopics)
        if (extracted.isEmpty()) {
            return 0.0
        }

        val topicAnswered = extracted.map { entry -> isTopicAnswered(sample.userInput, entry.turnIndex) }.toBooleanArray()
        val topicClassifications = extracted.map { entry -> isReferenceAligned(entry.topic, referenceTopics) }.toBooleanArray()

        val truePositives = topicAnswered.indices.count { i -> topicAnswered[i] && topicClassifications[i] }
        val falsePositives = topicAnswered.indices.count { i -> topicAnswered[i] && !topicClassifications[i] }
        val falseNegatives = topicAnswered.indices.count { i -> !topicAnswered[i] && topicClassifications[i] }

        val precision = safeDivide(truePositives.toDouble(), (truePositives + falsePositives).toDouble())
        val recall = safeDivide(truePositives.toDouble(), (truePositives + falseNegatives).toDouble())
        val score =
            when (mode) {
                Mode.PRECISION -> precision
                Mode.RECALL -> recall
                Mode.F1 -> if (precision + recall == 0.0) 0.0 else (2.0 * precision * recall) / (precision + recall)
            }
        return clamp01(score)
    }

    private fun extractTopics(
        messages: List<ConversationMessage>,
        referenceTopics: List<String>,
    ): List<ExtractedTopic> {
        val extracted = mutableListOf<ExtractedTopic>()

        messages.forEachIndexed { index, message ->
            if (message !is HumanMessage) {
                return@forEachIndexed
            }
            val content = message.content
            val contentTokens = topicTokenSet(content)

            var matchedReference = false
            referenceTopics.forEach { topic ->
                val topicTokens = topicTokenSet(topic)
                if (hasSufficientTopicOverlap(topicTokens, contentTokens)) {
                    extracted += ExtractedTopic(topic = topic, turnIndex = index)
                    matchedReference = true
                }
            }

            if (!matchedReference) {
                inferOutOfScopeTopic(content)?.let { inferred ->
                    extracted += ExtractedTopic(topic = inferred, turnIndex = index)
                }
            }
        }

        return extracted.distinctBy { "${it.turnIndex}|${it.topic}" }
    }

    private fun isTopicAnswered(
        messages: List<ConversationMessage>,
        humanTurnIndex: Int,
    ): Boolean {
        val nextHumanIndex =
            messages
                .indices
                .firstOrNull { idx -> idx > humanTurnIndex && messages[idx] is HumanMessage } ?: messages.size

        val aiReplies =
            messages
                .subList(humanTurnIndex + 1, nextHumanIndex)
                .filterIsInstance<AiMessage>()

        if (aiReplies.isEmpty()) {
            return false
        }

        val nonRefusalReply =
            aiReplies.any { ai ->
                val normalized = ai.content.lowercase().trim()
                normalized.isNotBlank() &&
                    REFUSAL_PATTERNS.none { pattern -> pattern.containsMatchIn(normalized) }
            }

        return nonRefusalReply || aiReplies.any { ai -> !ai.toolCalls.isNullOrEmpty() }
    }

    private fun isReferenceAligned(
        topic: String,
        referenceTopics: List<String>,
    ): Boolean {
        val topicTokens = topicTokenSet(topic)
        if (topicTokens.isEmpty()) {
            return false
        }

        return referenceTopics.any { ref ->
            val refTokens = topicTokenSet(ref)
            hasSufficientTopicOverlap(topicTokens, refTokens)
        }
    }

    private fun topicTokenSet(text: String): Set<String> =
        tokenize(normalizeTopic(text)).filter { token -> token !in COMMON_STOP_WORDS }.toSet()

    private fun hasSufficientTopicOverlap(
        left: Set<String>,
        right: Set<String>,
    ): Boolean {
        if (left.isEmpty() || right.isEmpty()) {
            return false
        }
        val overlap = left.intersect(right).size.toDouble()
        if (overlap < MIN_TOPIC_TOKEN_OVERLAP) {
            return false
        }

        val leftCoverage = overlap / left.size.toDouble()
        val rightCoverage = overlap / right.size.toDouble()
        return leftCoverage >= MIN_TOPIC_OVERLAP_RATIO || rightCoverage >= MIN_TOPIC_OVERLAP_RATIO
    }

    private fun inferOutOfScopeTopic(content: String): String? {
        val normalized = content.lowercase()
        OUT_OF_SCOPE_KEYWORDS.forEach { (topic, keywords) ->
            if (keywords.any { keyword -> keyword in normalized }) {
                return topic
            }
        }
        return null
    }

    private fun normalizeTopic(text: String): String =
        text
            .lowercase()
            .replace(Regex("[^\\p{L}\\p{N}\\s]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()

    private fun safeDivide(
        numerator: Double,
        denominator: Double,
    ): Double =
        if (denominator <= 0.0) {
            0.0
        } else {
            numerator / denominator
        }

    private data class ExtractedTopic(
        val topic: String,
        val turnIndex: Int,
    )

    private companion object {
        const val MIN_TOPIC_TOKEN_OVERLAP = 1.0
        const val MIN_TOPIC_OVERLAP_RATIO = 0.5
        val REFUSAL_PATTERNS =
            listOf(
                Regex("\\b(can'?t|cannot|won'?t|unable|refuse|decline|sorry)\\b"),
                Regex("\\b(not\\s+able|do\\s+not\\s+have|don't\\s+have|outside\\s+my\\s+scope)\\b"),
            )

        val OUT_OF_SCOPE_KEYWORDS =
            mapOf(
                "sports" to setOf("sports", "football", "soccer", "nba", "nfl", "cricket"),
                "finance" to setOf("finance", "stock", "market", "bitcoin", "crypto", "investment"),
                "politics" to setOf("politics", "election", "senate", "president", "government"),
                "travel" to setOf("travel", "hotel", "flight", "tourism", "visa"),
                "cooking" to setOf("cooking", "recipe", "kitchen", "bake", "meal"),
            )
    }
}
