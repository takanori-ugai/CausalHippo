package ragas.metrics.collections

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.MultiTurnMetric
import ragas.metrics.clamp01
import ragas.metrics.tokenize
import ragas.model.AiMessage
import ragas.model.ConversationMessage
import ragas.model.HumanMessage
import ragas.model.MultiTurnSample
import ragas.model.ToolMessage

/**
 * Implements [AgentGoalAccuracyWithReferenceMetric].
 */
class AgentGoalAccuracyWithReferenceMetric(
    name: String = "agent_goal_accuracy_with_reference",
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.MULTI_TURN to setOf("user_input", "reference")),
        outputType = MetricOutputType.BINARY,
    ),
    MultiTurnMetric {
    /**
     * Executes multiTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun multiTurnAscore(sample: MultiTurnSample): Any {
        val desiredOutcome = sample.reference.orEmpty().trim()
        if (desiredOutcome.isBlank()) {
            return 0.0
        }
        val endState = inferEndState(sample.userInput)
        return if (isGoalAchieved(desiredOutcome, endState)) 1.0 else 0.0
    }
}

/**
 * Implements [AgentGoalAccuracyWithoutReferenceMetric].
 */
class AgentGoalAccuracyWithoutReferenceMetric(
    name: String = "agent_goal_accuracy_without_reference",
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.MULTI_TURN to setOf("user_input")),
        outputType = MetricOutputType.BINARY,
    ),
    MultiTurnMetric {
    /**
     * Executes multiTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun multiTurnAscore(sample: MultiTurnSample): Any {
        val desiredOutcome = inferDesiredOutcome(sample.userInput)
        if (desiredOutcome.isBlank()) {
            return 0.0
        }
        val endState = inferEndState(sample.userInput)
        return if (isGoalAchieved(desiredOutcome, endState)) 1.0 else 0.0
    }
}

/**
 * Implements [AgentWorkflowCompletionMetric].
 */
class AgentWorkflowCompletionMetric(
    name: String = "agent_workflow_completion",
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.MULTI_TURN to setOf("user_input")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    MultiTurnMetric {
    /**
     * Executes multiTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun multiTurnAscore(sample: MultiTurnSample): Any {
        val messages = sample.userInput
        if (messages.isEmpty()) {
            return 0.0
        }

        val aiMessages = messages.filterIsInstance<AiMessage>()
        val humanMessages = messages.filterIsInstance<HumanMessage>()
        val toolMessages = messages.filterIsInstance<ToolMessage>()
        val predictedToolCallCount = aiMessages.sumOf { ai -> ai.toolCalls.orEmpty().size }
        val finalAiMessage = messages.lastOrNull() as? AiMessage
        val finalAiPresent = finalAiMessage?.content?.isNotBlank() == true
        val refusalPenalty = if (isFailureOrRefusal(finalAiMessage?.content.orEmpty())) 0.2 else 1.0

        val score =
            if (predictedToolCallCount == 0) {
                when {
                    finalAiPresent -> 1.0 * refusalPenalty
                    aiMessages.isNotEmpty() -> 0.7 * refusalPenalty
                    else -> 0.0
                }
            } else {
                val toolExecutionCoverage =
                    minOf(toolMessages.size.toDouble() / predictedToolCallCount.toDouble(), 1.0)
                val finalResponseScore = if (finalAiPresent) 1.0 else 0.0
                val turnBalanceScore =
                    if (humanMessages.isEmpty()) {
                        0.0
                    } else {
                        minOf(aiMessages.size.toDouble() / humanMessages.size.toDouble(), 1.0)
                    }
                ((0.6 * toolExecutionCoverage) + (0.3 * finalResponseScore) + (0.1 * turnBalanceScore)) * refusalPenalty
            }

        return clamp01(score)
    }
}

private fun inferDesiredOutcome(messages: List<ConversationMessage>): String {
    val humanContents =
        messages
            .filterIsInstance<HumanMessage>()
            .map { msg -> msg.content.trim() }
            .filter { text -> text.isNotBlank() }
    if (humanContents.isEmpty()) {
        return ""
    }

    val actionableMessage =
        humanContents.firstOrNull { text ->
            ACKNOWLEDGEMENT_PATTERNS.none { regex -> regex.containsMatchIn(text) }
        }

    return actionableMessage ?: humanContents.first()
}

private fun inferEndState(messages: List<ConversationMessage>): String {
    val lastAi =
        messages
            .asReversed()
            .filterIsInstance<AiMessage>()
            .firstOrNull { it.content.isNotBlank() }
            ?.content
            .orEmpty()
            .trim()
    if (lastAi.isNotBlank()) {
        return lastAi
    }
    return messages
        .asReversed()
        .filterIsInstance<ToolMessage>()
        .firstOrNull { it.content.isNotBlank() }
        ?.content
        .orEmpty()
        .trim()
}

private fun isGoalAchieved(
    desiredOutcome: String,
    arrivedOutcome: String,
): Boolean {
    if (desiredOutcome.isBlank() || arrivedOutcome.isBlank()) {
        return false
    }
    if (isFailureOrRefusal(arrivedOutcome)) {
        return false
    }

    val desiredTokens = normalizeGoalTokens(desiredOutcome)
    val arrivedTokens = normalizeGoalTokens(arrivedOutcome)
    if (desiredTokens.isEmpty() || arrivedTokens.isEmpty()) {
        return false
    }

    val overlap = desiredTokens.intersect(arrivedTokens)
    if (overlap.isEmpty()) {
        return false
    }

    val recall = overlap.size.toDouble() / desiredTokens.size.toDouble()
    val precision = overlap.size.toDouble() / arrivedTokens.size.toDouble()
    val jaccard = overlap.size.toDouble() / desiredTokens.union(arrivedTokens).size.toDouble()

    return recall >= MIN_RECALL_THRESHOLD ||
        (recall >= MIN_COMBINED_THRESHOLD && precision >= MIN_COMBINED_THRESHOLD) ||
        jaccard >= MIN_JACCARD_THRESHOLD
}

private const val MIN_RECALL_THRESHOLD = 0.5
private const val MIN_COMBINED_THRESHOLD = 0.4
private const val MIN_JACCARD_THRESHOLD = 0.33

private fun normalizeGoalTokens(text: String): Set<String> =
    tokenize(text)
        .map { token -> token.lowercase() }
        .filter { token -> token.length >= 3 && token !in GOAL_STOP_WORDS }
        .toSet()

private fun isFailureOrRefusal(text: String): Boolean {
    val normalized = text.lowercase()
    return FAILURE_OR_REFUSAL_PATTERNS.any { pattern -> pattern.containsMatchIn(normalized) }
}

private val ACKNOWLEDGEMENT_PATTERNS =
    listOf(
        Regex("^thanks[.! ]*$", RegexOption.IGNORE_CASE),
        Regex("^thank\\s+you[.! ]*$", RegexOption.IGNORE_CASE),
        Regex("^ok(?:ay)?[.! ]*$", RegexOption.IGNORE_CASE),
        Regex("^great[.! ]*$", RegexOption.IGNORE_CASE),
        Regex("^sounds\\s+good[.! ]*$", RegexOption.IGNORE_CASE),
    )

private val FAILURE_OR_REFUSAL_PATTERNS =
    listOf(
        Regex("\\b(can'?t|cannot|won'?t|unable|refuse|decline|sorry)\\b"),
        Regex("\\b(failed|failure|error|unavailable|not\\s+found|did\\s+not|could\\s+not|couldn't)\\b"),
    )

private val GOAL_STOP_WORDS =
    setOf(
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "you",
        "your",
        "from",
        "into",
        "about",
        "please",
        "want",
        "need",
        "have",
        "has",
        "had",
        "would",
        "could",
        "should",
        "will",
        "shall",
        "just",
        "then",
        "than",
        "also",
    )
