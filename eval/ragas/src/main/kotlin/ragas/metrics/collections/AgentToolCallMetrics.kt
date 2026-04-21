package ragas.metrics.collections

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.MultiTurnMetric
import ragas.metrics.clamp01
import ragas.model.AiMessage
import ragas.model.MultiTurnSample
import ragas.model.ToolCall
import kotlin.math.round

/**
 * Implements [ToolCallAccuracyMetric].
 *
 * @property strictOrder Whether tool-call order must match exactly.
 */
class ToolCallAccuracyMetric(
    private val strictOrder: Boolean = true,
    name: String = "tool_call_accuracy",
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.MULTI_TURN to setOf("user_input", "reference_tool_calls")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    MultiTurnMetric {
    /**
     * Executes multiTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun multiTurnAscore(sample: MultiTurnSample): Any {
        val predictedToolCalls = extractPredictedToolCalls(sample)
        val referenceToolCalls = sample.referenceToolCalls.orEmpty()

        if (predictedToolCalls.isEmpty() && referenceToolCalls.isEmpty()) {
            return 1.0
        }
        if (predictedToolCalls.isEmpty() || referenceToolCalls.isEmpty()) {
            return 0.0
        }

        val predicted = if (strictOrder) predictedToolCalls else predictedToolCalls.sortedBy(::toolCallSortKey)
        val reference = if (strictOrder) referenceToolCalls else referenceToolCalls.sortedBy(::toolCallSortKey)

        val predictedSequence = predicted.map { it.name }
        val referenceSequence = reference.map { it.name }
        if (!isSequenceAligned(predictedSequence, referenceSequence)) {
            return 0.0
        }

        var score = 0.0

        reference.zip(predicted).forEach { (referenceToolCall, predictedToolCall) ->
            if (referenceToolCall.name == predictedToolCall.name) {
                score += argumentScore(predictedToolCall.args, referenceToolCall.args)
            }
        }

        score /= reference.size.toDouble()

        return clamp01(score)
    }

    private fun extractPredictedToolCalls(sample: MultiTurnSample): List<ToolCall> =
        sample.userInput
            .filterIsInstance<AiMessage>()
            .flatMap { it.toolCalls.orEmpty() }

    private fun argumentScore(
        predictedArgs: Map<String, JsonElement>,
        referenceArgs: Map<String, JsonElement>,
    ): Double {
        if (referenceArgs.isEmpty() && predictedArgs.isEmpty()) {
            return 1.0
        }
        if (referenceArgs.isEmpty()) {
            return 0.0
        }

        var matched = 0.0
        referenceArgs.forEach { (argName, referenceValue) ->
            val predictedValue = predictedArgs[argName]
            if (predictedValue != null && predictedValue == referenceValue) {
                matched += 1.0
            }
        }
        return matched / referenceArgs.size.toDouble()
    }

    private fun toolCallSortKey(toolCall: ToolCall): String {
        val sortedArgPairs =
            toolCall.args
                .toList()
                .sortedBy { (name, _) -> name }
                .joinToString("|") { (name, value) -> "$name=${canonicalizeJson(value)}" }
        return "${toolCall.name}|$sortedArgPairs"
    }

    private fun canonicalizeJson(value: JsonElement): String =
        when (value) {
            is JsonObject -> {
                value.entries
                    .sortedBy { (name, _) -> name }
                    .joinToString(prefix = "{", postfix = "}", separator = ",") { (name, child) ->
                        "\"$name\":${canonicalizeJson(child)}"
                    }
            }

            is JsonArray -> {
                value.joinToString(prefix = "[", postfix = "]", separator = ",") { child ->
                    canonicalizeJson(child)
                }
            }

            is JsonPrimitive -> {
                value.toString()
            }
        }

    private fun isSequenceAligned(
        predictedSequence: List<String>,
        referenceSequence: List<String>,
    ): Boolean =
        if (strictOrder) {
            predictedSequence == referenceSequence
        } else {
            predictedSequence.sorted() == referenceSequence.sorted()
        }
}

/**
 * Implements [ToolCallF1Metric].
 */
class ToolCallF1Metric(
    name: String = "tool_call_f1",
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.MULTI_TURN to setOf("user_input", "reference_tool_calls")),
        outputType = MetricOutputType.CONTINUOUS,
    ),
    MultiTurnMetric {
    /**
     * Executes multiTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun multiTurnAscore(sample: MultiTurnSample): Any {
        val expected =
            sample.referenceToolCalls
                .orEmpty()
                .map(::toHashableToolCall)
                .toSet()
        val actual =
            sample.userInput
                .filterIsInstance<AiMessage>()
                .flatMap { it.toolCalls.orEmpty() }
                .map(::toHashableToolCall)
                .toSet()

        val truePositives = (actual intersect expected).size
        val falsePositives = (actual - expected).size
        val falseNegatives = (expected - actual).size

        val precision =
            if (truePositives + falsePositives > 0) {
                truePositives.toDouble() / (truePositives + falsePositives).toDouble()
            } else {
                0.0
            }
        val recall =
            if (truePositives + falseNegatives > 0) {
                truePositives.toDouble() / (truePositives + falseNegatives).toDouble()
            } else {
                0.0
            }

        val f1 =
            if (precision + recall > 0.0) {
                (2 * precision * recall) / (precision + recall)
            } else {
                // Keep empty/empty at 0.0 for parity with the Python implementation.
                0.0
            }

        return round(f1 * 10000.0) / 10000.0
    }

    private fun toHashableToolCall(toolCall: ToolCall): HashableToolCall =
        HashableToolCall(
            name = toolCall.name,
            args = toHashableValue(JsonObject(toolCall.args)),
        )

    private fun toHashableValue(value: JsonElement): HashableValue =
        when (value) {
            is JsonObject -> {
                HashableValue.ObjectValue(
                    value.entries
                        .map { (key, child) -> key to toHashableValue(child) }
                        .sortedBy { (key, _) -> key },
                )
            }

            is JsonArray -> {
                HashableValue.ArrayValue(value.map(::toHashableValue))
            }

            is JsonPrimitive -> {
                HashableValue.PrimitiveValue(value.toString())
            }
        }

    private sealed interface HashableValue {
        data class ObjectValue(
            val entries: List<Pair<String, HashableValue>>,
        ) : HashableValue

        data class ArrayValue(
            val entries: List<HashableValue>,
        ) : HashableValue

        data class PrimitiveValue(
            val value: String,
        ) : HashableValue
    }

    private data class HashableToolCall(
        val name: String,
        val args: HashableValue,
    )
}
