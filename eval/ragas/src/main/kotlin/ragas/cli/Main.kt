package ragas.cli

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import ragas.backends.BACKEND_REGISTRY
import ragas.backends.anyToJsonElement
import ragas.backends.jsonElementToAny
import ragas.defaultMetrics
import ragas.evaluate
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample
import ragas.tier1Metrics
import ragas.tier2Metrics
import ragas.tier3Metrics
import ragas.tier4Metrics
import java.io.File
import kotlin.system.exitProcess

private val jsonPretty = Json { prettyPrint = true }

/**
 * Parsed CLI options split into key/value options and boolean flags.
 *
 * @property values Option key/value map.
 * @property flags Boolean option flags set.
 */
data class CliOptions(
    val values: Map<String, String>,
    val flags: Set<String>,
)

/**
 * CLI entry point that executes a command and exits with its status code.
 *
 * @param args Command-line arguments.
 */
fun main(args: Array<String>) {
    val code = runCli(args)
    if (code != 0) {
        exitProcess(code)
    }
}

/**
 * Parses CLI arguments, runs the selected command, and returns an exit code.
 *
 * @param args Command-line arguments.
 * @param out Output stream used for standard CLI messages.
 * @param err Output stream used for CLI error messages.
 */
fun runCli(
    args: Array<String>,
    out: (String) -> Unit = ::println,
    err: (String) -> Unit = { message -> System.err.println(message) },
): Int {
    val command = args.firstOrNull() ?: "help"
    return runCatching {
        when (command) {
            "help", "--help", "-h" -> {
                printHelp(out)
                0
            }

            "status" -> {
                printStatus(out)
                0
            }

            "backends" -> {
                printBackends(out)
                0
            }

            "eval" -> {
                val options = parseOptions(args.drop(1))
                runEval(options, out)
            }

            "report" -> {
                val options = parseOptions(args.drop(1))
                runReport(options, out)
            }

            "compare" -> {
                val options = parseOptions(args.drop(1))
                runCompare(options, out)
            }

            else -> {
                err("Unknown command: $command")
                printHelp(err)
                1
            }
        }
    }.getOrElse { throwable ->
        err(throwable.message ?: throwable::class.simpleName.orEmpty())
        1
    }
}

private fun parseOptions(tokens: List<String>): CliOptions {
    val values = mutableMapOf<String, String>()
    val flags = mutableSetOf<String>()
    var index = 0
    while (index < tokens.size) {
        val token = tokens[index]
        require(token.startsWith("--")) { "Unexpected argument '$token'. Expected --key value pairs." }
        val key = token.removePrefix("--")
        val next = tokens.getOrNull(index + 1)
        if (next == null || next.startsWith("--")) {
            flags += key
            index += 1
            continue
        }
        values[key] = next
        index += 2
    }
    return CliOptions(values = values, flags = flags)
}

private fun runEval(
    options: CliOptions,
    out: (String) -> Unit,
): Int {
    val inputPath = options.values["input"] ?: error("--input is required for eval")
    val outputPath = options.values["output"]
    val format = options.values["format"] ?: detectFormat(inputPath)
    val metricSpec = options.values["metrics"] ?: "default"

    val rows = readRows(inputPath, format)
    val dataset = EvaluationDataset(rows.map(::rowToSingleTurnSample))
    val metrics = resolveMetrics(metricSpec)
    val result = evaluate(dataset = dataset, metrics = metrics)

    val metricNames = result.scores.flatMap { row -> row.keys }.toSortedSet()
    val means = metricNames.associateWith { metric -> result.metricMean(metric) }

    val payload =
        mapOf(
            "row_count" to rows.size,
            "metric_names" to metricNames.toList(),
            "metric_means" to means,
            "scores" to result.scores,
            "dataset_features" to dataset.features().toList().sorted(),
        )

    val encoded = encodeJsonObject(payload)
    if (outputPath == null) {
        out(encoded)
    } else {
        File(outputPath).writeText(encoded)
        out("wrote evaluation report: $outputPath")
    }
    return 0
}

private fun runReport(
    options: CliOptions,
    out: (String) -> Unit,
): Int {
    val inputPath = options.values["input"] ?: error("--input is required for report")
    val outputPath = options.values["output"]
    val scores = extractScores(inputPath)

    val metricNames = scores.flatMap { row -> row.keys }.toSortedSet()
    val means =
        metricNames.associateWith { metric ->
            val values = scores.mapNotNull { row -> row[metric] as? Number }.map { it.toDouble() }
            if (values.isEmpty()) null else values.average()
        }
    val counts =
        metricNames.associateWith { metric ->
            scores.count { row -> row[metric] != null }
        }

    val summary =
        mapOf(
            "row_count" to scores.size,
            "metric_names" to metricNames.toList(),
            "metric_means" to means,
            "metric_non_null_counts" to counts,
        )

    val encoded = encodeJsonObject(summary)
    if (outputPath == null) {
        out(encoded)
    } else {
        File(outputPath).writeText(encoded)
        out("wrote report summary: $outputPath")
    }
    return 0
}

private fun runCompare(
    options: CliOptions,
    out: (String) -> Unit,
): Int {
    val baselinePath = options.values["baseline"] ?: error("--baseline is required for compare")
    val candidatePath = options.values["candidate"] ?: error("--candidate is required for compare")
    val outputPath = options.values["output"]
    val selectedMetrics =
        options.values["metrics"]
            ?.split(',')
            ?.map { token -> token.trim() }
            ?.filter { token -> token.isNotEmpty() }
    val gates = parseThresholdMap(options.values["gate"])

    val baselineMeans = computeMetricMeans(extractScores(baselinePath))
    val candidateMeans = computeMetricMeans(extractScores(candidatePath))

    val metricsToCompare =
        if (selectedMetrics.isNullOrEmpty()) {
            (baselineMeans.keys intersect candidateMeans.keys).toSortedSet().toList()
        } else {
            selectedMetrics
        }
    if (!selectedMetrics.isNullOrEmpty()) {
        val missingInBaseline = selectedMetrics.filter { metric -> baselineMeans[metric] == null }
        val missingInCandidate = selectedMetrics.filter { metric -> candidateMeans[metric] == null }
        require(missingInBaseline.isEmpty() && missingInCandidate.isEmpty()) {
            buildString {
                append("Requested metrics are unavailable for comparison.")
                if (missingInBaseline.isNotEmpty()) {
                    append(" Missing in baseline: ${missingInBaseline.distinct().sorted().joinToString(", ")}.")
                }
                if (missingInCandidate.isNotEmpty()) {
                    append(" Missing in candidate: ${missingInCandidate.distinct().sorted().joinToString(", ")}.")
                }
            }
        }
    }

    val deltas =
        metricsToCompare.associateWith { metric ->
            val baseline = baselineMeans[metric]
            val candidate = candidateMeans[metric]
            if (baseline == null || candidate == null) {
                null
            } else {
                candidate - baseline
            }
        }

    val failedGates =
        gates.entries.mapNotNull { (metric, threshold) ->
            val delta = deltas[metric]
            if (delta == null || delta < threshold) {
                mapOf(
                    "metric" to metric,
                    "required_delta" to threshold,
                    "actual_delta" to delta,
                    "reason" to if (delta == null) "metric_missing" else "below_threshold",
                )
            } else {
                null
            }
        }

    val status = if (failedGates.isEmpty()) "pass" else "fail"
    val payload =
        mapOf(
            "status" to status,
            "baseline_means" to baselineMeans,
            "candidate_means" to candidateMeans,
            "compared_metrics" to metricsToCompare,
            "deltas" to deltas,
            "gate" to mapOf("thresholds" to gates, "failed" to failedGates),
        )

    val encoded = encodeJsonObject(payload)
    if (outputPath == null) {
        out(encoded)
    } else {
        File(outputPath).writeText(encoded)
        out("wrote comparison report: $outputPath")
    }

    return if (status == "pass") 0 else 2
}

private fun detectFormat(path: String): String = if (path.endsWith(".jsonl")) "jsonl" else "json"

private fun readRows(
    inputPath: String,
    format: String,
): List<Map<String, Any?>> {
    val file = File(inputPath)
    require(file.exists()) { "Input file not found: $inputPath" }
    return when (format) {
        "json" -> {
            val parsed = Json.parseToJsonElement(file.readText())
            when (parsed) {
                is JsonArray -> {
                    parsed.map { element -> jsonElementToMap(element) }
                }

                is JsonObject -> {
                    val rowsElement = parsed["rows"] ?: parsed["samples"] ?: error("JSON input object must contain 'rows' or 'samples'.")
                    require(rowsElement is JsonArray) { "'rows'/'samples' must be an array." }
                    rowsElement.map { element -> jsonElementToMap(element) }
                }

                else -> {
                    error("JSON input must be an array of row objects or object with rows/samples array.")
                }
            }
        }

        "jsonl" -> {
            file.useLines { lines ->
                lines
                    .map { line -> line.trim() }
                    .filter { line -> line.isNotEmpty() }
                    .map { line -> jsonElementToMap(Json.parseToJsonElement(line)) }
                    .toList()
            }
        }

        else -> {
            error("Unsupported format '$format'. Use json or jsonl.")
        }
    }
}

private fun jsonElementToMap(element: JsonElement): Map<String, Any?> {
    require(element is JsonObject) { "Each row must be a JSON object." }
    return element.mapValues { (_, value) -> jsonElementToAny(value) }
}

private fun rowToSingleTurnSample(row: Map<String, Any?>): SingleTurnSample =
    SingleTurnSample(
        userInput = row["user_input"] as? String,
        retrievedContexts = anyToStringList(row["retrieved_contexts"]),
        referenceContexts = anyToStringList(row["reference_contexts"]),
        retrievedContextIds = anyToStringList(row["retrieved_context_ids"]),
        referenceContextIds = anyToStringList(row["reference_context_ids"]),
        response = row["response"] as? String,
        multiResponses = anyToStringList(row["multi_responses"]),
        reference = row["reference"] as? String,
        rubrics = anyToStringMap(row["rubrics"]),
        personaName = row["persona_name"] as? String,
        queryStyle = row["query_style"] as? String,
        queryLength = row["query_length"] as? String,
    )

private fun anyToStringList(value: Any?): List<String>? {
    val list = value as? List<*> ?: return null
    return list.map { item -> item?.toString().orEmpty() }
}

private fun anyToStringMap(value: Any?): Map<String, String>? {
    val map = value as? Map<*, *> ?: return null
    return map.entries.associate { entry -> entry.key.toString() to (entry.value?.toString() ?: "") }
}

private fun resolveMetrics(spec: String): List<ragas.metrics.Metric> {
    val allMetrics =
        (defaultMetrics() + tier1Metrics() + tier2Metrics() + tier3Metrics() + tier4Metrics())
            .associateBy { metric -> metric.name }
    val resolved = mutableListOf<ragas.metrics.Metric>()

    spec
        .split(',')
        .map { token -> token.trim() }
        .filter { token -> token.isNotEmpty() }
        .forEach { token ->
            when (token) {
                "default" -> {
                    resolved += defaultMetrics()
                }

                "tier1" -> {
                    resolved += tier1Metrics()
                }

                "tier2" -> {
                    resolved += tier2Metrics()
                }

                "tier3" -> {
                    resolved += tier3Metrics()
                }

                "tier4" -> {
                    resolved += tier4Metrics()
                }

                "all" -> {
                    resolved += allMetrics.values
                }

                else -> {
                    val metric = allMetrics[token] ?: error("Unknown metric '$token'.")
                    resolved += metric
                }
            }
        }

    val deduped = linkedMapOf<String, ragas.metrics.Metric>()
    resolved.forEach { metric -> deduped[metric.name] = metric }
    return if (deduped.isEmpty()) defaultMetrics() else deduped.values.toList()
}

private fun extractScores(path: String): List<Map<String, Any?>> {
    val parsed = Json.parseToJsonElement(File(path).readText())
    return when (parsed) {
        is JsonArray -> {
            parsed.map { element -> jsonElementToMap(element) }
        }

        is JsonObject -> {
            val scores = parsed["scores"] ?: error("Expected a 'scores' array in report file: $path")
            require(scores is JsonArray) { "'scores' must be an array in report file: $path" }
            scores.map { element -> jsonElementToMap(element) }
        }

        else -> {
            error("Invalid report file format: $path")
        }
    }
}

private fun computeMetricMeans(scores: List<Map<String, Any?>>): Map<String, Double?> {
    val metricNames = scores.flatMap { row -> row.keys }.toSortedSet()
    return metricNames.associateWith { metric ->
        val values = scores.mapNotNull { row -> row[metric] as? Number }.map { it.toDouble() }
        if (values.isEmpty()) null else values.average()
    }
}

private fun parseThresholdMap(raw: String?): Map<String, Double> {
    if (raw.isNullOrBlank()) {
        return emptyMap()
    }
    return raw
        .split(',')
        .map { token -> token.trim() }
        .filter { token -> token.isNotEmpty() }
        .associate { token ->
            val split = token.split('=', limit = 2)
            require(split.size == 2) { "Gate token must be metric=threshold, got '$token'." }
            val metric = split[0].trim()
            require(metric.isNotEmpty()) { "Gate metric name cannot be empty in token '$token'." }
            val threshold = split[1].trim().toDoubleOrNull() ?: error("Invalid threshold in gate token '$token'.")
            metric to threshold
        }
}

private fun encodeJsonObject(payload: Map<String, Any?>): String {
    val objectElement = JsonObject(payload.mapValues { (_, value) -> anyToJsonElement(value) })
    return jsonPretty.encodeToString(JsonElement.serializer(), objectElement)
}

private fun printStatus(out: (String) -> Unit) {
    out("ragas-kotlin status")
    out("- core evaluation: available")
    out("- default metrics: available")
    out("- testset scaffold: available")
    out("- integrations: scaffold only")
}

private fun printBackends(out: (String) -> Unit) {
    out("available backends:")
    BACKEND_REGISTRY.listBackendInfo().forEach { info ->
        val aliasSuffix =
            if (info.aliases.isEmpty()) {
                ""
            } else {
                " (aliases: ${info.aliases.joinToString(", ")})"
            }
        out("- ${info.name}$aliasSuffix [${info.source}]")
    }
}

private fun printHelp(out: (String) -> Unit) {
    out("ragas-kotlin CLI")
    out("usage: ragas <command> [options]")
    out("commands:")
    out("  status    Show conversion/runtime status")
    out("  backends  List registered backends")
    out("  eval      Run evaluation from JSON/JSONL dataset and emit report JSON")
    out("  report    Aggregate metrics from a report JSON")
    out("  compare   Compare candidate vs baseline reports with optional gate thresholds")
    out("examples:")
    out("  ragas eval --input dataset.json --metrics default,tier1 --output run.json")
    out("  ragas report --input run.json")
    out("  ragas compare --baseline run_a.json --candidate run_b.json --gate faithfulness=0.01")
}
