package ragas.cli

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CliParityTest {
    @Test
    fun evalCommandProducesReportJson() {
        withTempDir("ragas-cli-eval") { dir ->
            val dataset = dir.resolve("dataset.json")
            val report = dir.resolve("report.json")

            dataset.writeText(
                """
                [
                  {
                    "user_input": "What is Kotlin?",
                    "response": "Kotlin is a JVM language.",
                    "retrieved_contexts": ["Kotlin is a statically typed language for JVM and Android."],
                    "reference_contexts": ["Kotlin targets JVM and Android."],
                    "reference": "Kotlin is a language for JVM and Android."
                  }
                ]
                """.trimIndent(),
            )

            val code = runCli(arrayOf("eval", "--input", dataset.absolutePath, "--output", report.absolutePath, "--metrics", "default"))

            assertEquals(0, code)
            assertTrue(report.exists())

            val parsed = Json.parseToJsonElement(report.readText()).jsonObject
            assertEquals(
                1,
                parsed
                    .getValue("row_count")
                    .jsonPrimitive.content
                    .toInt(),
            )
            assertTrue(parsed.containsKey("scores"))
            assertTrue(parsed.containsKey("metric_means"))
        }
    }

    @Test
    fun reportCommandAggregatesScores() {
        withTempDir("ragas-cli-report") { dir ->
            val report = dir.resolve("input-report.json")
            val summary = dir.resolve("summary.json")

            report.writeText(
                """
                {
                  "scores": [
                    {"m1": 0.5, "m2": 0.2},
                    {"m1": 0.7, "m2": 0.4}
                  ]
                }
                """.trimIndent(),
            )

            val code = runCli(arrayOf("report", "--input", report.absolutePath, "--output", summary.absolutePath))

            assertEquals(0, code)
            val parsed = Json.parseToJsonElement(summary.readText()).jsonObject
            val means = parsed.getValue("metric_means").jsonObject
            assertEquals(
                0.6,
                means
                    .getValue("m1")
                    .jsonPrimitive.content
                    .toDouble(),
                1e-9,
            )
            assertEquals(
                0.3,
                means
                    .getValue("m2")
                    .jsonPrimitive.content
                    .toDouble(),
                1e-9,
            )
        }
    }

    @Test
    fun compareCommandReturnsNonZeroWhenGateFails() {
        withTempDir("ragas-cli-compare") { dir ->
            val baseline = dir.resolve("baseline.json")
            val candidate = dir.resolve("candidate.json")

            baseline.writeText("""{"scores":[{"faithfulness":0.5},{"faithfulness":0.7}]}""")
            candidate.writeText("""{"scores":[{"faithfulness":0.55},{"faithfulness":0.65}]}""")

            val failCode =
                runCli(
                    arrayOf(
                        "compare",
                        "--baseline",
                        baseline.absolutePath,
                        "--candidate",
                        candidate.absolutePath,
                        "--gate",
                        "faithfulness=0.10",
                    ),
                )
            assertEquals(2, failCode)

            val passCode =
                runCli(
                    arrayOf(
                        "compare",
                        "--baseline",
                        baseline.absolutePath,
                        "--candidate",
                        candidate.absolutePath,
                        "--gate",
                        "faithfulness=-0.10",
                    ),
                )
            assertEquals(0, passCode)
        }
    }

    @Test
    fun compareCommandIncludesGateFailureReason() {
        withTempDir("ragas-cli-compare-reason") { dir ->
            val baseline = dir.resolve("baseline.json")
            val candidate = dir.resolve("candidate.json")

            baseline.writeText("""{"scores":[{"faithfulness":0.5},{"faithfulness":0.7}]}""")
            candidate.writeText("""{"scores":[{"faithfulness":0.55},{"faithfulness":0.65}]}""")

            val belowThresholdOutput = mutableListOf<String>()
            val belowThresholdCode =
                runCli(
                    arrayOf(
                        "compare",
                        "--baseline",
                        baseline.absolutePath,
                        "--candidate",
                        candidate.absolutePath,
                        "--gate",
                        "faithfulness=0.10",
                    ),
                    out = { line -> belowThresholdOutput += line },
                )
            assertEquals(2, belowThresholdCode)
            val belowThresholdFailed =
                Json
                    .parseToJsonElement(belowThresholdOutput.single())
                    .jsonObject
                    .getValue("gate")
                    .jsonObject
                    .getValue("failed")
                    .jsonArray
            assertEquals(
                "below_threshold",
                belowThresholdFailed[0]
                    .jsonObject
                    .getValue("reason")
                    .jsonPrimitive.content,
            )

            val missingMetricOutput = mutableListOf<String>()
            val missingMetricCode =
                runCli(
                    arrayOf(
                        "compare",
                        "--baseline",
                        baseline.absolutePath,
                        "--candidate",
                        candidate.absolutePath,
                        "--gate",
                        "answer_relevancy=0.01",
                    ),
                    out = { line -> missingMetricOutput += line },
                )
            assertEquals(2, missingMetricCode)
            val missingMetricFailed =
                Json
                    .parseToJsonElement(missingMetricOutput.single())
                    .jsonObject
                    .getValue("gate")
                    .jsonObject
                    .getValue("failed")
                    .jsonArray
            assertEquals(
                "metric_missing",
                missingMetricFailed[0]
                    .jsonObject
                    .getValue("reason")
                    .jsonPrimitive.content,
            )
        }
    }

    @Test
    fun compareCommandFailsWhenSelectedMetricIsMissing() {
        withTempDir("ragas-cli-compare-metrics") { dir ->
            val baseline = dir.resolve("baseline.json")
            val candidate = dir.resolve("candidate.json")

            baseline.writeText("""{"scores":[{"faithfulness":0.5},{"faithfulness":0.7}]}""")
            candidate.writeText("""{"scores":[{"faithfulness":0.55},{"faithfulness":0.65}]}""")

            val errors = mutableListOf<String>()
            val code =
                runCli(
                    arrayOf(
                        "compare",
                        "--baseline",
                        baseline.absolutePath,
                        "--candidate",
                        candidate.absolutePath,
                        "--metrics",
                        "answer_relevancy",
                    ),
                    err = { message -> errors += message },
                )

            assertEquals(1, code)
            assertTrue(errors.single().contains("Requested metrics are unavailable for comparison."))
            assertTrue(errors.single().contains("Missing in baseline: answer_relevancy."))
            assertTrue(errors.single().contains("Missing in candidate: answer_relevancy."))
        }
    }

    @Test
    fun compareCommandRejectsEmptyGateMetricName() {
        withTempDir("ragas-cli-compare-gate-parse") { dir ->
            val baseline = dir.resolve("baseline.json")
            val candidate = dir.resolve("candidate.json")

            baseline.writeText("""{"scores":[{"faithfulness":0.5},{"faithfulness":0.7}]}""")
            candidate.writeText("""{"scores":[{"faithfulness":0.55},{"faithfulness":0.65}]}""")

            val errors = mutableListOf<String>()
            val code =
                runCli(
                    arrayOf(
                        "compare",
                        "--baseline",
                        baseline.absolutePath,
                        "--candidate",
                        candidate.absolutePath,
                        "--gate",
                        "=0.1",
                    ),
                    err = { message -> errors += message },
                )

            assertEquals(1, code)
            assertTrue(errors.single().contains("Gate metric name cannot be empty"))
        }
    }
}

private inline fun withTempDir(
    prefix: String,
    block: (java.io.File) -> Unit,
) {
    val dir = createTempDirectory(prefix).toFile()
    try {
        block(dir)
    } finally {
        dir.deleteRecursively()
    }
}
