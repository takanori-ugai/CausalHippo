package ragas.examples.rageval

import kotlinx.coroutines.runBlocking
import ragas.backends.LocalCsvBackend
import ragas.backends.rowToJsonLine
import ragas.experiment
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.metrics.MetricType
import ragas.metrics.primitives.DiscreteMetric
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import java.io.File

private data class EvalRow(
    val question: String,
    val gradingNotes: String,
)

private class RuleBasedEvalLlm : BaseRagasLlm {
    override var runConfig: RunConfig = RunConfig()

    /**
     * Executes generateText.
     */
    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        val lower = prompt.lowercase()
        val value =
            if (
                "experimentation" in lower ||
                "metrics" in lower ||
                "experiments/" in lower
            ) {
                "pass"
            } else {
                "fail"
            }
        return LlmResult(generations = listOf(LlmGeneration(value)))
    }
}

private fun loadDataset(rootDir: String = "evals"): List<EvalRow> {
    val datasetRows =
        listOf(
            EvalRow(
                question = "What is ragas 0.3",
                gradingNotes =
                    "- experimentation as the central pillar - provides abstraction for datasets, " +
                        "experiments and metrics - supports evals for RAG, LLM workflows and Agents",
            ),
            EvalRow(
                question = "how are experiment results stored in ragas 0.3?",
                gradingNotes =
                    "- configured using different backends like local, gdrive, etc - stored under " +
                        "experiments/ folder in the backend storage",
            ),
            EvalRow(
                question = "What metrics are supported in ragas 0.3?",
                gradingNotes = "- provides abstraction for discrete, numerical and ranking metrics",
            ),
        )

    val backend = LocalCsvBackend(rootDir)
    backend.saveDataset(
        name = "test_dataset",
        data =
            datasetRows.map { row ->
                mapOf(
                    "question" to row.question,
                    "grading_notes" to row.gradingNotes,
                )
            },
    )
    return datasetRows
}

/**
 * Executes main.
 */
fun main() =
    runBlocking {
        val llm = RuleBasedEvalLlm()
        val ragClient = defaultRagClient(chatClient = EchoChatClient(), logDir = "evals/logs")

        val myMetric =
            DiscreteMetric(
                name = "correctness",
                prompt =
                    "Check if the response covers the grading notes and return pass or fail. " +
                        "Response: {response} Grading Notes: {reference}",
                llm = llm,
                allowedValues = listOf("pass", "fail"),
                requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response", "reference")),
            )

        val dataset = loadDataset(rootDir = "evals")
        println("dataset loaded successfully: size=${dataset.size}")

        val runner =
            experiment<EvalRow>(backend = LocalCsvBackend("evals"), namePrefix = "rag-eval") { row ->
                val response = ragClient.query(row.question)
                val score =
                    myMetric.singleTurnAscore(
                        SingleTurnSample(
                            userInput = row.question,
                            response = response.answer,
                            reference = row.gradingNotes,
                        ),
                    )
                mapOf(
                    "question" to row.question,
                    "grading_notes" to row.gradingNotes,
                    "response" to response.answer,
                    "score" to score,
                    "log_file" to response.logs,
                )
            }

        val experimentResults = runner.arun(dataset = dataset, name = "baseline")
        println("Experiment completed successfully!")
        println("Experiment name: ${experimentResults.name}")
        val csvPath = File("evals/experiments/${experimentResults.name}.csv")
        println("Experiment results saved to: ${csvPath.absolutePath}")

        val debugJsonlPath = File("evals/experiments/${experimentResults.name}.jsonl")
        val stored = experimentResults.rows()
        debugJsonlPath.parentFile.mkdirs()
        debugJsonlPath.writeText(stored.joinToString("\n") { row -> rowToJsonLine(row) } + "\n")
    }
