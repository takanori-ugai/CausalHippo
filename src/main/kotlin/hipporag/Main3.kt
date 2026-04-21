package hipporag

import hipporag.evaluation.QAExactMatch
import hipporag.evaluation.QAF1Score
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import java.util.concurrent.atomic.AtomicInteger

/**
 * CLI entry point for running and evaluating QA over Musique JSONL samples.
 */
fun main(args: Array<String>) {
    runBlocking {
        val parsed = MusiqueArgs.parse(args)
        val samples = readMusiqueSamples(parsed.inputPath, parsed.limit)

        if (samples.isEmpty()) {
            println("No samples found.")
            return@runBlocking
        }

        val parallelism = parsed.parallelism ?: 4
        val semaphore = Semaphore(parallelism)
        val progress = AtomicInteger(0)
        val results =
            samples
                .mapIndexed { index, sample ->
                    async(Dispatchers.IO) {
                        semaphore.withPermit {
                            val config = buildMusiqueConfig(parsed, sample.id)
                            val hipporag = HippoRag(config = config)
                            val docs =
                                sample.paragraphs
                                    .map { paragraph ->
                                        val title = paragraph.title?.trim().orEmpty()
                                        if (title.isNotEmpty()) {
                                            "$title\n${paragraph.paragraphText}"
                                        } else {
                                            paragraph.paragraphText
                                        }
                                    }.filter { it.isNotBlank() }

                            if (docs.isEmpty()) {
                                val gold =
                                    buildList {
                                        add(sample.answer)
                                        addAll(sample.answerAliases)
                                    }.distinct()
                                val processed = progress.incrementAndGet()
                                println("Skipping sample ${sample.id}: no non-blank passages.")
                                if (processed % 10 == 0 || processed == samples.size) {
                                    println("Processed $processed/${samples.size} samples.")
                                }
                                return@withPermit ResultRow(index, "", gold)
                            }

                            hipporag.index(docs)
                            val result =
                                hipporag.ragQa(
                                    queries = listOf(sample.question),
                                    goldDocs = null,
                                    goldAnswers = null,
                                )
                            val answer =
                                result.solutions
                                    .firstOrNull()
                                    ?.answer
                                    .orEmpty()

                            val gold =
                                buildList {
                                    add(sample.answer)
                                    addAll(sample.answerAliases)
                                }.distinct()

                            val processed = progress.incrementAndGet()
                            println("Answer: $answer")
                            println("Gold: $gold")
                            if (processed % 10 == 0 || processed == samples.size) {
                                println("Processed $processed/${samples.size} samples.")
                            }

                            ResultRow(index, answer, gold)
                        }
                    }
                }.awaitAll()

        val orderedResults = results.sortedBy { it.index }
        val predictions = orderedResults.map { it.answer }
        val goldAnswers = orderedResults.map { it.goldAnswers }

        val emEvaluator = QAExactMatch()
        val f1Evaluator = QAF1Score()
        val (overallEm, _) =
            emEvaluator.calculateMetricScores(
                goldAnswers = goldAnswers,
                predictedAnswers = predictions,
                aggregationFn = { values -> values.maxOrNull() ?: 0.0 },
            )
        val (overallF1, _) =
            f1Evaluator.calculateMetricScores(
                goldAnswers = goldAnswers,
                predictedAnswers = predictions,
                aggregationFn = { values -> values.maxOrNull() ?: 0.0 },
            )

        println("=== Evaluation ===")
        println("ExactMatch: ${"%.4f".format(overallEm.getValue("ExactMatch"))}")
        println("F1: ${"%.4f".format(overallF1.getValue("F1"))}")
    }
}

private data class ResultRow(
    val index: Int,
    val answer: String,
    val goldAnswers: List<String>,
)
