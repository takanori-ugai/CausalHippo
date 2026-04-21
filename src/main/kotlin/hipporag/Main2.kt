package hipporag

import hipporag.evaluation.QAExactMatch
import hipporag.evaluation.QAF1Score

/**
 * CLI entry point for running and evaluating QA over Musique JSONL samples.
 */
fun main(args: Array<String>) {
    val parsed = MusiqueArgs.parse(args)
    val samples = readMusiqueSamples(parsed.inputPath, parsed.limit)

    if (samples.isEmpty()) {
        println("No samples found.")
        return
    }

    val predictions = mutableListOf<String>()
    val goldAnswers = mutableListOf<List<String>>()

    samples.forEachIndexed { index, sample ->
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
            println("Skipping sample ${sample.id}: no non-blank passages.")
            predictions.add("")
            val gold =
                buildList {
                    add(sample.answer)
                    addAll(sample.answerAliases)
                }.distinct()
            goldAnswers.add(gold)
            return@forEachIndexed
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
        predictions.add(answer)

        val gold =
            buildList {
                add(sample.answer)
                addAll(sample.answerAliases)
            }.distinct()
        goldAnswers.add(gold)
        println("Answer: $answer")
        println("Gold: $gold")

        if ((index + 1) % 10 == 0 || index == samples.lastIndex) {
            println("Processed ${index + 1}/${samples.size} samples.")
        }
    }

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
