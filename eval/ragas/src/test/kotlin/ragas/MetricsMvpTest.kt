package ragas

import kotlinx.coroutines.runBlocking
import ragas.evaluation.evaluate
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.llms.StructuredOutputRagasLlm
import ragas.metrics.primitives.DiscreteMetric
import ragas.metrics.primitives.NumericMetric
import ragas.metrics.primitives.RankingMetric
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class MetricsMvpTest {
    @Test
    fun evaluateUsesDefaultSingleTurnMetricsWhenNotProvided() {
        val dataset =
            EvaluationDataset(
                listOf(
                    SingleTurnSample(
                        userInput = "What is Kotlin?",
                        response = "Kotlin is a JVM language.",
                        retrievedContexts = listOf("Kotlin is a statically typed language for the JVM."),
                        reference = "Kotlin is a statically typed language for the JVM.",
                        referenceContexts = listOf("Kotlin targets the JVM and is statically typed."),
                    ),
                ),
            )

        val result = evaluate(dataset = dataset, metrics = null)
        val scoreRow = result.scores.first()

        assertTrue(scoreRow.containsKey("answer_relevancy"))
        assertTrue(scoreRow.containsKey("context_precision"))
        assertTrue(scoreRow.containsKey("faithfulness"))
        assertTrue(scoreRow.containsKey("context_recall"))
    }

    @Test
    fun primitiveMetricsParseLlmOutputs() =
        runBlocking {
            val llm =
                FakeLlm(
                    outputs =
                        listOf(
                            "0.83",
                            "pass",
                            "1) alpha\n2) beta\n3) gamma",
                        ),
                )

            val sample =
                SingleTurnSample(
                    userInput = "u",
                    response = "r",
                    reference = "ref",
                    retrievedContexts = listOf("ctx"),
                )

            val numeric =
                NumericMetric(
                    name = "numeric_test",
                    prompt = "Rate: {response}",
                    llm = llm,
                    allowedRange = 0.0..1.0,
                )
            val discrete =
                DiscreteMetric(
                    name = "discrete_test",
                    prompt = "Decide: {response}",
                    llm = llm,
                    allowedValues = listOf("pass", "fail"),
                )
            val ranking =
                RankingMetric(
                    name = "ranking_test",
                    prompt = "Rank: {response}",
                    llm = llm,
                    expectedSize = 3,
                )

            assertEquals(0.83, numeric.singleTurnAscore(sample))
            assertEquals("pass", discrete.singleTurnAscore(sample))
            assertEquals(listOf("alpha", "beta", "gamma"), ranking.singleTurnAscore(sample))
        }

    @Test
    fun discreteMetricReturnsNullWhenLlmOutputDoesNotMatchAllowedValues() =
        runBlocking {
            val llm = FakeLlm(outputs = listOf("undetermined"))
            val sample = SingleTurnSample(userInput = "u", response = "r")
            val discrete =
                DiscreteMetric(
                    name = "discrete_test",
                    prompt = "Decide: {response}",
                    llm = llm,
                    allowedValues = listOf("pass", "fail"),
                )

            assertEquals(null, discrete.singleTurnAscore(sample))
        }

    @Test
    fun discreteMetricPrefersLongestWordBoundaryMatch() =
        runBlocking {
            val llm = FakeLlm(outputs = listOf("Result: partially correct."))
            val sample = SingleTurnSample(userInput = "u", response = "r")
            val discrete =
                DiscreteMetric(
                    name = "discrete_overlap",
                    prompt = "Decide: {response}",
                    llm = llm,
                    allowedValues = listOf("correct", "partially correct"),
                )

            assertEquals("partially correct", discrete.singleTurnAscore(sample))
        }

    @Test
    fun rankingMetricParsesCommonListPrefixes() =
        runBlocking {
            val sample = SingleTurnSample(userInput = "u", response = "r", reference = "ref", retrievedContexts = listOf("ctx"))

            val itemPrefixLlm = FakeLlm(outputs = listOf("Item 1: alpha\nItem 2: beta\nItem 3: gamma"))
            val itemPrefixMetric =
                RankingMetric(
                    name = "ranking_item_prefix",
                    prompt = "Rank: {response}",
                    llm = itemPrefixLlm,
                    expectedSize = 3,
                )
            assertEquals(listOf("alpha", "beta", "gamma"), itemPrefixMetric.singleTurnAscore(sample))

            val alphaPrefixLlm = FakeLlm(outputs = listOf("A) red, B) green, C) blue"))
            val alphaPrefixMetric =
                RankingMetric(
                    name = "ranking_alpha_prefix",
                    prompt = "Rank: {response}",
                    llm = alphaPrefixLlm,
                    expectedSize = 3,
                )
            assertEquals(listOf("red", "green", "blue"), alphaPrefixMetric.singleTurnAscore(sample))

            val numericContentLlm = FakeLlm(outputs = listOf("1. 2024 Report\n2. 42 Insights"))
            val numericContentMetric =
                RankingMetric(
                    name = "ranking_numeric_content",
                    prompt = "Rank: {response}",
                    llm = numericContentLlm,
                    expectedSize = 2,
                )
            assertEquals(listOf("2024 Report", "42 Insights"), numericContentMetric.singleTurnAscore(sample))

            val noPunctuationNumericLlm = FakeLlm(outputs = listOf("2024 Report, 42 Insights"))
            val noPunctuationNumericMetric =
                RankingMetric(
                    name = "ranking_no_punctuation_numeric_content",
                    prompt = "Rank: {response}",
                    llm = noPunctuationNumericLlm,
                    expectedSize = 2,
                )
            assertEquals(listOf("2024 Report", "42 Insights"), noPunctuationNumericMetric.singleTurnAscore(sample))
        }

    @Test
    fun primitiveMetricsUseStructuredOutputWhenAvailable() =
        runBlocking {
            val llm =
                FakeStructuredLlm(
                    numeric = 0.91,
                    discrete = "pass",
                    ranking = listOf("alpha", "beta", "gamma"),
                )

            val sample =
                SingleTurnSample(
                    userInput = "u",
                    response = "r",
                    reference = "ref",
                    retrievedContexts = listOf("ctx"),
                )

            val numeric =
                NumericMetric(
                    name = "numeric_structured",
                    prompt = "Rate: {response}",
                    llm = llm,
                    allowedRange = 0.0..1.0,
                )
            val discrete =
                DiscreteMetric(
                    name = "discrete_structured",
                    prompt = "Decide: {response}",
                    llm = llm,
                    allowedValues = listOf("pass", "fail"),
                )
            val ranking =
                RankingMetric(
                    name = "ranking_structured",
                    prompt = "Rank: {response}",
                    llm = llm,
                    expectedSize = 3,
                )

            assertEquals(0.91, numeric.singleTurnAscore(sample))
            assertEquals("pass", discrete.singleTurnAscore(sample))
            assertEquals(listOf("alpha", "beta", "gamma"), ranking.singleTurnAscore(sample))
        }

    @Test
    fun numericMetricThrowsWhenNoNumericValueCanBeParsed() =
        runBlocking {
            val llm = FakeLlm(outputs = listOf("no numeric value here"))
            val sample = SingleTurnSample(userInput = "u", response = "r")
            val metric =
                NumericMetric(
                    name = "numeric_parse_fail",
                    prompt = "Rate: {response}",
                    llm = llm,
                    allowedRange = 0.0..1.0,
                )

            assertFailsWith<IllegalStateException> {
                metric.singleTurnAscore(sample)
            }

            Unit
        }
}

private class FakeLlm(
    private val outputs: List<String>,
) : BaseRagasLlm {
    private var cursor: Int = 0

    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        val next = outputs[cursor]
        cursor += 1
        return LlmResult(generations = listOf(LlmGeneration(text = next)))
    }
}

private class FakeStructuredLlm(
    private val numeric: Double?,
    private val discrete: String?,
    private val ranking: List<String>,
) : BaseRagasLlm,
    StructuredOutputRagasLlm {
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult =
        LlmResult(
            generations =
                listOf(
                    LlmGeneration(text = "fallback"),
                ),
        )

    override suspend fun generateNumericValue(prompt: String): Double? = numeric

    override suspend fun generateDiscreteValue(prompt: String): String? = discrete

    override suspend fun generateRankingItems(prompt: String): List<String>? = ranking
}
