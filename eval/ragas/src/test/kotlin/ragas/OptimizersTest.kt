package ragas

import ragas.cache.InMemoryCacheBackend
import ragas.optimizers.DspyAdapter
import ragas.optimizers.DspyCompileContext
import ragas.optimizers.DspyOptimizer
import ragas.optimizers.GeneticOptimizer
import ragas.optimizers.OptimizationDataset
import ragas.optimizers.OptimizationExample
import ragas.optimizers.OptimizerConfig
import ragas.optimizers.OptimizerOutcome
import ragas.optimizers.OptimizerPrompt
import ragas.optimizers.asTextPrompt
import ragas.prompt.PromptContentPart
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class OptimizersTest {
    @Test
    fun geneticOptimizerFindsPromptContainingTargetWord() {
        val optimizer = GeneticOptimizer()
        val dataset =
            OptimizationDataset(
                metricName = "dummy",
                examples = listOf(OptimizationExample(promptInput = mapOf("q" to "x"), expectedOutput = "y")),
            )

        val result =
            optimizer.optimize(
                dataset = dataset,
                initialPrompts = listOf("Answer freely", "Return concise result"),
                evaluator = { prompt, _ -> if ("concise" in prompt.lowercase()) 1.0 else 0.2 },
                config = OptimizerConfig(iterations = 4, populationSize = 6, seed = 7),
            )

        assertTrue(result.containsKey("optimized_prompt"))
        assertTrue("concise" in result.getValue("optimized_prompt").lowercase())
    }

    @Test
    fun geneticOptimizerIsDeterministicWithSeed() {
        val optimizer = GeneticOptimizer()
        val dataset =
            OptimizationDataset(
                metricName = "dummy",
                examples = listOf(OptimizationExample(promptInput = mapOf("q" to "x"), expectedOutput = "y")),
            )

        val cfg = OptimizerConfig(iterations = 3, populationSize = 5, seed = 123)
        val eval = { prompt: String, _: OptimizationDataset -> if ("final" in prompt.lowercase()) 0.9 else 0.1 }

        val r1 = optimizer.optimize(dataset, listOf("Return final value", "Explain"), eval, cfg)
        val r2 = optimizer.optimize(dataset, listOf("Return final value", "Explain"), eval, cfg)
        assertEquals(r1, r2)
    }

    @Test
    fun dspyOptimizerOptimizesTextPrompts() {
        val optimizer = DspyOptimizer()
        val dataset =
            OptimizationDataset(
                metricName = "dummy",
                examples = listOf(OptimizationExample(promptInput = mapOf("q" to "x"), expectedOutput = "y")),
            )

        val result =
            optimizer.optimize(
                dataset = dataset,
                initialPrompts = listOf("Answer with detail"),
                evaluator = { prompt, _ -> if ("concise" in prompt.lowercase()) 1.0 else 0.1 },
                config = OptimizerConfig(iterations = 2),
            )

        assertTrue("concise" in result.getValue("optimized_prompt").lowercase())
        assertEquals("dspy", result.getValue("optimizer"))
    }

    @Test
    fun dspyOptimizerUsesCacheForRepeatedEvaluation() {
        val cache = InMemoryCacheBackend()
        val adapter =
            DspyAdapter { context: DspyCompileContext ->
                listOf(
                    OptimizerPrompt.Text(context.currentBestPrompt.toString()),
                    OptimizerPrompt.Text("Return concise result"),
                )
            }
        val optimizer = DspyOptimizer(adapter = adapter, cache = cache)
        val dataset =
            OptimizationDataset(
                metricName = "dummy",
                examples = listOf(OptimizationExample(promptInput = mapOf("q" to "x"), expectedOutput = "y")),
            )

        var calls = 0
        val evaluator = { prompt: OptimizerPrompt, _: OptimizationDataset ->
            calls += 1
            when (prompt) {
                is OptimizerPrompt.Text -> if ("concise" in prompt.value.lowercase()) 1.0 else 0.1
                is OptimizerPrompt.MultiModal -> 0.0
            }
        }

        optimizer.optimizePrompts(
            dataset = dataset,
            initialPrompts = listOf(OptimizerPrompt.Text("Answer freely")),
            evaluator = evaluator,
            config = OptimizerConfig(iterations = 2),
        )
        val firstRunCalls = calls

        optimizer.optimizePrompts(
            dataset = dataset,
            initialPrompts = listOf(OptimizerPrompt.Text("Answer freely")),
            evaluator = evaluator,
            config = OptimizerConfig(iterations = 2),
        )

        assertTrue(firstRunCalls > 0)
        assertEquals(firstRunCalls, calls)
    }

    @Test
    fun geneticOptimizerSupportsTypedPromptObjects() {
        val optimizer = GeneticOptimizer()
        val dataset =
            OptimizationDataset(
                metricName = "dummy",
                examples = listOf(OptimizationExample(promptInput = mapOf("q" to "x"), expectedOutput = "y")),
            )

        val outcome =
            optimizer.optimizePrompts(
                dataset = dataset,
                initialPrompts =
                    listOf(
                        OptimizerPrompt.Text("Answer freely"),
                        OptimizerPrompt.Text("Return concise result"),
                    ),
                evaluator = { prompt, _ ->
                    when (prompt) {
                        is OptimizerPrompt.Text -> if ("concise" in prompt.value.lowercase()) 1.0 else 0.2
                        is OptimizerPrompt.MultiModal -> 0.0
                    }
                },
                config = OptimizerConfig(iterations = 4, populationSize = 6, seed = 7),
            )

        val optimized = outcome.optimizedPrompt
        assertTrue(optimized is OptimizerPrompt.Text)
        assertTrue("concise" in optimized.value.lowercase())
    }

    @Test
    fun geneticOptimizerSupportsMultimodalPromptObjects() {
        val optimizer = GeneticOptimizer()
        val dataset =
            OptimizationDataset(
                metricName = "dummy",
                examples = listOf(OptimizationExample(promptInput = mapOf("q" to "x"), expectedOutput = "y")),
            )

        val outcome =
            optimizer.optimizePrompts(
                dataset = dataset,
                initialPrompts =
                    listOf(
                        OptimizerPrompt.MultiModal(
                            listOf(
                                PromptContentPart.Text("Describe the chart."),
                                PromptContentPart.ImageUrl("https://example.com/chart.png"),
                            ),
                        ),
                        OptimizerPrompt.MultiModal(
                            listOf(
                                PromptContentPart.Text("Be concise when describing the chart."),
                                PromptContentPart.ImageUrl("https://example.com/chart.png"),
                            ),
                        ),
                    ),
                evaluator = { prompt, _ ->
                    when (prompt) {
                        is OptimizerPrompt.MultiModal -> {
                            val projected = prompt.content.joinToString("\n") { it.toPromptText() }
                            if ("concise" in projected.lowercase()) 1.0 else 0.1
                        }

                        is OptimizerPrompt.Text -> {
                            0.0
                        }
                    }
                },
                config = OptimizerConfig(iterations = 3, populationSize = 4, seed = 9),
            )

        val optimized = outcome.optimizedPrompt
        assertTrue(optimized is OptimizerPrompt.MultiModal)
        val projected = optimized.content.joinToString("\n") { it.toPromptText() }
        assertTrue("concise" in projected.lowercase())
    }

    @Test
    fun optimizerOutcomeLegacyMapKeepsReservedKeysFromCoreFields() {
        val outcome =
            OptimizerOutcome(
                optimizedPrompt =
                    OptimizerPrompt.MultiModal(
                        listOf(
                            PromptContentPart.Text("Core prompt text"),
                            PromptContentPart.ImageUrl("https://example.com/core.png"),
                        ),
                    ),
                metadata =
                    mapOf(
                        "optimized_prompt" to "metadata_override_attempt",
                        "optimized_prompt_type" to "metadata_override_attempt",
                        "optimizer" to "genetic",
                    ),
            )

        val legacy = outcome.toLegacyMap()

        assertEquals("genetic", legacy["optimizer"])
        assertEquals("multimodal", legacy["optimized_prompt_type"])
        assertEquals(outcome.optimizedPrompt.asTextPrompt(), legacy["optimized_prompt"])
    }

    @Test
    fun optimizerConfigRejectsInvalidValues() {
        assertFailsWith<IllegalArgumentException> {
            OptimizerConfig(iterations = 0)
        }
        assertFailsWith<IllegalArgumentException> {
            OptimizerConfig(populationSize = 0)
        }
        assertFailsWith<IllegalArgumentException> {
            OptimizerConfig(mutationProbability = -0.1)
        }
        assertFailsWith<IllegalArgumentException> {
            OptimizerConfig(mutationProbability = 1.1)
        }
    }
}
