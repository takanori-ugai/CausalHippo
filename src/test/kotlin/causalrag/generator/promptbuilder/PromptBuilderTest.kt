package causalrag.generator.promptbuilder

import shared.chunking.countTokensWithJTokKit
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PromptBuilderTest {
    private val query = "What causes coastal flooding?"
    private val passages =
        listOf(
            "Climate change causes rising sea levels.",
            "Rising sea levels lead to coastal flooding.",
        )
    private val causalPaths =
        listOf(
            listOf("Climate change", "Rising sea levels", "Coastal flooding"),
        )
    private val causalGraphSummary = "Climate change increases flood risk through sea level rise."

    private fun buildTestPrompt(templateStyle: String): String =
        buildPrompt(
            query = query,
            passages = passages,
            causalPaths = causalPaths,
            causalGraphSummary = causalGraphSummary,
            templateStyle = templateStyle,
        )

    @Test
    fun buildPromptFallsBackToDefaultJteTemplateForBasicStyle() {
        val prompt = buildTestPrompt("basic")

        assertContains(prompt, "You are a causal reasoning assistant. Answer the following question using:")
        assertContains(prompt, "Relevant causal relationships:")
        assertContains(prompt, "Climate change → Rising sea levels → Coastal flooding")
        assertContains(prompt, "Question: What causes coastal flooding?")
        assertContains(prompt, "Final answer requirements:")
        assertContains(prompt, "Final answer:")
    }

    @Test
    fun buildPromptUsesStructuredTemplateStyle() {
        val prompt = buildTestPrompt("structured")

        assertContains(prompt, "You are a causal reasoning assistant that explains complex relationships between concepts.")
        assertContains(prompt, "Relevant causal pathways:")
        assertContains(prompt, "Climate change → Rising sea levels → Coastal flooding")
        assertContains(prompt, "Causal graph structure: Climate change increases flood risk through sea level rise.")
        assertContains(prompt, "Your structured causal answer:")
    }

    @Test
    fun buildPromptUsesChainOfThoughtTemplateStyle() {
        val prompt = buildTestPrompt("chain_of_thought")

        assertContains(prompt, "You are an expert in causal reasoning who answers complex questions by tracing causal mechanisms.")
        assertContains(prompt, "CAUSAL RELATIONSHIPS TO CONSIDER:")
        assertContains(prompt, "Climate change → Rising sea levels → Coastal flooding")
        assertContains(prompt, "Global causal structure: Climate change increases flood risk through sea level rise.")
        assertContains(prompt, "STEP-BY-STEP REASONING:")
    }

    @Test
    fun buildPromptUsesExperimentsTemplateStyle() {
        val prompt = buildTestPrompt("experiments")

        assertContains(prompt, "Output requirements (strict):")
        assertContains(prompt, "\"answer\":\"<short answer span>\"")
        assertContains(prompt, "\"explanation\":\"<causal explanation>\"")
        assertContains(prompt, "JSON response:")
    }

    @Test
    fun summaryRequestsStayWithinBudget() {
        val directory =
            java.nio.file.Files
                .createTempDirectory("short-causal-template")
        try {
            java.nio.file.Files
                .writeString(directory.resolve("causal_prompt.jte"), "@param String query\n\${query}")
            val model = io.mockk.mockk<causalrag.generator.llm.LLMInterface>(relaxed = true)
            val builder = PromptBuilder(templateStyle = "basic", llmInterface = model, templatesDir = directory.toString())
            val baseline = builder.buildBudgetedPrompt(query, emptyList(), causalPaths, 100, summarize = false)
            val budget = countTokensWithJTokKit(baseline.text)
            val result = builder.buildBudgetedPrompt(query, emptyList(), causalPaths, budget, summarize = true)
            assertEquals(causalPaths, result.paths)
            assertEquals(baseline.text, result.text)
            io.mockk.verify(exactly = 0) {
                model.generate(any(), any(), any(), any(), any(), any())
            }
        } finally {
            directory.toFile().deleteRecursively()
        }
    }

    @Test
    fun rejectsQuestionsThatCannotFitTheBarePrompt() {
        val builder = PromptBuilder(templateStyle = "basic")
        val failure =
            assertFailsWith<IllegalArgumentException> {
                builder.buildBudgetedPrompt(query, passages, causalPaths, 1, summarize = false)
            }
        assertContains(failure.message.orEmpty(), "exceed")
    }

    @Test
    fun summarizesLongPathsWithAndWithoutABudget() {
        val llm = io.mockk.mockk<causalrag.generator.llm.LLMInterface>()
        io.mockk.every { llm.generate(any(), any(), any(), any(), any(), any()) } returns "Short causal summary."
        val builder = PromptBuilder(templateStyle = "detailed", llmInterface = llm)
        val paths = listOf(listOf("rain", "river", "flood", "displacement"))
        val bounded = builder.buildBudgetedPrompt(query, emptyList(), paths, 1_000, summarize = true)
        assertContains(bounded.text, "Short causal summary.")
        assertContains(builder.buildPrompt(query, emptyList(), paths), "Short causal summary.")
        io.mockk.verify(exactly = 2) { llm.generate(match { it.contains("Causal path:") }, any(), any(), any(), any(), any()) }
    }

    @Test
    fun retainsUnsummarizedEvidenceWhenTheSummaryWouldExceedTheBudget() {
        val llm = io.mockk.mockk<causalrag.generator.llm.LLMInterface>()
        io.mockk.every { llm.generate(any(), any(), any(), any(), any(), any()) } returns "oversized summary ".repeat(1_000)
        val builder = PromptBuilder(templateStyle = "detailed", llmInterface = llm)
        val baseline = builder.buildBudgetedPrompt(query, passages, causalPaths, 1_000, summarize = false)
        val withSummary = builder.buildBudgetedPrompt(query, passages, causalPaths, 1_000, summarize = true)
        assertEquals(baseline, withSummary)
        io.mockk.verify(exactly = 1) { llm.generate(any(), any(), any(), any(), any(), any()) }
    }

    @Test
    fun summaryRequestWithoutPathsDoesNotCallTheModel() {
        val llm = io.mockk.mockk<causalrag.generator.llm.LLMInterface>()
        val builder = PromptBuilder(templateStyle = "detailed", llmInterface = llm)
        val result = builder.buildBudgetedPrompt(query, passages, emptyList(), 1_000, summarize = true)
        assertEquals(passages, result.passages)
        io.mockk.verify(exactly = 0) { llm.generate(any(), any(), any(), any(), any(), any()) }
    }

    @Test
    fun summaryRequestWithoutAModelKeepsOriginalEvidence() {
        val builder = PromptBuilder(templateStyle = "detailed")
        assertEquals(
            builder.buildBudgetedPrompt(query, passages, causalPaths, 1_000, summarize = false),
            builder.buildBudgetedPrompt(query, passages, causalPaths, 1_000, summarize = true),
        )
    }

    @Test
    fun budgetCountsTheRenderedPromptAndKeepsShorterEvidence() {
        val builder = PromptBuilder(templateStyle = "detailed")
        val short = passages.last()
        val usablePath = causalPaths.single()
        val budget = countTokensWithJTokKit(builder.buildPrompt(query, listOf(short), listOf(usablePath)))
        val oversized = "unrelated material ".repeat(budget)

        val result =
            builder.buildBudgetedPrompt(
                query,
                listOf(oversized, short),
                listOf(listOf(oversized, "flooding"), usablePath),
                budget,
                summarize = false,
            )

        assertEquals(listOf(short), result.passages)
        assertEquals(listOf(usablePath), result.paths)
        assertTrue(countTokensWithJTokKit(result.text) <= budget)
    }
}
