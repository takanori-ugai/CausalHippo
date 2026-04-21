package causalrag.generator.promptbuilder

import kotlin.test.Test
import kotlin.test.assertContains

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
}
