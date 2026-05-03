package causalrag.examples

import causalhippo.CausalHippoPipeline
import causalrag.retriever.HippoRagSemanticMode
import hipporag.config.BaseConfig
import java.nio.file.Files
import java.nio.file.Path

/**
 * Demonstrates a two-stage pipeline:
 * 1. HippoRAG casts a wide net for candidate passages.
 * 2. CausalRAG reranks those candidates using explicit causal paths.
 *
 * This example requires a HippoRAG-compatible LLM/embedding setup such as OpenAI or Ollama.
 */
fun main() {
    val documents =
        listOf(
            "Heavy rainfall saturates soil and causes rivers to overflow.",
            "When rivers overflow, downstream neighborhoods experience flooding.",
            "Flooding damages roads, homes, and electrical infrastructure.",
            "Installing retention basins reduces runoff and lowers flood risk.",
            "Wetlands absorb excess water and can prevent severe flooding after storms.",
            "Sea walls protect shorelines, but they do not address upstream rainfall runoff.",
        )

    val hippoConfig =
        BaseConfig(
            llmName = "gpt-5.4-mini",
            embeddingModelName = "text-embedding-3-small",
            saveDir = "outputs/hippo-causal-demo",
        )

    val pipeline =
        CausalHippoPipeline(
            modelName = "gpt-5.4-mini",
            embeddingModel = "text-embedding-3-small",
            hippoConfig = hippoConfig,
            hippoSemanticMode = HippoRagSemanticMode.GRAPH,
        )

    println("Indexing documents into HippoRAG + causal graph...")
    pipeline.index(documents)
    val causalGraphPath = Path.of(hippoConfig.saveDir, "causal-graph.json")
    Files.createDirectories(causalGraphPath.parent)
    val graphSaved = pipeline.graphBuilder.save(causalGraphPath.toString())
    if (graphSaved) {
        println("Saved causal graph to: $causalGraphPath")
    } else {
        println("Failed to save causal graph to: $causalGraphPath")
    }

    val query = "What reduces flooding caused by heavy rainfall?"
    println("\nQuery: $query")

    val context = pipeline.retrieveContext(query, topK = 3)
    println("\nCausally reranked context:")
    context.forEachIndexed { index, passage ->
        println("${index + 1}. $passage")
    }

    val paths = pipeline.retrieveCausalPaths(query, maxPaths = 3)
    println("\nRelevant causal paths:")
    if (paths.isEmpty()) {
        println("No causal paths found.")
    } else {
        paths.forEach { path ->
            println("- ${path.joinToString(" -> ")}")
        }
    }

    println("\nGenerating answer...")
    val result = pipeline.runWithContext(query, topK = 3)
    println("Answer: ${result.answer}")
}
