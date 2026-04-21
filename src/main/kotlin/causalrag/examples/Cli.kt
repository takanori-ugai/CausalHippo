package causalrag.examples

import causalrag.causalgraph.builder.CausalGraphBuilder
import causalrag.causalgraph.explainer.CausalGraphExplainer
import causalrag.retriever.HippoRagSemanticMode
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.isDirectory
import kotlin.io.path.name
import kotlin.streams.toList

private const val VERSION = "0.0.1"
private const val GRAPH_FILENAME = "graph.json"
private const val GRAPH_HTML_FILENAME = "graph.html"

/**
 * Entry point for the repository CLI.
 *
 * @param args Command-line arguments.
 */
fun main(args: Array<String>) {
    if (args.isEmpty()) {
        printUsage()
        return
    }

    if (args.contains("--version")) {
        println("HybridRAG Kotlin version $VERSION")
        return
    }

    when (args.first()) {
        "index" -> handleIndex(args.drop(1))
        "query" -> handleQuery(args.drop(1))
        "visualize" -> handleVisualize(args.drop(1))
        "serve" -> handleServe()
        "evaluate" -> handleEvaluate(args.drop(1))
        else -> printUsage()
    }
}

private fun handleIndex(args: List<String>) {
    val opts = CliUtils.parseOptions(args)
    val input = opts["input"] ?: opts["i"]
    val output = opts["output"] ?: opts["o"]
    val model = opts["model"]
    val llmModel = opts["llm-model"] ?: opts["generator-model"] ?: model
    val embeddingModel = opts["embedding-model"] ?: model
    val config = opts["config"]
    val semanticMode = parseSemanticMode(opts["semantic-mode"])

    if (input == null || output == null) {
        println("Missing required --input/-i or --output/-o")
        return
    }

    val documents = loadDocuments(input)
    if (documents.isEmpty()) {
        println("No documents found to index")
        return
    }

    val pipeline =
        CliUtils.createHybridPipeline(
            configPath = config,
            workingDir = output,
            modelName = llmModel,
            embeddingModel = embeddingModel,
            semanticMode = semanticMode,
        )

    println("Indexing ${documents.size} documents with HybridRAG...")
    pipeline.index(documents)
    val graphSaved = pipeline.graphBuilder.save(indexGraphPath(output))
    if (graphSaved) {
        println("Saved index to $output")
    } else {
        println("Indexing complete but failed to save to $output")
    }
}

private fun handleQuery(args: List<String>) {
    val opts = CliUtils.parseOptions(args)
    val indexDir = opts["index"] ?: opts["i"]
    val query = opts["query"] ?: opts["q"]
    val model = opts["model"]
    val llmModel = opts["llm-model"] ?: opts["generator-model"] ?: model
    val embeddingModel = opts["embedding-model"] ?: model
    val topK = (opts["top-k"] ?: "5").toIntOrNull() ?: 5
    val config = opts["config"]
    val semanticMode = parseSemanticMode(opts["semantic-mode"])

    if (indexDir == null || query == null) {
        println("Missing required --index/-i or --query/-q")
        return
    }

    val pipeline =
        CliUtils.createHybridPipeline(
            configPath = config,
            workingDir = indexDir,
            modelName = llmModel,
            embeddingModel = embeddingModel,
            semanticMode = semanticMode,
        )

    if (!pipeline.graphBuilder.load(indexGraphPath(indexDir))) {
        println("Failed to load index from $indexDir")
        return
    }

    println("\n" + "=".repeat(80))
    println("Query: $query")
    println("=".repeat(80))

    val answer = pipeline.run(query, topK = topK)
    println("\nAnswer: $answer")

    println("\nSupporting Context:")
    val context = pipeline.retrieveContext(query, topK = topK)
    context.forEachIndexed { idx, ctx ->
        val preview = if (ctx.length > 200) ctx.substring(0, 200) + "..." else ctx
        println("[${idx + 1}] $preview")
    }

    val causalPaths = pipeline.graphRetriever.retrievePaths(query, maxPaths = 3)
    if (causalPaths.isNotEmpty()) {
        println("\nRelevant Causal Pathways:")
        causalPaths.forEachIndexed { idx, path ->
            println("[${idx + 1}] ${path.joinToString(" -> ")}")
        }
    }
}

private fun handleServe() {
    println("Serve is not implemented in the Kotlin port yet.")
}

private fun handleVisualize(args: List<String>) {
    val opts = CliUtils.parseOptions(args)
    val indexDir = opts["index"] ?: opts["i"]
    val graphPath = opts["graph"] ?: indexDir?.let(::indexGraphPath)
    val outputPath = opts["output"] ?: opts["o"] ?: indexDir?.let { Path.of(it, GRAPH_HTML_FILENAME).toString() } ?: GRAPH_HTML_FILENAME
    val highlightNodes = parseCsvList(opts["highlight-nodes"])

    if (graphPath == null) {
        println("Missing required --index/-i or --graph")
        return
    }

    val graphBuilder = CausalGraphBuilder()
    if (!graphBuilder.load(graphPath)) {
        println("Failed to load graph from $graphPath")
        return
    }

    val explainer = CausalGraphExplainer(graphBuilder.getGraph(), graphBuilder.nodeText)
    val html = explainer.generateGraphVizHtml(highlightNodes = highlightNodes)

    val output = Path.of(outputPath)
    output.parent?.let { Files.createDirectories(it) }
    Files.writeString(output, html, StandardCharsets.UTF_8)

    println("Saved graph visualization to $outputPath")
}

private fun handleEvaluate(args: List<String>) {
    val opts = CliUtils.parseOptions(args)
    val evalData = opts["eval-data"]
    val outputDir = opts["output-dir"] ?: "./eval_results"
    val modelName = opts["model-name"] ?: "gpt-4"
    val evalModel = opts["eval-model"]
    val embeddingModel = opts["embedding-model"] ?: "text-embedding-3-small"
    val apiKey = opts["api-key"]
    val provider = opts["provider"] ?: "openai"
    val indexDir = opts["index"]

    if (evalData == null) {
        println("Missing required --eval-data")
        return
    }

    val result =
        CliUtils.runEvaluation(
            config =
                CliUtils.EvalRunConfig(
                    evalDataPath = evalData,
                    outputDir = outputDir,
                    modelName = modelName,
                    evalModel = evalModel,
                    embeddingModel = embeddingModel,
                    apiKey = apiKey,
                    provider = provider,
                    indexDir = indexDir,
                ),
            warn = { msg -> println(msg) },
            error = { msg -> println(msg) },
        )
            ?: return

    println("Evaluation complete! Summary:")
    result.results.metrics.forEach { (metric, score) ->
        println("  $metric: ${"%.4f".format(score)}")
    }
    println("Detailed results saved to ${result.outputDir}")
}

private fun loadDocuments(input: String): List<String> {
    val path = Path.of(input)
    return if (path.isDirectory()) {
        Files.walk(path).use { stream ->
            stream
                .filter { Files.isRegularFile(it) }
                .filter { it.name.endsWith(".txt") }
                .toList()
                .mapNotNull { readTextFile(it) }
        }
    } else {
        readTextFile(path)?.let { listOf(it) } ?: emptyList()
    }
}

private fun readTextFile(path: Path): String? =
    try {
        val text = Files.readString(path, StandardCharsets.UTF_8)
        if (text.isBlank()) null else text
    } catch (_: java.io.IOException) {
        null
    }

private fun parseSemanticMode(raw: String?): HippoRagSemanticMode =
    when (raw?.lowercase()) {
        null, "", "graph" -> {
            HippoRagSemanticMode.GRAPH
        }

        "dpr" -> {
            HippoRagSemanticMode.DPR
        }

        else -> {
            println("Unknown --semantic-mode '$raw'; defaulting to graph")
            HippoRagSemanticMode.GRAPH
        }
    }

private fun parseCsvList(raw: String?): List<String>? {
    if (raw.isNullOrBlank()) return null
    val parts =
        raw
            .split(',')
            .map { it.trim() }
            .filter { it.isNotEmpty() }
    return if (parts.isEmpty()) null else parts
}

private fun indexGraphPath(indexDir: String): String = Path.of(indexDir, GRAPH_FILENAME).toString()

private fun printUsage() {
    println(
        """
HybridRAG Kotlin CLI

Usage:
  cli index --input <dir|file> --output <dir> [--config <path>] [--llm-model <model>] [--embedding-model <model>] [--semantic-mode <graph|dpr>]
  cli query --index <dir> --query <text> [--config <path>] [--llm-model <model>] [--embedding-model <model>] [--top-k <n>] [--semantic-mode <graph|dpr>]
  cli visualize [--index <dir> | --graph <path>] [--output <path>] [--highlight-nodes <a,b,c>]
  cli serve
  cli evaluate --eval-data <path> [--index <dir>] [--output-dir <dir>] [--model-name <llm_model>] [--eval-model <llm_model>] [--embedding-model <embedding_model>] [--api-key <key>] [--provider <name>]
  cli --version
        """.trimIndent(),
    )
}
