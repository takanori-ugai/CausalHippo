package causalrag

import causalrag.causalgraph.builder.CausalGraphBuilder
import causalrag.causalgraph.retriever.CausalPathRetriever
import causalrag.generator.llm.LLMInterface
import causalrag.generator.promptbuilder.buildPrompt
import causalrag.reranker.CausalPathReranker
import causalrag.retriever.Bm25Retriever
import causalrag.retriever.HybridRetriever
import causalrag.retriever.VectorStoreRetriever
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import shared.config.CommonRagConfigLoader
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path

private val logger = KotlinLogging.logger {}

/**
 * Serializable configuration for constructing a [CausalRAGPipeline].
 *
 * @property modelName Chat model name used for answer generation.
 * @property embeddingModel Embedding model name used for retrieval and graph building.
 * @property graphPath Optional path to a serialized causal graph.
 * @property indexPath Optional path to a serialized vector index.
 * @property llmProvider LLM backend identifier.
 * @property llmApiKey API key used by the configured LLM provider.
 * @property llmBaseUrl Optional base URL override for self-hosted providers.
 * @property embeddingApiKey API key used by the embedding provider.
 * @property templateStyle Prompt template style.
 * @property semanticMode HippoRAG semantic retrieval mode (`graph` or `dpr`) for HybridRAG.
 * @property minCausalMatches Minimum matched causal nodes required to keep a candidate passage.
 */
@Serializable
data class PipelineConfig(
    val modelName: String? = null,
    val embeddingModel: String? = null,
    val graphPath: String? = null,
    val indexPath: String? = null,
    val llmProvider: String? = null,
    val llmApiKey: String? = null,
    val llmBaseUrl: String? = null,
    val embeddingApiKey: String? = null,
    val templateStyle: String? = null,
    val semanticMode: String? = null,
    val minCausalMatches: Int? = null,
)

/**
 * Output produced by [CausalRAGPipeline.runWithContext].
 *
 * @property answer Generated answer text.
 * @property context Retrieved passages used as supporting context.
 * @property causalPaths Retrieved causal paths supplied to generation.
 */
data class PipelineRunResult(
    val answer: String,
    val context: List<String>,
    val causalPaths: List<List<String>>,
)

/**
 * Top-level orchestration of the CausalRAG pipeline.
 */
@Suppress("TooGenericExceptionCaught")
class CausalRAGPipeline(
    modelName: String = "gpt-4",
    embeddingModel: String = "all-MiniLM-L6-v2",
    graphPath: String? = null,
    indexPath: String? = null,
    configPath: String? = null,
    templateStyle: String? = null,
    dynamicWeightingEnabled: Boolean = false,
    twoPassAdaptiveEnabled: Boolean = false,
    confidenceBasedSwitchEnabled: Boolean = false,
) {
    private val config: PipelineConfig? = configPath?.let { loadConfig(it) }
    private val effectiveModelName = config?.modelName ?: modelName
    private val effectiveEmbeddingModel = config?.embeddingModel ?: embeddingModel
    private val effectiveGraphPath = config?.graphPath ?: graphPath
    private val effectiveIndexPath = config?.indexPath ?: indexPath
    private val effectiveLlmProvider = config?.llmProvider ?: "openai"
    private val effectiveLlmApiKey = config?.llmApiKey ?: System.getenv("OPENAI_API_KEY")
    private val effectiveLlmBaseUrl = config?.llmBaseUrl
    private val effectiveEmbeddingApiKey = config?.embeddingApiKey ?: System.getenv("OPENAI_API_KEY")
    private val effectiveTemplateStyle = templateStyle ?: config?.templateStyle ?: "detailed"
    private val effectiveMinCausalMatches = config?.minCausalMatches ?: 0

    // Core components
    internal val llm: LLMInterface =
        LLMInterface(
            modelName = effectiveModelName,
            provider = effectiveLlmProvider,
            apiKey = effectiveLlmApiKey,
            baseUrl = effectiveLlmBaseUrl,
        )
    internal val graphBuilder: CausalGraphBuilder =
        CausalGraphBuilder(
            modelName = effectiveEmbeddingModel,
            graphPath = effectiveGraphPath,
            embeddingApiKey = effectiveEmbeddingApiKey,
            extractorMethod = "hybrid",
            llmInterface = llm,
        )
    internal val vectorRetriever: VectorStoreRetriever =
        VectorStoreRetriever(
            embeddingModel = effectiveEmbeddingModel,
            indexPath = effectiveIndexPath,
            embeddingApiKey = effectiveEmbeddingApiKey,
        )
    internal val bm25Retriever: Bm25Retriever = Bm25Retriever()
    internal val graphRetriever: CausalPathRetriever = CausalPathRetriever(graphBuilder)
    internal val hybridRetriever: HybridRetriever =
        HybridRetriever(
            semanticRetriever = vectorRetriever,
            graphRetriever,
            semanticWeight = 0.4,
            causalWeight = 0.5,
            bm25Weight = 0.1,
            minCausalMatches = effectiveMinCausalMatches,
            bm25Retriever = bm25Retriever,
            dynamicWeightingEnabled = dynamicWeightingEnabled,
            twoPassAdaptiveEnabled = twoPassAdaptiveEnabled,
            confidenceBasedSwitchEnabled = confidenceBasedSwitchEnabled,
        )
    internal val reranker: CausalPathReranker = CausalPathReranker(graphRetriever)

    /** Load configuration from file. */
    private fun loadConfig(configPath: String): PipelineConfig {
        val path = Path.of(configPath)
        require(Files.exists(path)) { "Config file not found: $configPath" }
        return try {
            val content = Files.readString(path)
            CommonRagConfigLoader.parseOrNull(content)?.toPipelineConfig()
                ?: Json { ignoreUnknownKeys = true }.decodeFromString(PipelineConfig.serializer(), content)
        } catch (ex: IOException) {
            logger.error(ex) { "Failed to load config from $configPath" }
            throw ex
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Failed to load config from $configPath" }
            throw ex
        }
    }

    /**
     * Builds the causal graph and retrieval indexes from the supplied documents.
     *
     * @param documents Source documents to ingest.
     */
    fun index(documents: List<String>) {
        graphBuilder.indexDocuments(documents)
        vectorRetriever.indexCorpus(documents)
        bm25Retriever.indexDocuments(documents)
    }

    /**
     * Replaces the current in-memory indexes with [documents].
     *
     * This keeps heavyweight clients alive while ensuring per-run retrieval state does not leak.
     */
    fun reindex(documents: List<String>) {
        graphBuilder.clear()
        vectorRetriever.clear()
        bm25Retriever.clear()
        index(documents)
    }

    /**
     * Saves the graph and vector index to a directory.
     *
     * @param dir Target directory.
     * @return `true` when both graph and vector index are saved successfully.
     */
    fun save(dir: String): Boolean =
        try {
            val path = Path.of(dir)
            Files.createDirectories(path)
            val graphSaved = graphBuilder.save(path.resolve("graph.json").toString())
            val vectorsSaved = vectorRetriever.saveIndex(path.resolve("vector_cache").toString())
            graphSaved && vectorsSaved
        } catch (ex: IOException) {
            logger.error(ex) { "Failed to save pipeline data to $dir" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Failed to save pipeline data to $dir" }
            false
        }

    /**
     * Loads the graph and vector index from a directory.
     *
     * @param dir Directory containing serialized pipeline artifacts.
     * @return `true` when both graph and vector index are loaded successfully.
     */
    fun load(dir: String): Boolean =
        try {
            val path = Path.of(dir)
            val graphLoaded = graphBuilder.load(path.resolve("graph.json").toString())
            val vectorsLoaded = vectorRetriever.loadIndex(path.resolve("vector_cache").toString())
            if (vectorsLoaded) {
                val passages = vectorRetriever.getPassages()
                if (passages.isNotEmpty()) {
                    bm25Retriever.indexDocuments(passages)
                } else {
                    logger.warn { "Vector index loaded but no passages found to rebuild BM25 index." }
                }
            }
            graphLoaded && vectorsLoaded
        } catch (ex: IOException) {
            logger.error(ex) { "Failed to load pipeline data from $dir" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Failed to load pipeline data from $dir" }
            false
        }

    /**
     * Runs the end-to-end pipeline and returns only the generated answer.
     *
     * @param query User query.
     * @param topK Number of passages to keep after reranking.
     * @return Generated answer text.
     */
    fun run(
        query: String,
        topK: Int = 5,
    ): String = runWithContext(query, topK = topK).answer

    /**
     * Runs the end-to-end pipeline and returns the answer together with retrieval context.
     *
     * @param query User query.
     * @param topK Number of passages to keep after reranking.
     * @return Answer, context passages, and causal paths used for prompting.
     */
    fun runWithContext(
        query: String,
        topK: Int = 5,
    ): PipelineRunResult {
        // Step 1: Hybrid retrieval
        val candidates = hybridRetriever.retrieve(query, topK = topK)

        // Step 2: Rerank via causal path
        val reranked = reranker.rerank(query, candidates)
        val rerankedPassages = reranked.map { it.first }.take(topK)

        // Step 3: Build prompt with causal context
        val causalNodes = graphRetriever.retrievePathNodes(query)
        val causalPaths = graphRetriever.retrievePaths(query, maxPaths = 3)
        val prompt =
            buildPrompt(
                query,
                rerankedPassages,
                causalPaths = causalPaths,
                causalNodes = causalNodes,
                templateStyle = effectiveTemplateStyle,
                llmInterface = llm,
            )

        // Step 4: Generate answer
        val answer = llm.generate(prompt, jsonMode = requiresJsonResponseFormat(effectiveTemplateStyle))
        return PipelineRunResult(answer, rerankedPassages, causalPaths)
    }

    private fun requiresJsonResponseFormat(style: String): Boolean = style.equals("experiments", ignoreCase = true)

    /**
     * Retrieves supporting passages without invoking answer generation.
     *
     * @param query User query.
     * @param topK Maximum number of passages to return.
     * @return Retrieved passages.
     */
    fun retrieveContext(
        query: String,
        topK: Int = 5,
    ): List<String> = hybridRetriever.retrieve(query, topK = topK)

    /**
     * Retrieves causal paths relevant to a query.
     *
     * @param query User query.
     * @param maxPaths Maximum number of paths to return.
     * @return Relevant causal paths expressed as ordered node labels.
     */
    fun retrieveCausalPaths(
        query: String,
        maxPaths: Int = 3,
    ): List<List<String>> = graphRetriever.retrievePaths(query, maxPaths = maxPaths)
}
