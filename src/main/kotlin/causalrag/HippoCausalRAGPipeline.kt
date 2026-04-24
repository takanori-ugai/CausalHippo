package causalrag

import causalrag.causalgraph.builder.CausalGraphBuilder
import causalrag.causalgraph.retriever.CausalPathRetriever
import causalrag.generator.llm.LLMInterface
import causalrag.generator.promptbuilder.buildPrompt
import causalrag.reranker.CausalPathReranker
import causalrag.retriever.Bm25Retriever
import causalrag.retriever.HippoRagSemanticMode
import causalrag.retriever.HippoRagSemanticRetrieverAdapter
import causalrag.retriever.HybridRetriever
import hipporag.HippoRag
import hipporag.config.BaseConfig
import kotlinx.serialization.json.Json
import shared.config.CommonRagConfigLoader
import java.nio.file.Files
import java.nio.file.Path

/**
 * Two-stage QA pipeline that uses HippoRAG for broad candidate recall and CausalRAG for causal validation.
 *
 * The design keeps HippoRAG's graph-assisted semantic recall while relying on the causal graph to rerank
 * and explain answers for causality-sensitive questions.
 */
@Suppress("TooGenericExceptionCaught")
class HippoCausalRAGPipeline(
    modelName: String = "gpt-4o-mini",
    embeddingModel: String = "text-embedding-3-small",
    configPath: String? = null,
    templateStyle: String? = null,
    hippoConfig: BaseConfig? = null,
    hippoSemanticMode: HippoRagSemanticMode = HippoRagSemanticMode.GRAPH,
    semanticWeight: Double = 0.4,
    causalWeight: Double = 0.5,
    bm25Weight: Double = 0.1,
    minCausalMatches: Int = 0,
    dynamicWeightingEnabled: Boolean = false,
    twoPassAdaptiveEnabled: Boolean = false,
    confidenceBasedSwitchEnabled: Boolean = false,
) {
    private val config: PipelineConfig? = configPath?.let { loadConfig(it) }
    private val effectiveModelName = config?.modelName ?: modelName
    private val effectiveEmbeddingModel = config?.embeddingModel ?: embeddingModel
    private val effectiveGraphPath = config?.graphPath
    private val effectiveLlmProvider = config?.llmProvider ?: "openai"
    private val effectiveLlmApiKey = config?.llmApiKey ?: System.getenv("OPENAI_API_KEY")
    private val effectiveLlmBaseUrl = config?.llmBaseUrl
    private val effectiveEmbeddingApiKey = config?.embeddingApiKey ?: System.getenv("OPENAI_API_KEY")
    private val effectiveTemplateStyle = templateStyle ?: config?.templateStyle ?: "detailed"
    private val effectiveHippoSemanticMode = config?.semanticMode?.let { parseSemanticMode(it) } ?: hippoSemanticMode
    private val effectiveMinCausalMatches = config?.minCausalMatches ?: minCausalMatches
    private var indexedDocs: List<String> = emptyList()

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
    internal val hippoRag: HippoRag = HippoRag(resolveHippoConfig(hippoConfig))
    internal val semanticRetriever = HippoRagSemanticRetrieverAdapter(hippoRag, mode = effectiveHippoSemanticMode)
    internal val bm25Retriever: Bm25Retriever = Bm25Retriever()
    internal val graphRetriever: CausalPathRetriever = CausalPathRetriever(graphBuilder)
    internal val hybridRetriever: HybridRetriever =
        HybridRetriever(
            semanticRetriever = semanticRetriever,
            graphRetriever = graphRetriever,
            semanticWeight = semanticWeight,
            causalWeight = causalWeight,
            bm25Weight = bm25Weight,
            minCausalMatches = effectiveMinCausalMatches,
            bm25Retriever = bm25Retriever,
            dynamicWeightingEnabled = dynamicWeightingEnabled,
            twoPassAdaptiveEnabled = twoPassAdaptiveEnabled,
            confidenceBasedSwitchEnabled = confidenceBasedSwitchEnabled,
        )
    internal val reranker: CausalPathReranker = CausalPathReranker(graphRetriever)

    /**
     * Indexes documents into both HippoRAG and the causal graph.
     */
    fun index(documents: List<String>) {
        val cleaned = documents.filter { it.isNotBlank() }
        if (cleaned.isEmpty()) return
        hippoRag.index(cleaned)
        graphBuilder.indexDocuments(cleaned)
        bm25Retriever.indexDocuments(cleaned)
        indexedDocs = cleaned
    }

    /**
     * Replaces the current in-memory indexes with [documents].
     *
     * This preserves heavyweight clients while preventing cross-sample index contamination.
     */
    fun reindex(documents: List<String>) {
        val cleaned = documents.filter { it.isNotBlank() }
        hybridRetriever.clearCache()
        if (indexedDocs.isNotEmpty()) {
            hippoRag.delete(indexedDocs)
        }
        indexedDocs = emptyList()
        graphBuilder.clear()
        bm25Retriever.clear()
        if (cleaned.isEmpty()) return

        try {
            hippoRag.index(cleaned)
            graphBuilder.indexDocuments(cleaned)
            bm25Retriever.indexDocuments(cleaned)
            indexedDocs = cleaned
        } catch (e: Exception) {
            // Best-effort rollback to preserve per-sample index isolation after partial failures.
            runCatching { hippoRag.delete(cleaned) }
            runCatching { graphBuilder.clear() }
            runCatching { bm25Retriever.clear() }
            indexedDocs = emptyList()
            throw e
        }
    }

    /**
     * Retrieves supporting passages after HippoRAG recall and causal reranking.
     */
    fun retrieveContext(
        query: String,
        topK: Int = 5,
    ): List<String> {
        val candidateDetails = hybridRetriever.retrieveWithDetails(query, topK = topK)
        val candidates = candidateDetails.map { it["passage"] as String }
        val metadata = candidateDetails.map { mapOf("score" to (it["score"] as Double)) }
        return reranker.rerank(query, candidates, metadata).map { it.first }.take(topK)
    }

    /**
     * Retrieves causal paths relevant to the query.
     */
    fun retrieveCausalPaths(
        query: String,
        maxPaths: Int = 3,
    ): List<List<String>> = graphRetriever.retrievePaths(query, maxPaths = maxPaths)

    /**
     * Runs retrieval, causal reranking, prompt construction, and answer generation.
     */
    fun runWithContext(
        query: String,
        topK: Int = 5,
    ): PipelineRunResult {
        val candidateDetails = hybridRetriever.retrieveWithDetails(query, topK = topK)
        val candidates = candidateDetails.map { it["passage"] as String }
        val metadata = candidateDetails.map { mapOf("score" to (it["score"] as Double)) }
        val rerankedPassages = reranker.rerank(query, candidates, metadata).map { it.first }.take(topK)
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
        val answer = llm.generate(prompt, jsonMode = requiresJsonResponseFormat(effectiveTemplateStyle))
        return PipelineRunResult(answer, rerankedPassages, causalPaths)
    }

    /**
     * Convenience wrapper around [runWithContext].
     */
    fun run(
        query: String,
        topK: Int = 5,
    ): String = runWithContext(query, topK).answer

    private fun loadConfig(configPath: String): PipelineConfig {
        val path = Path.of(configPath)
        require(Files.exists(path)) { "Config file not found: $configPath" }
        val content = Files.readString(path)
        return CommonRagConfigLoader.parseOrNull(content)?.toPipelineConfig()
            ?: Json { ignoreUnknownKeys = true }.decodeFromString(PipelineConfig.serializer(), content)
    }

    private fun resolveHippoConfig(initial: BaseConfig?): BaseConfig =
        (initial ?: BaseConfig()).copy().apply {
            llmName = effectiveModelName
            embeddingModelName = effectiveEmbeddingModel
            llmProvider = normalizeHippoProvider(llmProvider, effectiveLlmProvider)
            embeddingProvider = normalizeHippoProvider(embeddingProvider, effectiveLlmProvider)
            if (openAiApiKey.isNullOrBlank()) {
                openAiApiKey = effectiveLlmApiKey ?: effectiveEmbeddingApiKey
            }
            if (llmBaseUrl == null) {
                llmBaseUrl = effectiveLlmBaseUrl
            }
            if (embeddingBaseUrl == null) {
                embeddingBaseUrl = effectiveLlmBaseUrl
            }
            if (saveDir == BaseConfig().saveDir) {
                saveDir = "outputs/hippo-causal"
            }
        }

    private fun normalizeHippoProvider(
        current: String?,
        fallback: String,
    ): String =
        current?.lowercase()
            ?: when (fallback.lowercase()) {
                "openai", "ollama" -> fallback.lowercase()
                else -> "openai"
            }

    private fun parseSemanticMode(raw: String): HippoRagSemanticMode =
        when (raw.lowercase()) {
            "graph" -> HippoRagSemanticMode.GRAPH
            "dpr" -> HippoRagSemanticMode.DPR
            else -> throw IllegalArgumentException("Invalid semanticMode '$raw'. Expected 'graph' or 'dpr'.")
        }

    private fun requiresJsonResponseFormat(style: String): Boolean = style.equals("experiments", ignoreCase = true)
}
