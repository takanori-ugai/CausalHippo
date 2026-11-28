package causalhippo

import causalrag.CausalRagRunResult
import causalrag.causalgraph.builder.CausalGraphBuilder
import causalrag.causalgraph.retriever.CausalPathRetriever
import causalrag.generator.llm.LLMInterface
import causalrag.generator.promptbuilder.PromptBuilder
import causalrag.retriever.Bm25Retriever
import causalrag.retriever.HippoRagSemanticMode
import causalrag.retriever.HippoRagSemanticRetrieverAdapter
import causalrag.retriever.HybridRetriever
import hipporag.HippoRag
import hipporag.config.BaseConfig
import kotlinx.serialization.json.Json
import shared.chunking.DEFAULT_INGEST_CHUNK_OVERLAP_TOKEN_SIZE
import shared.chunking.DEFAULT_INGEST_CHUNK_TOKEN_SIZE
import shared.chunking.DEFAULT_PROMPT_CONTEXT_TOKEN_BUDGET
import shared.chunking.DEFAULT_TIKTOKEN_MODEL
import shared.chunking.chunkByTokenSizeWithOverlap
import shared.config.CaualRagConfig
import shared.config.CommonRagConfigLoader
import shared.rag.IndexingReport
import java.nio.file.Files
import java.nio.file.Path

private val hippoCaualRagConfigJson = Json { ignoreUnknownKeys = true }

/**
 * Two-stage QA pipeline that uses HippoRAG for broad candidate recall and CausalRAG for causal validation.
 *
 * The design keeps HippoRAG's graph-assisted semantic recall while relying on the causal graph to rerank
 * and explain answers for causality-sensitive questions.
 */
@Suppress("TooGenericExceptionCaught")
class CausalHippoPipeline(
    modelName: String = "gpt-4o-mini",
    embeddingModel: String = "text-embedding-3-small",
    configPath: String? = null,
    templateStyle: String? = null,
    hippoConfig: BaseConfig? = null,
    hippoSemanticMode: HippoRagSemanticMode = HippoRagSemanticMode.GRAPH,
    private val semanticWeight: Double = 0.4,
    private val causalWeight: Double = 0.5,
    private val bm25Weight: Double = 0.1,
    minCausalMatches: Int = 0,
    private val dynamicWeightingEnabled: Boolean = true,
    private val twoPassAdaptiveEnabled: Boolean = true,
    private val confidenceBasedSwitchEnabled: Boolean = true,
) {
    private val config: CaualRagConfig? = configPath?.let { loadConfig(it) }
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
    private val effectiveIngestChunkTokenSize = config?.ingestChunkTokenSize ?: DEFAULT_INGEST_CHUNK_TOKEN_SIZE
    private val effectiveIngestChunkOverlapTokenSize =
        config?.ingestChunkOverlapTokenSize ?: DEFAULT_INGEST_CHUNK_OVERLAP_TOKEN_SIZE
    private val effectivePromptContextTokenBudget =
        config?.promptContextTokenBudget ?: DEFAULT_PROMPT_CONTEXT_TOKEN_BUDGET
    private val liveFiles = HybridIndexFiles(resolveHippoConfig(hippoConfig))
    private val promptBuilder by lazy { PromptBuilder(templateStyle = effectiveTemplateStyle, llmInterface = llm) }

    internal val llm: LLMInterface =
        LLMInterface(
            modelName = effectiveModelName,
            provider = effectiveLlmProvider,
            apiKey = effectiveLlmApiKey,
            baseUrl = effectiveLlmBaseUrl,
        )

    private data class Index(
        val hippo: HippoRag,
        val graph: CausalGraphBuilder,
        val bm25: Bm25Retriever,
        val paths: CausalPathRetriever,
        val hybrid: HybridRetriever,
    )

    private fun createGraph(): CausalGraphBuilder =
        CausalGraphBuilder(
            modelName = effectiveEmbeddingModel,
            embeddingApiKey = effectiveEmbeddingApiKey,
            extractorMethod = "hybrid",
            llmInterface = llm,
            ingestChunkTokenSize = effectiveIngestChunkTokenSize,
            ingestChunkOverlapTokenSize = effectiveIngestChunkOverlapTokenSize,
        )

    private fun readGraph(path: Path?): CausalGraphBuilder =
        createGraph().also { builder ->
            if (path != null) check(builder.load(path.toString())) { "Could not load causal graph from '$path'" }
        }

    private fun createIndex(
        hippo: HippoRag,
        graph: CausalGraphBuilder,
    ): Index {
        val bm25 = Bm25Retriever()
        val passages = hippo.indexedPassages()
        if (passages.isNotEmpty()) bm25.indexDocuments(passages)
        val paths = CausalPathRetriever(graph)
        val hybrid =
            HybridRetriever(
                semanticRetriever = HippoRagSemanticRetrieverAdapter(hippo, mode = effectiveHippoSemanticMode),
                graphRetriever = paths,
                semanticWeight = semanticWeight,
                causalWeight = causalWeight,
                bm25Weight = bm25Weight,
                minCausalMatches = effectiveMinCausalMatches,
                bm25Retriever = bm25,
                dynamicWeightingEnabled = dynamicWeightingEnabled,
                twoPassAdaptiveEnabled = twoPassAdaptiveEnabled,
                confidenceBasedSwitchEnabled = confidenceBasedSwitchEnabled,
                pathEvidenceMatcher = ::supportsDirectedRelation,
            )
        return Index(hippo, graph, bm25, paths, hybrid)
    }

    private var current =
        createIndex(
            HippoRag(liveFiles.config),
            readGraph(liveFiles.causalGraph.takeIf(Files::isRegularFile) ?: effectiveGraphPath?.let(Path::of)),
        )

    internal val hippoRag get() = current.hippo
    internal val graphBuilder get() = current.graph
    internal val graphRetriever get() = current.paths
    internal val bm25Retriever get() = current.bm25
    internal val hybridRetriever get() = current.hybrid

    var lastIndexingReport: IndexingReport = IndexingReport()
        private set

    /** Adds chunks to a staged copy of the current corpus. */
    @Synchronized
    fun index(documents: List<String>) {
        val prepared = chunkDocumentsForIngest(documents).distinct()
        if (prepared.isNotEmpty()) updateIndex(prepared, replace = false)
    }

    /** Replaces the complete corpus after all three retrieval indexes can be built. */
    @Synchronized
    fun reindex(documents: List<String>) {
        updateIndex(chunkDocumentsForIngest(documents).distinct(), replace = true)
    }

    private fun updateIndex(
        prepared: List<String>,
        replace: Boolean,
    ) {
        liveFiles.withStaging { staged ->
            if (!replace) liveFiles.copyTo(staged)
            val stagedHippo = HippoRag(staged.config)
            val oldPassages = stagedHippo.indexedPassages().toSet()
            val hasCausalGraph = Files.isRegularFile(staged.causalGraph)
            val graph = readGraph(staged.causalGraph.takeIf { hasCausalGraph })
            stagedHippo.indexPreparedChunks(prepared)
            val allPassages = stagedHippo.indexedPassages()
            check(allPassages.toSet() == oldPassages + prepared) { "HippoRAG did not index the complete corpus" }
            val causalInputs = allPassages.filterNot(graph::hasCompletedPassage)
            graph.indexDocuments(causalInputs)
            check(graph.save(staged.causalGraph.toString())) { "Could not stage causal graph" }
            val report = IndexingReport(graph.lastIndexingReport.failures + stagedHippo.lastIndexingReport.failures)
            publish(staged, graph)
            lastIndexingReport = report
        }
    }

    private fun publish(
        staged: HybridIndexFiles,
        graph: CausalGraphBuilder,
    ) {
        liveFiles.install(staged) {
            current = createIndex(HippoRag(liveFiles.config), graph)
        }
    }

    @Synchronized
    internal fun saveSnapshot(target: Path) {
        liveFiles.saveSnapshot(target, graphBuilder)
    }

    @Synchronized
    internal fun loadSnapshot(source: Path) {
        liveFiles.withStaging { staged ->
            liveFiles.readSnapshot(source, staged)
            val graph = readGraph(staged.causalGraph)
            publish(staged, graph)
            lastIndexingReport = IndexingReport()
        }
    }

    @Synchronized
    fun retrieveContext(
        query: String,
        topK: Int = 5,
    ): List<String> = preparePrompt(query, topK, maxPaths = 3, summarize = false).passages

    @Synchronized
    fun retrieveCausalPaths(
        query: String,
        maxPaths: Int = 3,
    ): List<List<String>> = current.paths.retrievePaths(query, maxPaths = maxPaths.coerceAtLeast(1))

    @Synchronized
    internal fun preparePrompt(
        query: String,
        topK: Int,
        maxPaths: Int,
        summarize: Boolean,
    ): PromptBuilder.BudgetedPrompt =
        promptBuilder.buildBudgetedPrompt(
            query,
            current.hybrid.retrieve(query, topK = topK.coerceAtLeast(1)),
            retrieveCausalPaths(query, maxPaths),
            effectivePromptContextTokenBudget,
            summarize,
        )

    @Synchronized
    fun runWithContext(
        query: String,
        topK: Int = 5,
        maxPaths: Int = 3,
    ): CausalRagRunResult {
        val evidence = preparePrompt(query, topK, maxPaths, summarize = true)
        val answer = llm.generate(evidence.text, jsonMode = requiresJsonResponseFormat(effectiveTemplateStyle))
        return CausalRagRunResult(answer, evidence.passages, evidence.paths)
    }

    fun run(
        query: String,
        topK: Int = 5,
    ): String = runWithContext(query, topK).answer

    private fun loadConfig(configPath: String): CaualRagConfig {
        val path = Path.of(configPath)
        require(Files.exists(path)) { "Config file not found: $configPath" }
        val content = Files.readString(path)
        return CommonRagConfigLoader.parseOrNull(content)?.toCaualRagConfig()
            ?: hippoCaualRagConfigJson.decodeFromString(CaualRagConfig.serializer(), content)
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
                "openai", "ollama", "gemini" -> fallback.lowercase()
                else -> "openai"
            }

    private fun parseSemanticMode(raw: String): HippoRagSemanticMode =
        when (raw.lowercase()) {
            "graph" -> HippoRagSemanticMode.GRAPH
            "dpr" -> HippoRagSemanticMode.DPR
            else -> throw IllegalArgumentException("Invalid semanticMode '$raw'. Expected 'graph' or 'dpr'.")
        }

    private fun chunkDocumentsForIngest(documents: List<String>): List<String> =
        documents
            .asSequence()
            .filter { it.isNotBlank() }
            .flatMap { doc ->
                chunkByTokenSizeWithOverlap(
                    content = doc,
                    chunkTokenSize = effectiveIngestChunkTokenSize,
                    chunkOverlapTokenSize = effectiveIngestChunkOverlapTokenSize,
                    model = DEFAULT_TIKTOKEN_MODEL,
                ).asSequence()
            }.map { it.content }
            .filter { it.isNotBlank() }
            .toList()

    private fun requiresJsonResponseFormat(style: String): Boolean = style.equals("experiments", ignoreCase = true)
}

@Deprecated(
    message = "Use CausalHippoPipeline",
    replaceWith = ReplaceWith("CausalHippoPipeline"),
)
typealias HippoCausalRAGPipeline = CausalHippoPipeline
