package com.youtu.graphrag.shared.retriever

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.graph.GraphNode
import com.youtu.graphrag.shared.graph.GraphRelationship
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.readText
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class KTRetrieverTest {
    private val mapper = ObjectMapper().registerKotlinModule()

    @Test
    fun `generatePrompt uses general retrieval prompt for default datasets`() {
        val config = ConfigManager("config/base_config.json")
        val retriever = createRetriever(datasetName = "demo", config = config)

        val prompt = retriever.generatePrompt(question = "Who directed the film?", context = "some context")

        assertTrue(prompt.contains("You are an expert knowledge assistant"))
        assertTrue(prompt.contains("Who directed the film?"))
    }

    @Test
    fun `generatePrompt uses chinese novel prompt for anony_chs dataset`() {
        val config = ConfigManager("config/base_config.json")
        val retriever = createRetriever(datasetName = "anony_chs", config = config)

        val prompt = retriever.generatePrompt(question = "PERSON#1是谁？", context = "小说知识")

        assertTrue(prompt.contains("你是小说知识助手"))
        assertTrue(prompt.contains("PERSON#1是谁？"))
    }

    @Test
    fun `generatePrompt uses english novel prompt for anony_eng dataset`() {
        val config = ConfigManager("config/base_config.json")
        val retriever = createRetriever(datasetName = "anony_eng", config = config)

        val prompt = retriever.generatePrompt(question = "Who is PERSON#1?", context = "novel context")

        assertTrue(prompt.contains("You are a novel knowledge assistant"))
        assertTrue(prompt.contains("Who is PERSON#1?"))
    }

    @Test
    fun `generatePrompt supports python prompt alias retrieval novel_chs`() {
        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
                "prompts" to
                    mapOf(
                        "retrieval" to
                            mapOf(
                                "general" to "GENERAL::{question}::{context}",
                                "novel_chs" to "NOVEL_CHS::{question}::{context}",
                            ),
                    ),
            ),
        )

        val retriever = createRetriever(datasetName = "anony_chs", config = config)
        val prompt = retriever.generatePrompt(question = "PERSON#1是谁？", context = "小说知识上下文")

        assertTrue(prompt.startsWith("NOVEL_CHS::"))
        assertTrue(prompt.contains("PERSON#1是谁？"))
        assertTrue(prompt.contains("小说知识上下文"))
    }

    @Test
    fun `generateIrcotPrompt selects source-specific template keys`() {
        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
                "prompts" to
                    mapOf(
                        "retrieval" to
                            mapOf(
                                "general" to "GENERAL::{question}::{context}",
                                "ircot" to "IRCOT::{current_query}::{step}",
                                "ircot_backend" to "BACKEND::{original_question}::{current_iteration_query}::{step}",
                                "ircot_main" to "MAIN::{current_query}::{step}",
                            ),
                    ),
            ),
        )

        val retriever = createRetriever(datasetName = "demo", config = config)

        val backendPrompt =
            retriever.generateIrcotPrompt(
                currentQuery = "Where is Project Alpha based?",
                originalQuestion = "Where is Project Alpha based?",
                context = "ctx",
                previousThoughts = "none",
                step = 2,
                promptSource = IrcotPromptSource.BACKEND,
            )
        val mainPrompt =
            retriever.generateIrcotPrompt(
                currentQuery = "Where is Project Alpha based?",
                originalQuestion = "Where is Project Alpha based?",
                context = "ctx",
                previousThoughts = "none",
                step = 3,
                promptSource = IrcotPromptSource.MAIN,
            )

        assertTrue(backendPrompt.startsWith("BACKEND::"))
        assertTrue("::2" in backendPrompt)
        assertTrue(mainPrompt.startsWith("MAIN::"))
        assertTrue("::3" in mainPrompt)
    }

    @Test
    fun `generateIrcotPrompt uses shared ircot for backend and python-compatible inline fallback for main`() {
        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
                "prompts" to
                    mapOf(
                        "retrieval" to
                            mapOf(
                                "general" to "GENERAL::{question}::{context}",
                                "ircot" to "SHARED::{current_query}::{step}",
                            ),
                    ),
            ),
        )

        val retriever = createRetriever(datasetName = "demo", config = config)
        val backendPrompt =
            retriever.generateIrcotPrompt(
                currentQuery = "current query",
                originalQuestion = "original question",
                context = "ctx",
                previousThoughts = "thought",
                step = 1,
                promptSource = IrcotPromptSource.BACKEND,
            )
        val mainPrompt =
            retriever.generateIrcotPrompt(
                currentQuery = "current query",
                originalQuestion = "original question",
                context = "ctx",
                previousThoughts = "thought",
                step = 1,
                promptSource = IrcotPromptSource.MAIN,
            )

        assertTrue(backendPrompt.startsWith("SHARED::"))
        assertTrue(mainPrompt.contains("You are an expert knowledge assistant using iterative retrieval with chain-of-thought reasoning."))
        assertTrue(mainPrompt.contains("Consider the initial analysis from noagent mode"))
        assertTrue(mainPrompt.contains("Build upon the initial analysis to provide deeper insights"))
        assertTrue("current query" in backendPrompt)
        assertTrue(mainPrompt.contains("Step 1:"))
    }

    @Test
    fun `enhanceQuery appends entities and key terms in python-compatible format`() {
        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
                "nlp" to
                    mapOf(
                        "provider" to "regex",
                        "stopwords" to listOf("where", "does", "at", "the"),
                    ),
            ),
        )

        val retriever = createRetriever(datasetName = "demo", config = config)
        val question = "Where does Alice work at ACME Corp?"
        val enhanced = retriever.enhanceQuery(question)

        assertTrue(enhanced.startsWith(question))
        assertTrue("Entities:" in enhanced)
        assertTrue("Key terms:" in enhanced)
    }

    @Test
    fun `extractQueryKeywords uses stopword filtering and entity retention`() {
        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
                "nlp" to
                    mapOf(
                        "provider" to "regex",
                        "stopwords" to listOf("what", "is", "the", "in"),
                    ),
            ),
        )

        val retriever = createRetriever(datasetName = "demo", config = config)
        val keywords = retriever.extractQueryKeywords("What is the capital city in France?")

        assertTrue("capital" in keywords)
        assertTrue("city" in keywords)
        assertTrue("france" in keywords)
        assertTrue("what" !in keywords)
        assertTrue("the" !in keywords)
    }

    @Test
    fun `processRetrievalResults returns python-compatible payload and retrieval time`() {
        val config = ConfigManager("config/base_config.json")
        val retriever = createRetriever(datasetName = "demo", config = config)
        val question = "Who leads OpenAI in San Francisco?"

        val (results, retrievalTime) = retriever.processRetrievalResults(question = question)
        val triples = (results["triples"] as? List<*>)?.map { it.toString() }.orEmpty()
        val chunkIds = (results["chunk_ids"] as? List<*>)?.map { it.toString() }.orEmpty()
        val chunkContents = (results["chunk_contents"] as? List<*>)?.map { it.toString() }.orEmpty()
        val chunkRetrievalResults = (results["chunk_retrieval_results"] as? List<*>)?.map { it.toString() }.orEmpty()

        assertTrue(retrievalTime >= 0.0)
        assertTrue(triples.isEmpty() || triples.all { triple -> triple.isNotBlank() })
        assertEquals(chunkIds.size, chunkContents.size)
        assertEquals(chunkIds.size, chunkRetrievalResults.size)
        assertTrue(chunkRetrievalResults.all { line -> line.startsWith("[Chunk ") })
        assertEquals(setOf("triples", "chunk_ids", "chunk_contents", "chunk_retrieval_results"), results.keys)
    }

    @Test
    fun `opennlp provider falls back to regex when models are unavailable`() {
        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
                "nlp" to
                    mapOf(
                        "provider" to "opennlp",
                    ),
            ),
        )

        val retriever = createRetriever(datasetName = "demo", config = config)
        val keywords = retriever.extractQueryKeywords("Who founded OpenAI in 2015?")

        assertTrue("openai" in keywords)
        assertTrue(keywords.isNotEmpty())
    }

    @Test
    fun `processRetrievalResults prioritizes involved relation types`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-filter-test")
        val config = createTestConfig(root)
        val datasetName = "filter_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "Alice",
                        sourceLabel = "person",
                        relation = "works_at",
                        targetName = "ACME",
                        targetLabel = "organization",
                        chunkId = "c1",
                    ),
                    relationship(
                        sourceName = "Alice",
                        sourceLabel = "person",
                        relation = "lives_in",
                        targetName = "Tokyo",
                        targetLabel = "location",
                        chunkId = "c2",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries =
                listOf(
                    "c1" to "Alice works at ACME as an engineer.",
                    "c2" to "Alice lives in Tokyo.",
                ),
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
                topK = 3,
                recallPaths = 2,
            )

        val (results, retrievalTime) =
            retriever.processRetrievalResults(
                question = "Where does Alice work?",
                involvedTypes =
                    mapOf(
                        "nodes" to listOf("person"),
                        "relations" to listOf("works_at"),
                        "attributes" to emptyList(),
                    ),
            )

        val triples =
            (results["triples"] as? List<*>)?.map { value -> value.toString() }
                ?: error("Expected 'triples' list in results")
        val chunkIds =
            (results["chunk_ids"] as? List<*>)?.map { value -> value.toString() }
                ?: error("Expected 'chunk_ids' list in results")

        assertTrue(retrievalTime >= 0.0)
        assertTrue(triples.isNotEmpty())
        assertTrue(triples.first().contains("works_at"))
        assertTrue(triples.any { triple -> "works_at" in triple })
        assertTrue("c1" in chunkIds)
    }

    @Test
    fun `processRetrievalResults expands to neighbor triples based on recallPaths`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-path-test")
        val config = createTestConfig(root)
        val datasetName = "path_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "Alpha",
                        sourceLabel = "entity",
                        relation = "linked_to",
                        targetName = "Beta",
                        targetLabel = "entity",
                        chunkId = "c1",
                    ),
                    relationship(
                        sourceName = "Beta",
                        sourceLabel = "entity",
                        relation = "connected_to",
                        targetName = "Gamma",
                        targetLabel = "entity",
                        chunkId = "c2",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries =
                listOf(
                    "c1" to "Alpha is directly linked to Beta.",
                    "c2" to "Beta is connected to Gamma.",
                ),
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
                topK = 3,
                recallPaths = 2,
            )

        val (results, retrievalTime) = retriever.processRetrievalResults(question = "What is related to Alpha?")
        val triples =
            (results["triples"] as? List<*>)?.map { value -> value.toString() }
                ?: error("Expected 'triples' list in results")

        assertTrue(retrievalTime >= 0.0)
        assertTrue(triples.any { triple -> "linked_to" in triple })
        assertTrue(triples.any { triple -> "connected_to" in triple })
    }

    @Test
    fun `processRetrievalResults merges triple chunk ids with ranked chunk hits`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-chunk-merge-test")
        val config = createTestConfig(root)
        val datasetName = "chunk_merge_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "Delta",
                        sourceLabel = "entity",
                        relation = "mentions",
                        targetName = "Echo",
                        targetLabel = "entity",
                        chunkId = "c1",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries =
                listOf(
                    "c1" to "Delta and Echo are linked in the source text.",
                    "c2" to "Beta appears frequently in this standalone chunk.",
                ),
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
                topK = 2,
                recallPaths = 1,
            )

        val (results, retrievalTime) = retriever.processRetrievalResults(question = "Give Beta details")
        val chunkIds =
            (results["chunk_ids"] as? List<*>)?.map { value -> value.toString() }
                ?: error("Expected 'chunk_ids' list")
        val chunkContents =
            (results["chunk_contents"] as? List<*>)?.map { value -> value.toString() }
                ?: error("Expected 'chunk_contents' list")
        val chunkRetrievalResults =
            (results["chunk_retrieval_results"] as? List<*>)?.map { value -> value.toString() }
                ?: error("Expected 'chunk_retrieval_results' list")

        assertTrue(retrievalTime >= 0.0)
        assertEquals(2, chunkIds.size)
        assertTrue("c1" in chunkIds)
        assertTrue("c2" in chunkIds)
        assertEquals(2, chunkContents.size)
        assertTrue(chunkContents.any { value -> "Delta" in value })
        assertTrue(chunkContents.any { value -> "Beta" in value })
        assertEquals(2, chunkRetrievalResults.size)
        assertTrue(chunkRetrievalResults.all { line -> line.startsWith("[Chunk ") })
    }

    @Test
    fun `processRetrievalResults keeps payload stable when reranking toggles`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-strategy-test")
        val config = createTestConfig(root)
        val datasetName = "strategy_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "NodeA",
                        sourceLabel = "entity",
                        relation = "related_to",
                        targetName = "NodeB",
                        targetLabel = "entity",
                        chunkId = "c1",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries = listOf("c1" to "NodeA related to NodeB."),
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
                topK = 2,
            )

        val (results, hybridTime) = retriever.processRetrievalResults(question = "How is NodeA connected?")
        assertTrue(hybridTime >= 0.0)
        assertEquals(setOf("triples", "chunk_ids", "chunk_contents", "chunk_retrieval_results"), results.keys)

        config.overrideConfig(mapOf("retrieval" to mapOf("enable_reranking" to false)))
        val (lexicalResults, lexicalTime) = retriever.processRetrievalResults(question = "How is NodeA connected?")
        assertTrue(lexicalTime >= 0.0)
        assertEquals(setOf("triples", "chunk_ids", "chunk_contents", "chunk_retrieval_results"), lexicalResults.keys)
    }

    @Test
    fun `processRetrievalResults keeps ranking stable between parallel and sequential strategy execution`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-parallel-stability-test")
        val config = createTestConfig(root)
        val datasetName = "parallel_stability_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "Alpha",
                        sourceLabel = "entity",
                        relation = "works_with",
                        targetName = "Beta",
                        targetLabel = "entity",
                        chunkId = "c1",
                    ),
                    relationship(
                        sourceName = "Beta",
                        sourceLabel = "entity",
                        relation = "located_in",
                        targetName = "Tokyo",
                        targetLabel = "location",
                        chunkId = "c2",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries =
                listOf(
                    "c1" to "Alpha works with Beta.",
                    "c2" to "Beta is located in Tokyo.",
                ),
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
                topK = 2,
                recallPaths = 2,
            )

        config.overrideConfig(
            mapOf(
                "retrieval" to
                    mapOf(
                        "strategy" to
                            mapOf(
                                "enable_parallel" to true,
                            ),
                    ),
            ),
        )
        val parallelResult = retriever.processRetrievalResults(question = "How is Alpha related to Tokyo?").first

        config.overrideConfig(
            mapOf(
                "retrieval" to
                    mapOf(
                        "strategy" to
                            mapOf(
                                "enable_parallel" to false,
                            ),
                    ),
            ),
        )
        val sequentialResult = retriever.processRetrievalResults(question = "How is Alpha related to Tokyo?").first

        assertEquals(parallelResult["triples"], sequentialResult["triples"])
        assertEquals(parallelResult["chunk_ids"], sequentialResult["chunk_ids"])
    }

    @Test
    fun `processRetrievalResults honors strategy enable-disable configuration`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-strategy-enable-disable-test")
        val config = createTestConfig(root)
        val datasetName = "strategy_toggle_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "Alpha",
                        sourceLabel = "entity",
                        relation = "related_to",
                        targetName = "Beta",
                        targetLabel = "entity",
                        chunkId = "c1",
                    ),
                    relationship(
                        sourceName = "Beta",
                        sourceLabel = "entity",
                        relation = "connected_to",
                        targetName = "Gamma",
                        targetLabel = "entity",
                        chunkId = "c2",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries =
                listOf(
                    "c1" to "Alpha related to Beta.",
                    "c2" to "Beta connected to Gamma.",
                ),
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
                topK = 2,
                recallPaths = 2,
            )

        config.overrideConfig(
            mapOf(
                "retrieval" to
                    mapOf(
                        "strategy" to
                            mapOf(
                                "enabled" to listOf("lexical_triple", "lexical_chunk"),
                            ),
                    ),
            ),
        )
        val noPathExpandTriples =
            (
                (retriever.processRetrievalResults(question = "What is related to Alpha?").first["triples"] as? List<*>)
                    ?.map { it.toString() }
            ).orEmpty()

        config.overrideConfig(
            mapOf(
                "retrieval" to
                    mapOf(
                        "strategy" to
                            mapOf(
                                "enabled" to listOf("lexical_triple", "path_expand", "lexical_chunk"),
                            ),
                    ),
            ),
        )
        val withPathExpandTriples =
            (
                (retriever.processRetrievalResults(question = "What is related to Alpha?").first["triples"] as? List<*>)
                    ?.map { it.toString() }
            ).orEmpty()

        assertTrue(noPathExpandTriples.any { triple -> "related_to" in triple })
        assertTrue(noPathExpandTriples.none { triple -> "connected_to" in triple })
        assertTrue(withPathExpandTriples.any { triple -> "connected_to" in triple })
    }

    @Test
    fun `processRetrievalResults keeps deterministic tie-break ordering`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-deterministic-tiebreak-test")
        val config = createTestConfig(root)
        val datasetName = "deterministic_tiebreak_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "OrderA",
                        sourceLabel = "entity",
                        relation = "aligned_with",
                        targetName = "OrderB",
                        targetLabel = "entity",
                        chunkId = "c1",
                    ),
                    relationship(
                        sourceName = "OrderC",
                        sourceLabel = "entity",
                        relation = "aligned_with",
                        targetName = "OrderD",
                        targetLabel = "entity",
                        chunkId = "c2",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries =
                listOf(
                    "c1" to "OrderA aligned with OrderB.",
                    "c2" to "OrderC aligned with OrderD.",
                ),
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
                topK = 2,
                recallPaths = 1,
            )

        config.overrideConfig(
            mapOf(
                "retrieval" to
                    mapOf(
                        "strategy" to
                            mapOf(
                                "enabled" to listOf("lexical_triple", "lexical_chunk"),
                            ),
                    ),
            ),
        )

        val firstRun =
            (
                (retriever.processRetrievalResults(question = "aligned_with").first["triples"] as? List<*>)
                    ?.map { it.toString() }
            ).orEmpty()
        val secondRun =
            (
                (retriever.processRetrievalResults(question = "aligned_with").first["triples"] as? List<*>)
                    ?.map { it.toString() }
            ).orEmpty()

        assertEquals(firstRun, secondRun)
        assertTrue(firstRun.firstOrNull()?.contains("OrderA") == true)
    }

    @Test
    fun `buildIndices persists embedding caches when caching enabled`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-cache-enabled-test")
        val config = createTestConfig(root)
        val datasetName = "cache_enabled_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "CacheNodeA",
                        sourceLabel = "entity",
                        relation = "relates_to",
                        targetName = "CacheNodeB",
                        targetLabel = "entity",
                        chunkId = "c1",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries = listOf("c1" to "CacheNodeA relates to CacheNodeB."),
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
            )

        val cacheRoot = root.resolve("retriever/faiss_cache_new").resolve(datasetName)
        assertTrue(cacheRoot.resolve("triple_embedding_cache.json").exists())
        assertTrue(cacheRoot.resolve("triple_embedding_cache.npz").exists())
        assertTrue(cacheRoot.resolve("chunk_embedding_cache.json").exists())
        assertTrue(cacheRoot.resolve("chunk_embedding_cache.npz").exists())
    }

    @Test
    fun `buildIndices skips embedding cache persistence when caching disabled`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-cache-disabled-test")
        val config = createTestConfig(root)
        config.overrideConfig(
            mapOf(
                "retrieval" to
                    mapOf(
                        "enable_caching" to false,
                    ),
            ),
        )
        val datasetName = "cache_disabled_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "NoCacheA",
                        sourceLabel = "entity",
                        relation = "relates_to",
                        targetName = "NoCacheB",
                        targetLabel = "entity",
                        chunkId = "c1",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries = listOf("c1" to "NoCacheA relates to NoCacheB."),
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
            )

        val cacheRoot = root.resolve("retriever/faiss_cache_new").resolve(datasetName)
        assertTrue(!cacheRoot.resolve("triple_embedding_cache.json").exists())
        assertTrue(!cacheRoot.resolve("triple_embedding_cache.npz").exists())
        assertTrue(!cacheRoot.resolve("chunk_embedding_cache.json").exists())
        assertTrue(!cacheRoot.resolve("chunk_embedding_cache.npz").exists())
    }

    @Test
    fun `buildIndices ingests npz embedding caches`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-npz-cache-load-test")
        val config = createTestConfig(root)
        val datasetName = "npz_cache_load_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "PtNodeA",
                        sourceLabel = "entity",
                        relation = "relates_to",
                        targetName = "PtNodeB",
                        targetLabel = "entity",
                        chunkId = "c1",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries = listOf("c1" to "PtNodeA relates to PtNodeB."),
        )

        val cacheRoot = root.resolve("retriever/faiss_cache_new").resolve(datasetName)
        cacheRoot.createDirectories()
        val dims = HashTextEmbedder().dimensions
        val customVector = FloatArray(dims).also { vector -> vector[0] = 1.0f }
        NpzEmbeddingCache.write(
            path = cacheRoot.resolve("triple_embedding_cache.npz"),
            vectorsByKey = mapOf("[\"PtNodeA\", \"relates_to\", \"PtNodeB\"]" to customVector),
            expectedDimensions = dims,
        )
        NpzEmbeddingCache.write(
            path = cacheRoot.resolve("chunk_embedding_cache.npz"),
            vectorsByKey = mapOf("c1" to customVector),
            expectedDimensions = dims,
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
            )

        val tripleCacheTree = mapper.readTree(cacheRoot.resolve("triple_embedding_cache.json").toFile())
        val chunkCacheTree = mapper.readTree(cacheRoot.resolve("chunk_embedding_cache.json").toFile())
        val tripleFirst =
            tripleCacheTree
                .path("vectors")
                .path("[\"PtNodeA\", \"relates_to\", \"PtNodeB\"]")
                .path(0)
                .asDouble()
        val chunkFirst =
            chunkCacheTree
                .path("vectors")
                .path("c1")
                .path(0)
                .asDouble()

        assertEquals(1.0, tripleFirst)
        assertEquals(1.0, chunkFirst)
        assertTrue(cacheRoot.resolve("triple_embedding_cache.npz").exists())
        assertTrue(cacheRoot.resolve("chunk_embedding_cache.npz").exists())
    }

    @Test
    fun `buildIndices evicts stale cache keys`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-stale-cache-eviction-test")
        val config = createTestConfig(root)
        val datasetName = "stale_cache_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "AliveA",
                        sourceLabel = "entity",
                        relation = "maps_to",
                        targetName = "AliveB",
                        targetLabel = "entity",
                        chunkId = "c1",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries = listOf("c1" to "AliveA maps to AliveB."),
        )

        val cacheRoot = root.resolve("retriever/faiss_cache_new").resolve(datasetName)
        cacheRoot.createDirectories()
        val dims = HashTextEmbedder().dimensions
        NpzEmbeddingCache.write(
            path = cacheRoot.resolve("triple_embedding_cache.npz"),
            vectorsByKey =
                mapOf(
                    "[\"AliveA\", \"maps_to\", \"AliveB\"]" to FloatArray(dims) { 0.1f },
                    "[\"STALE_A\", \"stale_rel\", \"STALE_B\"]" to FloatArray(dims) { 0.2f },
                ),
            expectedDimensions = dims,
        )
        NpzEmbeddingCache.write(
            path = cacheRoot.resolve("chunk_embedding_cache.npz"),
            vectorsByKey =
                mapOf(
                    "c1" to FloatArray(dims) { 0.1f },
                    "stale_chunk" to FloatArray(dims) { 0.2f },
                ),
            expectedDimensions = dims,
        )

        createRetriever(
            datasetName = datasetName,
            config = config,
            graphPath = graphPath.toString(),
            rootDir = root,
        )

        val tripleCacheTree = mapper.readTree(cacheRoot.resolve("triple_embedding_cache.json").toFile())
        val chunkCacheTree = mapper.readTree(cacheRoot.resolve("chunk_embedding_cache.json").toFile())
        assertTrue(tripleCacheTree.path("vectors").path("[\"STALE_A\", \"stale_rel\", \"STALE_B\"]").isMissingNode)
        assertTrue(chunkCacheTree.path("vectors").path("stale_chunk").isMissingNode)
    }

    @Test
    fun `buildIndices rebuilds cache on metadata mismatch`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-cache-metadata-mismatch-test")
        val config = createTestConfig(root)
        val datasetName = "cache_mismatch_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "MetaA",
                        sourceLabel = "entity",
                        relation = "links_to",
                        targetName = "MetaB",
                        targetLabel = "entity",
                        chunkId = "c1",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries = listOf("c1" to "MetaA links to MetaB."),
        )

        val cacheRoot = root.resolve("retriever/faiss_cache_new").resolve(datasetName)
        cacheRoot.createDirectories()
        cacheRoot.resolve("triple_embedding_cache.json").writeText(
            """
            {
              "modelTag": "stale-model",
              "dimensions": 3,
              "vectors": {
                "[\"MetaA\", \"links_to\", \"MetaB\"]": [1.0, 0.0, 0.0]
              }
            }
            """.trimIndent(),
        )
        cacheRoot.resolve("chunk_embedding_cache.json").writeText(
            """
            {
              "modelTag": "stale-model",
              "dimensions": 3,
              "vectors": {
                "c1": [1.0, 0.0, 0.0]
              }
            }
            """.trimIndent(),
        )

        createRetriever(
            datasetName = datasetName,
            config = config,
            graphPath = graphPath.toString(),
            rootDir = root,
        )

        val tripleCacheText = cacheRoot.resolve("triple_embedding_cache.json").readText()
        val chunkCacheText = cacheRoot.resolve("chunk_embedding_cache.json").readText()
        assertTrue("\"stale-model\"" !in tripleCacheText)
        assertTrue("\"stale-model\"" !in chunkCacheText)
        assertTrue(cacheRoot.resolve("triple_embedding_cache.npz").exists())
        assertTrue(cacheRoot.resolve("chunk_embedding_cache.npz").exists())
    }

    @Test
    fun `processRetrievalResults tolerates missing chunk file`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-missing-chunk-file-test")
        val config = createTestConfig(root)
        val datasetName = "missing_chunk_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")

        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "ChunklessA",
                        sourceLabel = "entity",
                        relation = "mentions",
                        targetName = "ChunklessB",
                        targetLabel = "entity",
                        chunkId = "cx",
                    ),
                ),
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
            )
        val (results, retrievalTime) = retriever.processRetrievalResults(question = "What mentions ChunklessA?")
        val chunkIds = (results["chunk_ids"] as? List<*>)?.map { it.toString() }.orEmpty()
        val chunkContents = (results["chunk_contents"] as? List<*>)?.map { it.toString() }.orEmpty()

        assertTrue(retrievalTime >= 0.0)
        assertTrue(chunkIds.isEmpty())
        assertTrue(chunkContents.isEmpty())
    }

    @Test
    fun `processRetrievalResults tolerates malformed graph json`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-malformed-graph-test")
        val config = createTestConfig(root)
        val datasetName = "malformed_graph_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")
        graphPath.parent?.createDirectories()
        graphPath.writeText("{ malformed")
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries = listOf("c1" to "Fallback chunk content."),
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
            )
        val (results, retrievalTime) = retriever.processRetrievalResults(question = "Anything here?")
        val triples = (results["triples"] as? List<*>)?.map { it.toString() }.orEmpty()

        assertTrue(retrievalTime >= 0.0)
        assertTrue(triples.isEmpty())
    }

    @Test
    fun `executeStrategies isolates strategy failures`() {
        val config = ConfigManager("config/base_config.json")
        val retriever = createRetriever(datasetName = "demo", config = config)
        val method =
            retriever::class.java.getDeclaredMethod(
                "executeStrategies",
                List::class.java,
                Set::class.java,
                Boolean::class.javaPrimitiveType,
                Long::class.javaPrimitiveType,
                Int::class.javaPrimitiveType,
            )
        method.isAccessible = true

        val strategyDefinitions =
            listOf(
                "fast_ok" to ({ mapOf("item_fast" to 1.0) }),
                "throws_error" to ({ throw IllegalStateException("boom") }),
            )

        @Suppress("UNCHECKED_CAST")
        val scores =
            method.invoke(
                retriever,
                strategyDefinitions,
                setOf("fast_ok", "throws_error"),
                true,
                5L,
                3,
            ) as List<Any>

        fun readStrategyName(score: Any): String {
            val field = score::class.java.getDeclaredField("strategy")
            field.isAccessible = true
            return field.get(score).toString()
        }

        fun readScores(score: Any): Map<String, Double> {
            val field = score::class.java.getDeclaredField("scores")
            field.isAccessible = true
            @Suppress("UNCHECKED_CAST")
            return field.get(score) as Map<String, Double>
        }

        val byStrategy = scores.associateBy(::readStrategyName, ::readScores)
        assertTrue(byStrategy["fast_ok"].orEmpty().isNotEmpty())
        assertTrue(byStrategy["throws_error"].orEmpty().isEmpty())
    }

    @Test
    fun `processRetrievalResults remains stable with tiny strategy timeout`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-tiny-timeout-stability-test")
        val config = createTestConfig(root)
        config.overrideConfig(
            mapOf(
                "retrieval" to
                    mapOf(
                        "strategy" to
                            mapOf(
                                "timeout_ms" to 1,
                                "enable_parallel" to true,
                            ),
                    ),
            ),
        )
        val datasetName = "tiny_timeout_ds"
        val graphPath = root.resolve("output/graphs/${datasetName}_new.json")
        writeGraph(
            graphPath = graphPath,
            relationships =
                listOf(
                    relationship(
                        sourceName = "TimeoutA",
                        sourceLabel = "entity",
                        relation = "relates_to",
                        targetName = "TimeoutB",
                        targetLabel = "entity",
                        chunkId = "c1",
                    ),
                ),
        )
        writeChunks(
            chunkFile = root.resolve("output/chunks/$datasetName.txt"),
            entries = listOf("c1" to "TimeoutA relates to TimeoutB."),
        )

        val retriever =
            createRetriever(
                datasetName = datasetName,
                config = config,
                graphPath = graphPath.toString(),
                rootDir = root,
                topK = 2,
            )
        val (results, retrievalTime) = retriever.processRetrievalResults(question = "What relates to TimeoutA?")
        val keys = results.keys

        assertTrue(retrievalTime >= 0.0)
        assertEquals(setOf("triples", "chunk_ids", "chunk_contents", "chunk_retrieval_results"), keys)
    }

    @Test
    fun `buildIndices throws when topK is negative`() {
        val root = Files.createTempDirectory("youtu-graphrag-retriever-negative-topk")
        val config = createTestConfig(root)
        assertFailsWith<IllegalArgumentException> {
            createRetriever(
                datasetName = "negative_topk_ds",
                config = config,
                topK = -1,
            )
        }
    }

    private fun createRetriever(
        datasetName: String,
        config: ConfigManager,
        graphPath: String = "unused_graph.json",
        rootDir: Path = Path.of("."),
        topK: Int = 5,
        recallPaths: Int = 2,
    ): KTRetriever =
        KTRetriever.createAndBuild(
            datasetName = datasetName,
            graphPath = graphPath,
            recallPaths = recallPaths,
            schemaPath = "unused_schema.json",
            topK = topK,
            mode = "agent",
            config = config,
            rootDir = rootDir,
        )

    private fun createTestConfig(root: Path): ConfigManager {
        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
                "output" to
                    mapOf(
                        "base_dir" to root.resolve("output").toString(),
                        "graphs_dir" to root.resolve("output/graphs").toString(),
                        "chunks_dir" to root.resolve("output/chunks").toString(),
                        "logs_dir" to root.resolve("output/logs").toString(),
                    ),
                "retrieval" to
                    mapOf(
                        "cache_dir" to root.resolve("retriever/faiss_cache_new").toString(),
                    ),
            ),
        )
        return config
    }

    private fun writeGraph(
        graphPath: Path,
        relationships: List<GraphRelationship>,
    ) {
        graphPath.parent?.createDirectories()
        mapper.writerWithDefaultPrettyPrinter().writeValue(graphPath.toFile(), relationships)
    }

    private fun writeChunks(
        chunkFile: Path,
        entries: List<Pair<String, String>>,
    ) {
        chunkFile.parent?.createDirectories()
        val content =
            buildString {
                entries.forEach { (chunkId, text) ->
                    append("id: ")
                    append(chunkId)
                    append('\t')
                    append("Chunk: ")
                    append(text)
                    append('\n')
                }
            }
        chunkFile.writeText(content)
    }

    private fun relationship(
        sourceName: String,
        sourceLabel: String,
        relation: String,
        targetName: String,
        targetLabel: String,
        chunkId: String,
    ): GraphRelationship =
        GraphRelationship(
            startNode =
                GraphNode(
                    label = sourceLabel,
                    properties =
                        mapOf(
                            "name" to sourceName,
                            "chunk id" to chunkId,
                            "schema_type" to sourceLabel,
                        ),
                ),
            relation = relation,
            endNode =
                GraphNode(
                    label = targetLabel,
                    properties =
                        mapOf(
                            "name" to targetName,
                            "chunk id" to chunkId,
                            "schema_type" to targetLabel,
                        ),
                ),
        )
}
