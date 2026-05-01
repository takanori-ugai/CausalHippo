package pathrag

import kotlinx.coroutines.runBlocking
import pathrag.base.QueryParam
import shared.rag.CommonRag
import java.nio.file.Paths

/**
 * Demonstrates PathRAG usage through the CommonRag contract.
 *
 * This sample uses only CommonRag-provided functions:
 * `upsert`, `query`, `saveGraph`, `loadGraph`, and `drop`.
 */
fun main() =
    runBlocking {
        val env = EnvironmentConfig.load(Paths.get("../.env"))
        val kvStorage = env["KV_STORAGE"] ?: "JsonKVStorage"
        val vectorStorage = env["VECTOR_STORAGE"] ?: "NanoVectorDBStorage"
        val graphStorage = env["GRAPH_STORAGE"] ?: "NetworkXStorage"
        val workingDir = env["WORKING_DIR"] ?: "./sample_cache"
        val pathRag =
            PathRAG(
                workingDir = workingDir,
                kvStorage = kvStorage,
                vectorStorage = vectorStorage,
                graphStorage = graphStorage,
                chunkTokenSize = 800,
                chunkOverlapTokenSize = 120,
                language = env["LANGUAGE"] ?: "English",
                keywordExamples = "",
                // Optional: pin keywords instead of calling the LLM extractor
                // highLevelKeywords = listOf("themes", "Dickens"),
                // lowLevelKeywords = listOf("poverty", "class struggle", "redemption"),
                similarityCheckPrompt = pathrag.prompt.Prompts.SIMILARITY_CHECK,
                embeddingCacheConfig =
                    mapOf(
                        "enabled" to true,
                        "similarity_threshold" to 0.9,
                        "use_llm_check" to false,
                    ),
                addonParams =
                    pathrag.base.AddonParams(
                        entityTypes = listOf("organization", "person", "geo", "event", "category"),
                        // language is set at top-level already
                        exampleNumber = 3,
                    ),
            )
        val rag: CommonRag<QueryParam, String> = pathRag

        val graphPath = Paths.get(workingDir, "knowledge-graph.json")
        val documents =
            listOf(
                """
                Charles Dickens was an English writer and social critic.
                He created some of the world's best-known fictional characters
                and is regarded as one of the greatest novelists of the Victorian era.
                """.trimIndent(),
                """
                Oliver Twist is a novel by Dickens that critiques workhouses and child poverty.
                It follows an orphan navigating criminal underworlds and harsh social systems.
                """.trimIndent(),
                """
                A Christmas Carol tells the redemption story of Ebenezer Scrooge, shifting from greed to generosity.
                It explores themes of morality, compassion, and social responsibility.
                """.trimIndent(),
            )

        rag.upsert(documents)
        rag.saveGraph(graphPath.toString())
        println("Knowledge graph saved to: ${graphPath.toAbsolutePath()}")

        val question = "What themes does Dickens explore?"

        // Local mode: entity-centric context only.
        val localAnswer = rag.query(question, param = QueryParam(mode = "local", onlyNeedContext = true))
        println("Q (local): $question")
        println("A: $localAnswer\n")

        // Global mode: relationship-centric context.
        val globalAnswer = rag.query(question, param = QueryParam(mode = "global", onlyNeedContext = true))
        println("Q (global): $question")
        println("A: $globalAnswer\n")

        // Hybrid mode: intentionally fetches context and full answer separately (two queries).
        val context = rag.query(question, param = QueryParam(mode = "hybrid", onlyNeedContext = true))
        val hybridAnswer = rag.query(question, param = QueryParam(mode = "hybrid", onlyNeedContext = false))
        println("Context: $context")
        println("Q (hybrid): $question")
        println("A: $hybridAnswer\n")

        rag.drop()
        rag.loadGraph(graphPath.toString())
        val reloadedAnswer = rag.query(question, param = QueryParam(mode = "hybrid", onlyNeedContext = true))
        println("Q (reloaded): $question")
        println("A: $reloadedAnswer\n")

        pathRag.close()
        println("\nDone!")
    }
