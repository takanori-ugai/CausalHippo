package lightrag.examples

import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.createLightRagRuntime

/**
 * Demonstrates how to run LightRAG with streaming mode enabled.
 * Starts with the OpenAI defaults, inserts a document, and streams the response tokens.
 */
fun main() =
    runBlocking {
        val runtime =
            createLightRagRuntime(
                configTransform = { it.copy(provider = "openai") },
                appConfigTransform =
                    { appConfig, _ ->
                        appConfig.copy(
                            graphStorageName = "InMemoryGraphStorage",
                            vectorStorageName = "InMemoryVectorStorage",
                        )
                    },
            )
        val rag: LightRAG = runtime.rag

        prepareWorkingDir(
            "./dickens",
            filesToDelete =
                listOf(
                    "graph_chunk_entity_relation.graphml",
                    "kv_store_doc_status.json",
                    "kv_store_full_docs.json",
                    "kv_store_text_chunks.json",
                    "vdb_chunks.json",
                    "vdb_entities.json",
                    "vdb_relationships.json",
                ),
        )

        testEmbeddingModel(runtime.embeddingModel, "This is a test string for embedding.")
        rag.insert(loadBookContent())
        rag.rebuildDerivedStorageIfEmpty()

        val queryText = "What are the top themes related with king of England?"
        val modes = listOf("local", "global", "hybrid")
        modes.forEach { mode ->
            val contextResult =
                rag.query(
                    queryText,
                    QueryParam(
                        mode = mode,
                        onlyNeedContext = true,
                        topK = 5,
                        chunkTopK = 2,
                        includeReferences = true,
                    ),
                )
            println("\n=====================")
            println("Mode: $mode context preview")
            println("=====================")
            println(contextResult?.content ?: "(no context)")

            val queryParam =
                QueryParam(
                    mode = mode,
                    stream = true,
                    topK = 5,
                    chunkTopK = 2,
                    includeReferences = true,
                )

            val result = rag.query(queryText, queryParam)
            when {
                result == null -> {
                    println("No result generated for mode $mode.")
                }

                result.isStreaming && result.responseIterator != null -> {
                    println("\n=====================")
                    println("Mode: $mode (streaming)")
                    println("=====================")
                    result.responseIterator.collect { token -> print(token) }
                    println()
                }

                else -> {
                    println("\n=====================")
                    println("Mode: $mode")
                    println("=====================")
                    println(result.content)
                }
            }
        }

        println("\nDone!")
    }
