package lightrag.examples

import dev.langchain4j.model.chat.ChatModel
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.AppConfig
import lightrag.di.LightRagConfig
import lightrag.di.appModule
import lightrag.llm.DualChatModel
import lightrag.llm.LLMFactory
import org.koin.core.context.loadKoinModules
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.dsl.module

/**
 * Demonstrates how to run LightRAG with streaming mode enabled.
 * Starts with the OpenAI defaults, inserts a document, and streams the response tokens.
 */
fun main() =
    runBlocking {
        val koin =
            startKoin {
                allowOverride(true)
                modules(appModule)
            }.koin

        // Use in-memory storage so the demo runs without Neo4j
        loadKoinModules(
            module {
                single<ChatModel> {
                    val cfg = get<LightRagConfig>()
                    val chatModel =
                        LLMFactory.createChatModel(
                            binding = "openai",
                            modelName = cfg.openai.chatModelName,
                            apiKey = cfg.openai.apiKey,
                        )
                    val streamingChatModel =
                        LLMFactory.createStreamingChatModel(
                            binding = "openai",
                            modelName = cfg.openai.chatModelName,
                            apiKey = cfg.openai.apiKey,
                        )
                    DualChatModel(chatModel, streamingChatModel)
                }

                single<AppConfig> {
                    val cfg = get<LightRagConfig>()
                    AppConfig(
                        workingDir = cfg.storage.workingDir,
                        graphStorageName = "InMemoryGraphStorage",
                        vectorStorageName = "InMemoryVectorStorage",
                        addonConfig = addonConfigFrom(cfg),
                        chatModel = get(),
                        embeddingModel = get(),
                    )
                }

                single<Map<String, Any?>>(named("globalConfig")) {
                    val appConfig = get<AppConfig>()
                    globalConfigFrom(appConfig)
                }
            },
        )

        val rag: LightRAG = koin.get()

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

        testEmbeddingModel(koin.get(), "This is a test string for embedding.")
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
