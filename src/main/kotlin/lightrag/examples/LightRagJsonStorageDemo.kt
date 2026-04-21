package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.AppConfig
import lightrag.di.LightRagConfig
import lightrag.di.appModule
import lightrag.services.StorageManager
import org.koin.core.context.loadKoinModules
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.dsl.module
import org.koin.java.KoinJavaComponent.get
import java.io.File

/**
 * The main function for the LightRAG JSON storage demo.
 * This function demonstrates how to use LightRAG with JSON-backed storages.
 * It initializes the storages, inserts a document, queries it using different modes, and persists the data.
 */
fun main() =
    runBlocking {
        startKoin {
            allowOverride(true)
            modules(appModule, jsonStorageOverrideModule())
        }

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

        // Configure API key, chat model, embedding model through Koin modules

        // Initialize storages and reload any persisted state
        storageManager.initialize()

        val bookFile = File("./book.txt")
        val content =
            if (bookFile.exists()) {
                bookFile.readText()
            } else {
                println("Warning: ./book.txt not found. Using dummy content.")
                "It was the best of times, it was the worst of times."
            }

        println("Inserting document into JSON storage...")
        rag.insert(content)
        rag.rebuildDerivedStorageIfEmpty()

        println("\nDoc status snapshot (JsonDocStatusStorage):")
        println(storageManager.docStatusStorage.getStatusCounts())

        println("\nFull docs snapshot (JsonKVStorage):")
        println(storageManager.fullDocs.getByIds(listOf("0")))

        val modes = listOf("naive", "local", "global")
//        val queryText = "What are the top themes in this story?"
        val queryText = "What are the top themes related with King of England?"
        val hl = listOf("times", "wisdom", "foolishness")
        val ll = listOf("king", "queen", "England", "France")

        modes.forEach { mode ->
            println("\n=====================")
            println("Query mode: $mode")
            println("=====================")
            val param =
                QueryParam(
                    mode = mode,
                    includeReferences = true,
                    topK = 5,
                    chunkTopK = 5,
                    //                  hlKeywords = hl,
//                    llKeywords = ll,
                )
            val contextOnly = rag.query(queryText, param.copy(onlyNeedContext = true))
            println("Context preview:\n${contextOnly?.content ?: "(empty)"}\n")

            val result = rag.query(queryText, param)
            println(result?.content ?: "(no result)")
        }

        // Persist JSON-backed storages (KV, DocStatus, Vectors)
        storageManager.persist()

        println("\nDone! Data persisted using JSON-backed storages.")
    }

private fun jsonStorageOverrideModule() =
    module {
        single<AppConfig> {
            val cfg = get<LightRagConfig>()
            AppConfig(
                workingDir = "./json_demo_storage",
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
    }
