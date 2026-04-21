package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.di.AppConfig
import lightrag.di.LightRagConfig
import lightrag.di.appModule
import lightrag.services.StorageManager
import org.koin.core.context.loadKoinModules
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.dsl.module
import org.koin.java.KoinJavaComponent.get

/**
 * The main function for the LightRAG OpenAI MongoDB Graph demo.
 * This function demonstrates how to use LightRAG with OpenAI models and a MongoDB-backed graph storage.
 * It initializes the models and storage, inserts a document, and queries it using different modes.
 */
fun main() =
    runBlocking {
        startKoin {
            allowOverride(true)
            modules(appModule)
        }
        loadKoinModules(mongoOverrideModule())

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

        prepareWorkingDir("./mongodb_test_dir")
        storageManager.initialize()

        val content = loadBookContent()
        rag.insert(content)

        runDemoQueries(rag, "What are the top themes in this story?")
        storageManager.persist()
    }

private fun mongoOverrideModule() =
    module {
        single<AppConfig> {
            val cfg = get<LightRagConfig>()
            AppConfig(
                workingDir = cfg.storage.workingDir,
                graphStorageName = "MongoGraphStorage",
                vectorStorageName = cfg.storage.vectorStorageName,
                addonConfig = addonConfigFrom(cfg),
                llmBinding = "openai",
                llmModelName = cfg.openai.chatModelName,
                embeddingBinding = "openai",
                embeddingModelName = cfg.openai.embeddingModelName,
                chatModel = get(),
                embeddingModel = get(),
            )
        }

        single<Map<String, Any?>>(named("globalConfig")) {
            val appConfig = get<AppConfig>()
            globalConfigFrom(appConfig)
        }
    }
