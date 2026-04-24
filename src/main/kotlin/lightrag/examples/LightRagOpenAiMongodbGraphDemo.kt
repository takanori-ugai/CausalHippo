package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.di.createLightRagRuntime

/**
 * The main function for the LightRAG OpenAI MongoDB Graph demo.
 * This function demonstrates how to use LightRAG with OpenAI models and a MongoDB-backed graph storage.
 * It initializes the models and storage, inserts a document, and queries it using different modes.
 */
fun main() =
    runBlocking {
        val runtime =
            createLightRagRuntime(
                configTransform = { it.copy(provider = "openai") },
                appConfigTransform =
                    { appConfig, _ ->
                        appConfig.copy(graphStorageName = "MongoGraphStorage")
                    },
            )
        val rag = runtime.rag
        val storageManager = runtime.storageManager

        prepareWorkingDir("./mongodb_test_dir")
        storageManager.initialize()

        val content = loadBookContent()
        rag.insert(content)

        runDemoQueries(rag, "What are the top themes in this story?")
        storageManager.persist()
    }
