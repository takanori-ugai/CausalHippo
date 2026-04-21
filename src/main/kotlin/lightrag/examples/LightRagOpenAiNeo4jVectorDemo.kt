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

/**
 * Demo showing LightRAG with both graph storage and vector storage backed by Neo4j.
 *
 * Requirements:
 * - Neo4j running (defaults to bolt://localhost:7687 with neo4j/neo4j)
 * - OPENAI_API_KEY set
 */
fun main() =
    runBlocking {
        startKoin {
            allowOverride(true)
            modules(appModule)
        }
        loadKoinModules(neo4jOverrideModule())

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

        println("Initializing Neo4j vector/graph storage...")
        storageManager.initialize()
        println("Dropping existing storage data...")
        storageManager.drop()

        insertDemoContent(rag)
        runDemoQueries(
            rag,
            "What are key attractions in the capital of France?",
            modes = listOf("naive", "local", "global"),
        ) { mode ->
            QueryParam(
                mode = mode,
                includeReferences = true,
                topK = 3,
                chunkTopK = 3,
            )
        }

        storageManager.persist()
    }

/**
 * Overrides Koin bindings to force Neo4j-backed graph and vector storage for the demo run.
 */
private fun neo4jOverrideModule() =
    module {
        single<AppConfig> {
            val cfg = get<LightRagConfig>()
            AppConfig(
                workingDir = cfg.storage.workingDir,
                graphStorageName = "Neo4jGraphStorage",
                vectorStorageName = "Neo4jVectorStorage",
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

/**
 * Inserts the canned demo content into LightRAG.
 * @param rag LightRAG instance to ingest the sample text
 */
private suspend fun insertDemoContent(rag: LightRAG) {
    val content =
        """
        The capital of France is Paris. The Eiffel Tower is a landmark in Paris.
        The Louvre Museum houses famous artworks like the Mona Lisa.
        """.trimIndent()

    println("Inserting content...")
    rag.insert(content)
}
