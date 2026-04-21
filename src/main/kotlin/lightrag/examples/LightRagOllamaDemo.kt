package lightrag.examples

import dev.langchain4j.model.embedding.EmbeddingModel
import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.di.AppConfig
import lightrag.di.LightRagConfig
import lightrag.di.appModule
import lightrag.llm.LLMFactory
import org.koin.core.context.loadKoinModules
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.dsl.module

/**
 * The main function for the LightRAG Ollama demo.
 * This function demonstrates how to use LightRAG with Ollama-hosted models.
 * It initializes the models, inserts a document, and queries it using different modes.
 */
fun main() =
    runBlocking {
        val koin = startOllamaKoin()
        loadKoinModules(buildOllamaModule(koin))

        val rag: LightRAG = koin.get()
        prepareWorkingDir(
            "./ollama-demo",
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

        val bookContent = loadBookContent()
        rag.insert(bookContent)

        runDemoQueries(rag, "What are the top themes in this story?")
        println("\nDone!")
    }

private fun startOllamaKoin() =
    startKoin {
        allowOverride(true)
        modules(appModule)
    }.koin

private fun buildOllamaModule(koin: org.koin.core.Koin) =
    module {
        single<dev.langchain4j.model.chat.ChatModel> { createOllamaChatModel(koin.get()) }
        single<EmbeddingModel> { createOllamaEmbeddingModel(koin.get()) }
        single<AppConfig> { createOllamaAppConfig(koin.get(), koin.get(), koin.get()) }
        single<Map<String, Any?>>(named("globalConfig")) { globalConfigFrom(koin.get()) }
    }

private fun createOllamaChatModel(cfg: LightRagConfig) =
    LLMFactory.createChatModel(
        binding = "ollama",
        modelName = cfg.ollama.chatModelName,
        baseUrl = cfg.ollama.baseUrl,
    )

private fun createOllamaEmbeddingModel(cfg: LightRagConfig) =
    LLMFactory.createEmbeddingModel(
        binding = "ollama",
        modelName = cfg.ollama.embeddingModelName,
        baseUrl = cfg.ollama.baseUrl,
    )

private fun createOllamaAppConfig(
    cfg: LightRagConfig,
    chatModel: dev.langchain4j.model.chat.ChatModel,
    embeddingModel: EmbeddingModel,
): AppConfig =
    AppConfig(
        workingDir = cfg.storage.workingDir,
        graphStorageName = cfg.storage.graphStorageName,
        vectorStorageName = cfg.storage.vectorStorageName,
        addonConfig = addonConfigFrom(cfg),
        llmBinding = "ollama",
        llmModelName = cfg.ollama.chatModelName,
        embeddingBinding = "ollama",
        embeddingModelName = cfg.ollama.embeddingModelName,
        chatModel = chatModel,
        embeddingModel = embeddingModel,
    )
