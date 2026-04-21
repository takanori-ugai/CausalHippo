package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.di.appModule
import lightrag.services.StorageManager
import org.koin.core.context.startKoin

/**
 * The main function for the LightRAG OpenAI demo.
 * This function demonstrates how to use LightRAG with OpenAI models.
 * It initializes the models, inserts a document, and queries it using different modes.
 */
fun main() =
    runBlocking {
        val koin = startKoin { modules(appModule) }.koin
        val rag: LightRAG = koin.get()
        val storageManager: StorageManager = koin.get()

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

        // Initialize storages (connects Neo4j and loads persisted data) before insert/query.
        storageManager.initialize()

        testEmbeddingModel(koin.get(), "This is a test string for embedding.")
        rag.insert(loadBookContent())
        runDemoQueries(rag, "What are the top themes related with King of England")
        println("\nDone!")
    }
