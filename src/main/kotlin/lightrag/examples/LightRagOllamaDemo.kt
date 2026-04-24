package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.di.createLightRagRuntime

/**
 * The main function for the LightRAG Ollama demo.
 * This function demonstrates how to use LightRAG with Ollama-hosted models.
 * It initializes the models, inserts a document, and queries it using different modes.
 */
fun main() =
    runBlocking {
        val runtime = createLightRagRuntime(configTransform = { it.copy(provider = "ollama") })
        val rag = runtime.rag

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
        testEmbeddingModel(runtime.embeddingModel, "This is a test string for embedding.")

        rag.insert(loadBookContent())

        runDemoQueries(rag, "What are the top themes in this story?")
        println("\nDone!")
    }
