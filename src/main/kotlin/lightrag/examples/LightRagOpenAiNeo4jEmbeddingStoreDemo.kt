package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.createLightRagRuntime

/**
 * Demo showing LightRAG with Neo4jEmbeddingStoreVectorStorage (langchain4j community Neo4j embedding store).
 *
 * Requirements:
 * - Neo4j 5.15+ with vector index capability (defaults to bolt://localhost:7687 with neo4j/neo4j)
 * - OPENAI_API_KEY set
 */
fun main() =
    runBlocking {
        val runtime =
            createLightRagRuntime(
                configTransform = { it.copy(provider = "openai") },
                appConfigTransform =
                    { appConfig, _ ->
                        appConfig.copy(vectorStorageName = "Neo4jEmbeddingStoreVectorStorage")
                    },
            )
        val rag = runtime.rag
        val storageManager = runtime.storageManager

        println("Initializing Neo4j embedding store vector storage...")
        storageManager.initialize()
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
 * Inserts a small sample about France and Paris for querying.
 * @param rag LightRAG instance used for ingestion
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
