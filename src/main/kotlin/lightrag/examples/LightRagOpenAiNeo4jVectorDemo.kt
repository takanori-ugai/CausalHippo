package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.createLightRagRuntime

/**
 * Demo showing LightRAG with both graph storage and vector storage backed by Neo4j.
 *
 * Requirements:
 * - Neo4j running (defaults to bolt://localhost:7687 with neo4j/neo4j)
 * - OPENAI_API_KEY set
 */
fun main() =
    runBlocking {
        val runtime =
            createLightRagRuntime(
                configTransform = { it.copy(provider = "openai") },
                appConfigTransform =
                    { appConfig, _ ->
                        appConfig.copy(
                            graphStorageName = "Neo4jGraphStorage",
                            vectorStorageName = "Neo4jVectorStorage",
                        )
                    },
            )
        val rag = runtime.rag
        val storageManager = runtime.storageManager

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
