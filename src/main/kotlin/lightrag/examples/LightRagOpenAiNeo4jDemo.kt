package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.di.LightRagConfig
import lightrag.di.createLightRagRuntime
import lightrag.llm.LLMFactory
import lightrag.services.StorageManager

/**
 * The main function for the LightRAG OpenAI Neo4j demo.
 * This function demonstrates how to use LightRAG with OpenAI models and a Neo4j-backed graph storage.
 * It initializes the models and storage, inserts a document, and queries it using different modes.
 */
fun main() =
    runBlocking {
        val runtime =
            createLightRagRuntime(
                configTransform = { it.copy(provider = "openai") },
                chatModelFactory = { cfg -> createOpenAiLoggingChatModel(cfg) },
                appConfigTransform =
                    { appConfig, _ ->
                        appConfig.copy(graphStorageName = "Neo4jGraphStorage")
                    },
            )
        val rag: LightRAG = runtime.rag
        val storageManager: StorageManager = runtime.storageManager

        if (!initializeNeo4j(storageManager)) return@runBlocking

        insertContent(rag, loadNeo4jContent())
        runDemoQueries(
            rag,
            "What are the top themes related with king of England?",
            paramBuilder =
                { mode ->
                    lightrag.core.QueryParam(
                        mode = mode,
                        includeReferences = true,
                        topK = 2,
                        chunkTopK = 2,
                    )
                },
        )

        storageManager.persist()
        println("\nDone!")
    }

private fun createOpenAiLoggingChatModel(cfg: LightRagConfig) =
    LLMFactory.createChatModel(
        binding = "openai",
        modelName = cfg.openai.chatModelName,
        apiKey = cfg.openai.apiKey,
        logRequests = true,
        logResponses = true,
    )

/**
 * Initializes Neo4j-backed storages and drops any existing data so the demo starts fresh.
 * @param storageManager storage manager with configured Neo4j backends
 * @return true if initialization succeeds, false otherwise
 */
private suspend fun initializeNeo4j(storageManager: StorageManager): Boolean {
    println("Initializing Neo4j Graph Storage...")
    return try {
        storageManager.initialize()
        println("Dropping existing storage data...")
        storageManager.drop()
        true
    } catch (e: IllegalStateException) {
        println("Error initializing Neo4j storage: ${e.message}")
        println("Please ensure Neo4j is running and configured via config/common_rag.json or NEO4J_* environment variables")
        false
    } catch (e: IllegalArgumentException) {
        println("Error initializing Neo4j storage: ${e.message}")
        println("Please ensure Neo4j is running and configured via config/common_rag.json or NEO4J_* environment variables")
        false
    }
}

/**
 * Loads demo content from `book.txt` if available or uses a fallback snippet.
 * @return the text content used for ingestion
 */
private fun loadNeo4jContent(): String {
    val bookFile = java.io.File("book.txt")
    return if (bookFile.exists()) {
        bookFile.readText()
    } else {
        println("Warning: book.txt not found. Using dummy content.")
        """
        Neo4j is a graph database management system developed by Neo4j, Inc.
        It is an ACID-compliant transactional database with native graph storage and processing.
        LightRAG is a retrieval-augmented generation system that can use Neo4j as a backend.
        Kotlin is a cross-platform, statically typed, general-purpose programming language with type inference.
        """.trimIndent()
    }
}

/**
 * Inserts the provided content into LightRAG while handling common ingestion errors.
 * @param rag LightRAG instance to ingest the text
 * @param content text to insert
 */
private suspend fun insertContent(
    rag: LightRAG,
    content: String,
) {
    println("Inserting content...")
    try {
        rag.insert(content)
    } catch (e: IllegalStateException) {
        println("Error inserting content: ${e.message}")
    } catch (e: IllegalArgumentException) {
        println("Error inserting content: ${e.message}")
    }
}
