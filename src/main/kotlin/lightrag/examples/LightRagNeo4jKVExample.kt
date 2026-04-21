package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.AppConfig
import lightrag.di.appModule
import lightrag.kg.neo4j.Neo4jKVStorage
import lightrag.services.StorageManager
import org.koin.core.context.loadKoinModules
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.dsl.module
import org.koin.java.KoinJavaComponent.get

/**
 * Minimal example showing how to use Neo4jKVStorage for KV persistence.
 *
 * Prerequisites:
 *  - Neo4j running and accessible via NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD (or globalConfig["neo4j"])
 *  - OPENAI_API_KEY set for LLMs
 */
fun main() =
    runBlocking {
        startKoin {
            allowOverride(true)
            modules(appModule)
        }

        // Override StorageManager to use Neo4jKVStorage for all KV stores (full docs, chunks, entities, relations).
        val neo4jKvModule =
            module {
                single<StorageManager> {
                    val appConfig = get<AppConfig>()
                    val globalConfig = get<Map<String, Any?>>(named("globalConfig"))
                    val embeddingModel = appConfig.embeddingModel

                    fun neo4jKv(namespace: String) =
                        Neo4jKVStorage(
                            namespace = namespace,
                            workspace = System.getenv("NEO4J_WORKSPACE") ?: "default",
                            globalConfig = globalConfig,
                            embeddingFunc = embeddingModel,
                        )

                    StorageManager(
                        workingDir = appConfig.workingDir,
                        embeddingModel = embeddingModel,
                        graphStorageName = appConfig.graphStorageName,
                        vectorStorageName = appConfig.vectorStorageName,
                        addonConfig = appConfig.addonConfig,
                        globalConfig = globalConfig,
                        fullDocsStorageOverride = neo4jKv("full_docs"),
                        textChunksStorageOverride = neo4jKv("text_chunks"),
                        fullEntitiesStorageOverride = neo4jKv("full_entities"),
                        fullRelationsStorageOverride = neo4jKv("full_relations"),
                        docStatusStorageOverride = appConfig.docStatusStorageOverride,
                    )
                }
            }
        loadKoinModules(neo4jKvModule)

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

        // Initialize storages (creates Neo4j connections/indexes and loads any persisted data).
        storageManager.initialize()

        val text =
            """
            Neo4j is a native graph database.
            LightRAG can use Neo4j for vectors, graph, or KV cache.
            """.trimIndent()

        println("Inserting text...")
        rag.insert(text)

        val queryText = "How does LightRAG use Neo4j?"
        val modes = listOf("global", "local")

        modes.forEach { mode ->
            println("\n=====================")
            println("Query mode: $mode")
            println("=====================")
            val result =
                rag.query(
                    queryText,
                    QueryParam(
                        mode = mode,
                        includeReferences = true,
                        topK = 3,
                        chunkTopK = 3,
                    ),
                )
            println(result?.content ?: "No result")
        }

        println("Persisting storages...")
        storageManager.persist()
        println("Done. Hashing cache persisted in Neo4j and other stores saved locally.")
    }
