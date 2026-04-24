package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.createLightRagRuntime
import lightrag.kg.neo4j.Neo4jDocStatusStorage
import lightrag.services.StorageManager

/**
 * Example showing how to use Neo4jDocStatusStorage for doc status persistence.
 *
 * Prerequisites:
 *  - Neo4j reachable via NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD (or set in globalConfig["neo4j"])
 *  - OPENAI_API_KEY set
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
                storageManagerFactory =
                    { appConfig, globalConfig ->
                        val embeddingModel = appConfig.embeddingModel
                        StorageManager(
                            workingDir = appConfig.workingDir,
                            embeddingModel = embeddingModel,
                            graphStorageName = appConfig.graphStorageName,
                            vectorStorageName = appConfig.vectorStorageName,
                            addonConfig = appConfig.addonConfig,
                            globalConfig = globalConfig,
                            docStatusStorageOverride =
                                Neo4jDocStatusStorage(
                                    namespace = "doc_status",
                                    workspace = System.getenv("NEO4J_WORKSPACE") ?: "default",
                                    globalConfig = globalConfig,
                                    embeddingFunc = embeddingModel,
                                ),
                            fullDocsStorageOverride = appConfig.fullDocsStorageOverride,
                            textChunksStorageOverride = appConfig.textChunksStorageOverride,
                            fullEntitiesStorageOverride = appConfig.fullEntitiesStorageOverride,
                            fullRelationsStorageOverride = appConfig.fullRelationsStorageOverride,
                        )
                    },
            )

        val rag: LightRAG = runtime.rag
        val storageManager: StorageManager = runtime.storageManager

        storageManager.initialize()
        println("Inserting document (status tracked in Neo4j)...")
        rag.insert(loadBookContent())

        println("Doc status snapshot (Neo4jDocStatusStorage):")
        println(storageManager.docStatusStorage.getStatusCounts())

        runDemoQueries(
            rag,
            "What roles do the kings and queens have in this story?",
            modes = listOf("naive", "global", "local", "hybrid"),
        ) { mode ->
            QueryParam(
                mode = mode,
                includeReferences = true,
                topK = 3,
                chunkTopK = 2,
            )
        }

        storageManager.persist()
        println("Done. Doc statuses persisted in Neo4j.")
    }
