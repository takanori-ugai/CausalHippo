package lightrag.services

import dev.langchain4j.model.embedding.EmbeddingModel
import lightrag.core.AddonConfig
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.core.types.DocStatusStorage
import lightrag.kg.json.JsonDocStatusStorage
import lightrag.kg.json.JsonKVStorage
import lightrag.kg.memory.InMemoryGraphStorage
import lightrag.kg.memory.InMemoryVectorStorage
import lightrag.kg.mongo.MongoGraphStorage
import lightrag.kg.neo4j.Neo4jEmbeddingStoreVectorStorage
import lightrag.kg.neo4j.Neo4jGraphStorage
import lightrag.kg.neo4j.Neo4jVectorStorage

/**
 * Central factory/wrapper that builds and exposes the configured storage implementations (KV, vector, and graph).
 * Supports overriding storage backends via configuration for demos and production use.
 */
class StorageManager(
    workingDir: String,
    embeddingModel: EmbeddingModel,
    docStatusStorageOverride: DocStatusStorage? = null,
    fullDocsStorageOverride: BaseKVStorage? = null,
    textChunksStorageOverride: BaseKVStorage? = null,
    fullEntitiesStorageOverride: BaseKVStorage? = null,
    fullRelationsStorageOverride: BaseKVStorage? = null,
    graphStorageName: String = "InMemoryGraphStorage",
    vectorStorageName: String = "InMemoryVectorStorage",
    addonConfig: AddonConfig = AddonConfig(),
    globalConfig: Map<String, Any?>,
) {
    /**

     * The storage for document statuses.

     */

    val docStatusStorage: DocStatusStorage =

        docStatusStorageOverride

            ?: JsonDocStatusStorage(
                namespace = "doc_status",
                workspace = "default",
                globalConfig = mapOf("working_dir" to workingDir),
                embeddingFunc = embeddingModel,
            )

    /**

     * The storage for full documents.

     */

    val fullDocs: BaseKVStorage =

        fullDocsStorageOverride

            ?: JsonKVStorage(
                namespace = "full_docs",
                workspace = "default",
                globalConfig = mapOf("working_dir" to workingDir),
                embeddingFunc = embeddingModel,
            ) as BaseKVStorage // Explicitly cast to BaseKVStorage

    /**

     * The storage for text chunks.

     */

    val textChunks: BaseKVStorage =

        textChunksStorageOverride

            ?: JsonKVStorage(
                namespace = "text_chunks",
                workspace = "default",
                globalConfig = mapOf("working_dir" to workingDir),
                embeddingFunc = embeddingModel,
            ) as BaseKVStorage // Explicitly cast to BaseKVStorage

    /**

     * The storage for full entities.

     */

    val fullEntities: BaseKVStorage =

        fullEntitiesStorageOverride

            ?: JsonKVStorage(
                namespace = "full_entities",
                workspace = "default",
                globalConfig = mapOf("working_dir" to workingDir),
                embeddingFunc = embeddingModel,
            ) as BaseKVStorage // Explicitly cast to BaseKVStorage

    /**

     * The storage for full relations.

     */

    val fullRelations: BaseKVStorage =

        fullRelationsStorageOverride

            ?: JsonKVStorage(
                namespace = "full_relations",
                workspace = "default",
                globalConfig = mapOf("working_dir" to workingDir),
                embeddingFunc = embeddingModel,
            ) as BaseKVStorage // Explicitly cast to BaseKVStorage

    private fun createVectorStorage(
        namespace: String,
        vectorStorageName: String,
        globalConfig: Map<String, Any?>,
        embeddingModel: EmbeddingModel,
        addonConfig: AddonConfig,
    ): BaseVectorStorage =
        when (vectorStorageName) {
            "Neo4jEmbeddingStoreVectorStorage", "Neo4jEmbeddingStore" -> {
                Neo4jEmbeddingStoreVectorStorage(
                    namespace = namespace,
                    workspace = "default",
                    globalConfig = globalConfig,
                    embeddingFunc = embeddingModel,
                    cosineThreshold = addonConfig.cosineBetterThreshold,
                )
            }

            "Neo4jVectorStorage" -> {
                Neo4jVectorStorage(
                    namespace = namespace,
                    workspace = "default",
                    globalConfig = globalConfig,
                    embeddingFunc = embeddingModel,
                    cosineThreshold = addonConfig.cosineBetterThreshold,
                )
            }

            else -> {
                InMemoryVectorStorage(
                    namespace = namespace,
                    workspace = "default",
                    embeddingFunc = embeddingModel,
                    globalConfig = globalConfig,
                    cosineThreshold = addonConfig.cosineBetterThreshold,
                )
            }
        }

    /**

     * The vector storage for chunks.

     */

    val chunksVdb: BaseVectorStorage = createVectorStorage("chunks_vdb", vectorStorageName, globalConfig, embeddingModel, addonConfig)

    /**

     * The vector storage for entities.

     */

    val entitiesVdb: BaseVectorStorage = createVectorStorage("entities_vdb", vectorStorageName, globalConfig, embeddingModel, addonConfig)

    /**

     * The vector storage for relationships.

     */

    val relationshipsVdb: BaseVectorStorage =
        createVectorStorage("relationships_vdb", vectorStorageName, globalConfig, embeddingModel, addonConfig)

    /**

     * The graph storage for the chunk-entity-relation graph.

     */

    val chunkEntityRelationGraph: BaseGraphStorage =

        when (graphStorageName) {

            "MongoGraphStorage" -> {

                MongoGraphStorage(
                    namespace = "chunk_entity_relation_graph",
                    globalConfig = globalConfig,
                    embeddingFunc = embeddingModel,
                )
            }

            "Neo4jGraphStorage" -> {

                Neo4jGraphStorage(
                    namespace = "chunk_entity_relation_graph",
                    globalConfig = globalConfig,
                    embeddingFunc = embeddingModel,
                )
            }

            else -> {
                val graphGlobalConfig =
                    globalConfig
                        .filterValues { it != null }
                        .mapValues { it.value as Any }
                InMemoryGraphStorage(
                    namespace = "chunk_entity_relation_graph",
                    workspace = "default",
                    globalConfig = graphGlobalConfig,
                    embeddingFunc = embeddingModel,
                )
            }
        }

    /**

     * The key-value storage.

     */

    val kvStorage: BaseKVStorage = textChunks

    /**

     * The vector storage.

     */

    val vectorStorage: BaseVectorStorage = entitiesVdb

    /**

     * The graph storage.

     */

    val graphStorage: BaseGraphStorage = chunkEntityRelationGraph

    suspend fun initialize() {
        docStatusStorage.initialize()

        fullDocs.initialize()

        textChunks.initialize()

        fullEntities.initialize()

        fullRelations.initialize()

        chunkEntityRelationGraph.initialize()

        chunksVdb.initialize()

        entitiesVdb.initialize()

        relationshipsVdb.initialize()
    }

    suspend fun persist() {
        docStatusStorage.indexDoneCallback()

        fullDocs.indexDoneCallback()

        textChunks.indexDoneCallback()

        fullEntities.indexDoneCallback()

        fullRelations.indexDoneCallback()

        chunkEntityRelationGraph.indexDoneCallback()

        chunksVdb.indexDoneCallback()

        entitiesVdb.indexDoneCallback()

        relationshipsVdb.indexDoneCallback()
    }

    suspend fun drop(): Map<String, Map<String, String>> =
        mapOf(
            "doc_status" to docStatusStorage.drop(),
            "full_docs" to fullDocs.drop(),
            "text_chunks" to textChunks.drop(),
            "full_entities" to fullEntities.drop(),
            "full_relations" to fullRelations.drop(),
            "graph" to chunkEntityRelationGraph.drop(),
            "chunks_vdb" to chunksVdb.drop(),
            "entities_vdb" to entitiesVdb.drop(),
            "relationships_vdb" to relationshipsVdb.drop(),
        )
}
