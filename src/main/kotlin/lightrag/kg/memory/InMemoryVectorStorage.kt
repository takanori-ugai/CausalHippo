package lightrag.kg.memory

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.store.embedding.CosineSimilarity
import io.github.oshai.kotlinlogging.KLogger
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.SerializationException
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import lightrag.core.types.BaseVectorStorage
import java.io.File
import java.io.IOException

private val logger = KotlinLogging.logger {}

class InMemoryVectorStorage(
    namespace: String,
    workspace: String,
    globalConfig: Map<String, Any?> = emptyMap(),
    embeddingFunc: EmbeddingModel,
    cosineThreshold: Double? = null,
) : BaseVectorStorage by InMemoryVectorStorageDelegate(
        namespace = namespace,
        workspace = workspace,
        globalConfig = globalConfig,
        embeddingFunc = embeddingFunc,
        cosineThreshold = cosineThreshold,
    )

private class InMemoryVectorStorageDelegate(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any?> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
    private val cosineThreshold: Double? = null,
) : BaseVectorStorage {
    override val cosineBetterThanThreshold: Double =
        cosineThreshold ?: (globalConfig["cosine_better_than_threshold"] as? Double ?: 0.2)
    override val metaFields: Set<String> = emptySet()

    private val vectors = mutableMapOf<String, List<Float>>()
    private val metadata = mutableMapOf<String, Metadata>()
    private val workingDir = File(globalConfig["working_dir"] as? String ?: "./rag_storage")
    private val file = File(workingDir, "vdb_$namespace.json")
    private val json =
        Json {
            ignoreUnknownKeys = true
            prettyPrint = true
        }

    override suspend fun initialize() {
        if (!workingDir.exists()) {
            workingDir.mkdirs()
        }
        if (!file.exists()) return

        try {
            val content = file.readText()
            if (content.isNotBlank()) {
                val loadedData = json.decodeFromString<Map<String, JsonObject>>(content)
                loadedData.forEach { (id, jsonObject) ->
                    val vec = (jsonObject["vector"] as? JsonArray)?.mapNotNull { it.jsonPrimitive.doubleOrNull?.toFloat() }
                    val metaObj = jsonObject["metadata"] as? JsonObject
                    val rawMap =
                        metaObj
                            ?.mapValues { (_, value) -> value.toAny() }
                            ?.filterValues { it != null }
                            ?.mapValues { it.value as Any } ?: emptyMap()
                    if (vec != null) {
                        vectors[id] = vec
                    }
                    metadata[id] = mapToMetadata(rawMap)
                }
                logger.info { "Loaded ${vectors.size} vectors for '$namespace' from ${file.absolutePath}" }
            }
        } catch (e: IOException) {
            logger.error(e) { "I/O error loading vector storage from ${file.absolutePath}" }
        } catch (e: SerializationException) {
            logger.error(e) { "Serialization error loading vector storage from ${file.absolutePath}" }
        }
    }

    override suspend fun indexDoneCallback() {
        try {
            if (!workingDir.exists()) {
                workingDir.mkdirs()
            }
            val jsonObject =
                JsonObject(
                    vectors.mapValues { (id, vec) ->
                        val meta = metadata[id]?.raw ?: emptyMap()
                        JsonObject(
                            mapOf(
                                "vector" to JsonArray(vec.map { JsonPrimitive(it) }),
                                "metadata" to meta.toJsonElement(),
                            ),
                        )
                    },
                )
            val content = json.encodeToString(jsonObject)
            file.writeText(content)
            logger.debug { "[$namespace/$workspace] Persisted ${vectors.size} vectors to ${file.absolutePath}" }
        } catch (e: IOException) {
            logger.error(e) { "I/O error saving vector storage to ${file.absolutePath}" }
        } catch (e: SerializationException) {
            logger.error(e) { "Serialization error saving vector storage to ${file.absolutePath}" }
        }
    }

    override suspend fun drop(): Map<String, String> {
        vectors.clear()
        metadata.clear()
        if (file.exists()) {
            runCatching { file.delete() }.onFailure {
                logger.warn(it) { "Failed to delete vector store file ${file.absolutePath}" }
            }
        }
        return mapOf("status" to "success", "message" to "data dropped and file removed at ${file.absolutePath}")
    }

    override suspend fun query(
        query: String,
        topK: Int,
        queryEmbedding: List<Float>?,
    ): List<Map<String, Any>> {
        val queryVec = queryEmbedding ?: embedText(embeddingFunc, query, logger)
        if (queryVec.isEmpty()) {
            logger.warn { "Query vector is empty for query: '$query'" }
            return emptyList()
        }

        if (vectors.isEmpty()) {
            logger.warn { "Vector storage '$namespace' is empty during query." }
            return emptyList()
        }

        logger.debug {
            "[$namespace/$workspace] Query='$query', topK=$topK, vectors=${vectors.size}, metadata=${metadata.size}"
        }

        val results =
            vectors
                .mapNotNull { (id, vec) ->
                    val meta = metadata[id]
                    if (meta == null) {
                        logger.warn { "Skipping vector '$id' in '$namespace' due to missing metadata." }
                        return@mapNotNull null
                    }
                    if (vec.isEmpty()) {
                        logger.warn { "Skipping vector '$id' in '$namespace' because it is empty." }
                        return@mapNotNull null
                    }
                    if (vec.size != queryVec.size) {
                        logger.warn {
                            "Skipping vector '$id' in '$namespace' due to dimension mismatch: stored=${vec.size}, query=${queryVec.size}"
                        }
                        return@mapNotNull null
                    }
                    val similarity =
                        CosineSimilarity.between(
                            Embedding(queryVec.toFloatArray()),
                            Embedding(vec.toFloatArray()),
                        )
                    Triple(id, similarity, meta)
                }.filter {
                    it.second >= cosineBetterThanThreshold
                }.sortedByDescending { it.second }
                .take(topK)

        if (results.isEmpty()) {
            logger.warn {
                "No results found for query: '$query' in '$namespace'. Vectors count: ${vectors.size}"
            }
        }

        return results.map { (id, score, meta) ->
            val raw = meta.raw
            raw + mapOf("id" to id, "score" to score, "distance" to score)
        }
    }

    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        if (data.isEmpty()) {
            logger.warn { "No data to upsert into vector storage '$namespace'." }
            return
        }

        data.forEach { (id, meta) ->
            val content = meta["content"] as? String
            val vector =
                when {
                    content != null -> embedText(embeddingFunc, content, logger)
                    meta["vector"] != null -> (meta["vector"] as? List<*>)?.mapNotNull { (it as? Number)?.toFloat() } ?: emptyList()
                    else -> emptyList()
                }

            if (vector.isNotEmpty()) {
                vectors[id] = vector
            } else if (!vectors.containsKey(id)) {
                logger.warn { "Upsert skipped for '$id' due to missing content or vector." }
                return@forEach
            }

            val mapped = mapToMetadata(meta)
            metadata[id] = mapped
        }

        logger.debug { "[$namespace/$workspace] Upserted ${data.size} items. Total vectors=${vectors.size}, metadata=${metadata.size}" }
    }

    override suspend fun deleteEntity(entityName: String) {
        val keysToDelete = metadata.filterValues { it.entityName == entityName }.keys
        delete(keysToDelete.toList())
    }

    override suspend fun deleteEntityRelation(entityName: String) {
        val keysToDelete = metadata.filterValues { it.srcId == entityName || it.tgtId == entityName }.keys
        delete(keysToDelete.toList())
    }

    override suspend fun getById(id: String): Map<String, Any>? = metadata[id]?.raw

    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> = ids.mapNotNull { metadata[it]?.raw }

    override suspend fun delete(ids: List<String>) {
        ids.forEach {
            vectors.remove(it)
            metadata.remove(it)
        }
    }

    override suspend fun getVectorsByIds(ids: List<String>): Map<String, List<Float>> =
        ids
            .mapNotNull { id ->
                vectors[id]?.let { id to it }
            }.toMap()
}

private fun embedText(
    embeddingModel: EmbeddingModel,
    text: String,
    logger: KLogger,
): List<Float> =
    try {
        val response = embeddingModel.embed(text)
        response.content().vector().toList()
    } catch (e: IllegalStateException) {
        logger.error(e) { "Illegal state embedding text: '$text'" }
        emptyList()
    } catch (e: IllegalArgumentException) {
        logger.error(e) { "Invalid input embedding text: '$text'" }
        emptyList()
    }

private fun mapToMetadata(meta: Map<String, Any>): Metadata {
    val content = meta["content"] as? String

    val vector = (meta["vector"] as? List<*>)?.mapNotNull { (it as? Number)?.toFloat() }
    val entityName = meta["entity_name"] as? String
    val srcId = meta["src_id"] as? String
    val tgtId = meta["tgt_id"] as? String
    return Metadata(
        content = content,
        vector = vector,
        entityName = entityName,
        srcId = srcId,
        tgtId = tgtId,
        raw = meta,
    )
}

private fun Any?.toJsonElement(): JsonElement =
    when (this) {
        null -> {
            JsonNull
        }

        is Boolean -> {
            JsonPrimitive(this)
        }

        is Number -> {
            JsonPrimitive(this.toString())
        }

        is String -> {
            JsonPrimitive(this)
        }

        is List<*> -> {
            JsonArray(this.map { it.toJsonElement() })
        }

        is Map<*, *> -> {
            JsonObject(this.entries.associate { it.key.toString() to it.value.toJsonElement() })
        }

        else -> {
            JsonPrimitive(this.toString())
        }
    }

private fun JsonElement.toAny(): Any? =
    when (this) {
        is JsonNull -> {
            null
        }

        is JsonPrimitive -> {
            if (isString) {
                content
            } else {
                booleanOrNull ?: longOrNull ?: doubleOrNull ?: content
            }
        }

        is JsonArray -> {
            this.map { it.toAny() }
        }

        is JsonObject -> {
            this.mapValues { it.value.toAny() }
        }
    }
