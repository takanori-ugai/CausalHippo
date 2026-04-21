package lightrag.api.routers

import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.core.LightRAG
import lightrag.utils.computeMd5

private val logger = KotlinLogging.logger {}

/**
 * Represents the request body for updating an entity.
 * @property entityName The name of the entity to update.
 * @property updatedData A map of the data to update.
 * @property allowRename Whether to allow renaming the entity.
 * @property allowMerge Whether to allow merging the entity.
 */
@Serializable
data class EntityUpdateRequest(
    @SerialName("entity_name")
    val entityName: String,
    // Simplified to String map
    @SerialName("updated_data")
    val updatedData: Map<String, String>,
    @SerialName("allow_rename")
    val allowRename: Boolean = false,
    @SerialName("allow_merge")
    val allowMerge: Boolean = false,
)

/**
 * Represents the request body for updating a relation.
 * @property sourceId The source entity ID.
 * @property targetId The target entity ID.
 * @property updatedData The updated relation data.
 */
@Serializable
data class RelationUpdateRequest(
    @SerialName("source_id")
    val sourceId: String,
    @SerialName("target_id")
    val targetId: String,
    @SerialName("updated_data")
    val updatedData: Map<String, String>,
)

/**
 * Represents the request body for creating an entity.
 * @property entityName The name of the entity to create.
 * @property entityData A map of the entity's data.
 */
@Serializable
data class EntityCreateRequest(
    @SerialName("entity_name")
    val entityName: String,
    // Simplified
    @SerialName("entity_data")
    val entityData: Map<String, String>,
)

/**
 * Represents the request body for creating a relation.
 * @property sourceEntity The source entity of the relation.
 * @property targetEntity The target entity of the relation.
 * @property relationData A map of the relation's data.
 */
@Serializable
data class RelationCreateRequest(
    @SerialName("source_entity")
    val sourceEntity: String,
    @SerialName("target_entity")
    val targetEntity: String,
    // Simplified
    @SerialName("relation_data")
    val relationData: Map<String, String>,
)

/**
 * Represents the request body for merging entities.
 * @property entitiesToChange The list of entities to merge.
 * @property entityToChangeInto The entity to merge into.
 */
@Serializable
data class EntityMergeRequest(
    @SerialName("entities_to_change")
    val entitiesToChange: List<String>,
    @SerialName("entity_to_change_into")
    val entityToChangeInto: String,
)

/**
 * Configures the graph-related routes for the Ktor application.
 *
 * This function sets up endpoints for interacting with the knowledge graph,
 * including listing labels, retrieving graphs, and creating/editing entities and relations.
 *
 * @param rag The LightRAG instance to be used for graph operations.
 */
fun Application.configureGraphRoutes(rag: LightRAG) {
    logger.info { "Configuring graph routes for LightRAG storage manager ${rag.storageManager}" }
    routing {
        get("/graph/label/list") {
            runCatching { rag.storageManager.chunkEntityRelationGraph.getAllLabels() }
                .onSuccess { call.respond(it) }
                .onFailure {
                    logger.error(it) { "Error getting graph labels" }
                    call.respond(
                        HttpStatusCode.InternalServerError,
                        mapOf("status" to "error", "message" to "Error getting graph labels: ${it.message}"),
                    )
                }
        }

        get("/graph/label/popular") {
            val limit =
                call.request.queryParameters["limit"]
                    ?.toIntOrNull()
                    ?.coerceIn(1, 1000)
                    ?: 300
            runCatching { rag.storageManager.chunkEntityRelationGraph.getPopularLabels(limit) }
                .onSuccess { call.respond(it) }
                .onFailure {
                    logger.error(it) { "Error getting popular labels" }
                    call.respond(
                        HttpStatusCode.InternalServerError,
                        mapOf("status" to "error", "message" to "Error getting popular labels: ${it.message}"),
                    )
                }
        }

        get("/graph/label/search") {
            val query = call.request.queryParameters["q"]
            if (query.isNullOrBlank()) {
                call.respond(HttpStatusCode.BadRequest, mapOf("status" to "error", "message" to "Missing q parameter"))
                return@get
            }
            val limit =
                call.request.queryParameters["limit"]
                    ?.toIntOrNull()
                    ?.coerceIn(1, 100)
                    ?: 50
            runCatching { rag.storageManager.chunkEntityRelationGraph.searchLabels(query, limit) }
                .onSuccess { call.respond(it) }
                .onFailure {
                    logger.error(it) { "Error searching labels for query '$query'" }
                    call.respond(
                        HttpStatusCode.InternalServerError,
                        mapOf("status" to "error", "message" to "Error searching labels: ${it.message}"),
                    )
                }
        }

        get("/graph/entity/exists") {
            val name = call.request.queryParameters["name"]
            if (name.isNullOrBlank()) {
                call.respond(HttpStatusCode.BadRequest, mapOf("status" to "error", "message" to "Missing name parameter"))
                return@get
            }
            runCatching { rag.storageManager.chunkEntityRelationGraph.hasNode(name) }
                .onSuccess { call.respond(mapOf("exists" to it)) }
                .onFailure {
                    logger.error(it) { "Error checking entity existence for $name" }
                    call.respond(
                        HttpStatusCode.InternalServerError,
                        mapOf("status" to "error", "message" to "Error checking entity existence: ${it.message}"),
                    )
                }
        }

        route("/graph") {
            post("/entity/edit") {
                val request = call.receive<EntityUpdateRequest>()
                val graph = rag.storageManager.chunkEntityRelationGraph
                val fullEntities = rag.storageManager.fullEntities
                val entitiesVdb = rag.storageManager.entitiesVdb
                val relationshipsVdb = rag.storageManager.relationshipsVdb

                val existing = graph.getNode(request.entityName)
                if (existing == null) {
                    call.respond(
                        HttpStatusCode.NotFound,
                        mapOf("status" to "error", "message" to "Entity '${request.entityName}' not found"),
                    )
                    return@post
                }

                val desiredName = request.updatedData["entity_name"] ?: request.entityName
                val renaming = desiredName != request.entityName

                if (renaming && !request.allowRename) {
                    call.respond(
                        HttpStatusCode.BadRequest,
                        mapOf("status" to "error", "message" to "Renaming not allowed; set allow_rename=true"),
                    )
                    return@post
                }

                val targetExists = renaming && graph.hasNode(desiredName)
                if (targetExists && !request.allowMerge) {
                    call.respond(
                        HttpStatusCode.BadRequest,
                        mapOf(
                            "status" to "error",
                            "message" to "Target entity '$desiredName' already exists; set allow_merge=true to merge",
                        ),
                    )
                    return@post
                }

                val operationSummary =
                    mutableMapOf<String, Any?>(
                        "merged" to targetExists,
                        "merge_status" to if (targetExists) "success" else "not_attempted",
                        "merge_error" to null,
                        "operation_status" to "success",
                        "target_entity" to if (renaming) desiredName else null,
                        "final_entity" to desiredName,
                        "renamed" to renaming,
                    )

                runCatching {
                    val baseNode =
                        if (targetExists) {
                            val merged = graph.getNode(desiredName)?.toMutableMap() ?: mutableMapOf()
                            existing.forEach { (k, v) -> merged.putIfAbsent(k, v) }
                            merged
                        } else {
                            existing.toMutableMap()
                        }
                    baseNode.putAll(request.updatedData)
                    baseNode["entity_id"] = desiredName

                    if (renaming) {
                        val connectedEdges = graph.getNodeEdges(request.entityName).orEmpty()
                        connectedEdges.forEach { (src, tgt) ->
                            val other = if (src == request.entityName) tgt else src
                            val edgeData = graph.getEdge(src, tgt)?.toMutableMap() ?: mutableMapOf()
                            edgeData["src_id"] = desiredName
                            edgeData["tgt_id"] = other
                            graph.upsertEdge(desiredName, other, edgeData)
                        }
                        graph.deleteNode(request.entityName)
                        entitiesVdb.deleteEntity(request.entityName)
                        relationshipsVdb.deleteEntityRelation(request.entityName)
                        fullEntities.delete(listOf(request.entityName))
                    }

                    graph.upsertNode(desiredName, baseNode)
                    fullEntities.upsert(mapOf(desiredName to baseNode))

                    val description = baseNode["description"] ?: ""
                    val vdbData =
                        mapOf(
                            computeMd5(desiredName) to
                                mapOf(
                                    "content" to "$desiredName\n$description",
                                    "entity_name" to desiredName,
                                ),
                        )
                    entitiesVdb.upsert(vdbData)

                    mapOf("status" to "success", "message" to "Entity updated successfully", "data" to baseNode)
                }.onSuccess { response ->
                    call.respond(
                        response + mapOf("operation_summary" to operationSummary),
                    )
                }.onFailure {
                    logger.error(it) { "Error updating entity '${request.entityName}'" }
                    call.respond(
                        HttpStatusCode.InternalServerError,
                        mapOf("status" to "error", "message" to "Error updating entity: ${it.message}"),
                    )
                }
            }

            post("/relation/edit") {
                val request = call.receive<RelationUpdateRequest>()
                val graph = rag.storageManager.chunkEntityRelationGraph
                val existing = graph.getEdge(request.sourceId, request.targetId)
                if (existing == null) {
                    call.respond(
                        HttpStatusCode.NotFound,
                        mapOf(
                            "status" to "error",
                            "message" to "Relation between '${request.sourceId}' and '${request.targetId}' not found",
                        ),
                    )
                    return@post
                }

                runCatching {
                    val updated = existing.toMutableMap()
                    updated.putAll(request.updatedData)
                    updated["src_id"] = request.sourceId
                    updated["tgt_id"] = request.targetId
                    graph.upsertEdge(request.sourceId, request.targetId, updated)

                    val key = listOf(request.sourceId, request.targetId).sorted().joinToString("#")
                    rag.storageManager.fullRelations.upsert(mapOf(key to updated))

                    val description = updated["description"] ?: ""
                    val keywords = updated["keywords"] ?: ""
                    val relContent = "$keywords\t${request.sourceId}\n${request.targetId}\n$description"
                    val vdbData =
                        mapOf(
                            computeMd5(key) to
                                mapOf(
                                    "content" to relContent,
                                    "src_id" to request.sourceId,
                                    "tgt_id" to request.targetId,
                                ),
                        )
                    rag.storageManager.relationshipsVdb.upsert(vdbData)

                    updated
                }.onSuccess {
                    call.respond(
                        mapOf(
                            "status" to "success",
                            "message" to "Relation updated successfully",
                            "data" to it,
                        ),
                    )
                }.onFailure {
                    logger.error(it) {
                        "Error updating relation between '${request.sourceId}' and '${request.targetId}'"
                    }
                    call.respond(
                        HttpStatusCode.InternalServerError,
                        mapOf("status" to "error", "message" to "Error updating relation: ${it.message}"),
                    )
                }
            }

            post("/entity/create") {
                val request = call.receive<EntityCreateRequest>()
                val graph = rag.storageManager.chunkEntityRelationGraph
                if (request.entityName.isBlank()) {
                    call.respond(
                        HttpStatusCode.BadRequest,
                        mapOf("status" to "error", "message" to "entity_name cannot be blank"),
                    )
                    return@post
                }

                runCatching {
                    require(!graph.hasNode(request.entityName)) { "Entity '${request.entityName}' already exists" }
                    val data = request.entityData.toMutableMap()
                    data.putIfAbsent("entity_id", request.entityName)
                    graph.upsertNode(request.entityName, data)
                    rag.storageManager.fullEntities.upsert(mapOf(request.entityName to data))

                    val description = data["description"] ?: ""
                    val vdbData =
                        mapOf(
                            computeMd5(request.entityName) to
                                mapOf(
                                    "content" to "${request.entityName}\n$description",
                                    "entity_name" to request.entityName,
                                ),
                        )
                    rag.storageManager.entitiesVdb.upsert(vdbData)

                    data
                }.onSuccess {
                    call.respond(
                        mapOf(
                            "status" to "success",
                            "message" to "Entity '${request.entityName}' created successfully",
                            "data" to it,
                        ),
                    )
                }.onFailure {
                    logger.error(it) { "Error creating entity '${request.entityName}'" }
                    val status =
                        if (it is IllegalArgumentException) {
                            HttpStatusCode.BadRequest
                        } else {
                            HttpStatusCode.InternalServerError
                        }
                    call.respond(
                        status,
                        mapOf("status" to "error", "message" to "Error creating entity: ${it.message}"),
                    )
                }
            }

            post("/relation/create") {
                val request = call.receive<RelationCreateRequest>()
                val graph = rag.storageManager.chunkEntityRelationGraph
                if (!graph.hasNode(request.sourceEntity) || !graph.hasNode(request.targetEntity)) {
                    call.respond(
                        HttpStatusCode.BadRequest,
                        mapOf("status" to "error", "message" to "Both source and target entities must exist"),
                    )
                    return@post
                }

                runCatching {
                    val data = request.relationData.toMutableMap()
                    data["src_id"] = request.sourceEntity
                    data["tgt_id"] = request.targetEntity
                    graph.upsertEdge(request.sourceEntity, request.targetEntity, data)

                    val key = listOf(request.sourceEntity, request.targetEntity).sorted().joinToString("#")
                    rag.storageManager.fullRelations.upsert(mapOf(key to data))

                    val description = data["description"] ?: ""
                    val keywords = data["keywords"] ?: ""
                    val relContent = "$keywords\t${request.sourceEntity}\n${request.targetEntity}\n$description"
                    val vdbData =
                        mapOf(
                            computeMd5(key) to
                                mapOf(
                                    "content" to relContent,
                                    "src_id" to request.sourceEntity,
                                    "tgt_id" to request.targetEntity,
                                ),
                        )
                    rag.storageManager.relationshipsVdb.upsert(vdbData)

                    data
                }.onSuccess {
                    call.respond(
                        mapOf(
                            "status" to "success",
                            "message" to "Relation created successfully between '${request.sourceEntity}' and '${request.targetEntity}'",
                            "data" to it,
                        ),
                    )
                }.onFailure {
                    logger.error(it) {
                        "Error creating relation between '${request.sourceEntity}' and '${request.targetEntity}'"
                    }
                    call.respond(
                        HttpStatusCode.InternalServerError,
                        mapOf("status" to "error", "message" to "Error creating relation: ${it.message}"),
                    )
                }
            }

            post("/entities/merge") {
                val request = call.receive<EntityMergeRequest>()
                val graph = rag.storageManager.chunkEntityRelationGraph

                if (request.entitiesToChange.isEmpty()) {
                    call.respond(
                        HttpStatusCode.BadRequest,
                        mapOf("status" to "error", "message" to "entities_to_change cannot be empty"),
                    )
                    return@post
                }

                runCatching {
                    require(graph.hasNode(request.entityToChangeInto)) {
                        "Target entity '${request.entityToChangeInto}' does not exist"
                    }

                    request.entitiesToChange.forEach { source ->
                        require(graph.hasNode(source)) { "Source entity '$source' does not exist" }
                    }

                    request.entitiesToChange.forEach { source ->
                        val edges = graph.getNodeEdges(source).orEmpty()
                        edges.forEach { (src, tgt) ->
                            val other = if (src == source) tgt else src
                            val edgeData = graph.getEdge(src, tgt)?.toMutableMap() ?: mutableMapOf()
                            edgeData["src_id"] = request.entityToChangeInto
                            edgeData["tgt_id"] = other
                            graph.upsertEdge(request.entityToChangeInto, other, edgeData)
                        }
                        graph.deleteNode(source)
                        rag.storageManager.entitiesVdb.deleteEntity(source)
                        rag.storageManager.relationshipsVdb.deleteEntityRelation(source)
                        rag.storageManager.fullEntities.delete(listOf(source))
                    }

                    mapOf(
                        "merged_entity" to request.entityToChangeInto,
                        "deleted_entities" to request.entitiesToChange,
                    )
                }.onSuccess {
                    call.respond(
                        mapOf(
                            "status" to "success",
                            "message" to
                                "Successfully merged ${request.entitiesToChange.size} entities into '${request.entityToChangeInto}'",
                            "data" to it,
                        ),
                    )
                }.onFailure {
                    logger.error(it) {
                        "Error merging entities ${request.entitiesToChange} into '${request.entityToChangeInto}'"
                    }
                    val status =
                        if (it is IllegalArgumentException) {
                            HttpStatusCode.BadRequest
                        } else {
                            HttpStatusCode.InternalServerError
                        }
                    call.respond(
                        status,
                        mapOf("status" to "error", "message" to "Error merging entities: ${it.message}"),
                    )
                }
            }
        }

        get("/graph/graphs") {
            val label = call.request.queryParameters["label"]
            if (label.isNullOrBlank()) {
                call.respond(HttpStatusCode.BadRequest, mapOf("status" to "error", "message" to "Missing label parameter"))
                return@get
            }
            val maxDepth = call.request.queryParameters["max_depth"]?.toIntOrNull() ?: 3
            val maxNodes = call.request.queryParameters["max_nodes"]?.toIntOrNull() ?: 1000

            runCatching {
                rag.storageManager.chunkEntityRelationGraph.getKnowledgeGraph(
                    nodeLabel = label,
                    maxDepth = maxDepth,
                    maxNodes = maxNodes,
                )
            }.onSuccess { graph ->
                call.respond(graph)
            }.onFailure {
                logger.error(it) { "Error getting knowledge graph for label '$label'" }
                call.respond(
                    HttpStatusCode.InternalServerError,
                    mapOf("status" to "error", "message" to "Error getting knowledge graph: ${it.message}"),
                )
            }
        }
    }
}
