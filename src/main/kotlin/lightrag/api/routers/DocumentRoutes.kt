package lightrag.api.routers

import io.ktor.http.HttpStatusCode
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.routing.delete
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.core.LightRAG

/**
 * Represents the request body for inserting a single text document.
 * @property text The text content to be inserted.
 * @property fileSource Optional source identifier for the text.
 */
@Serializable
data class InsertTextRequest(
    val text: String,
    @SerialName("file_source") val fileSource: String? = null,
)

/**
 * Represents the request body for inserting multiple text documents.
 * @property texts A list of text contents to be inserted.
 * @property fileSources Optional list of source identifiers for the texts.
 */
@Serializable
data class InsertTextsRequest(
    val texts: List<String>,
    @SerialName("file_sources") val fileSources: List<String>? = null,
)

/**
 * Represents the response after an insert operation.
 * @property status The status of the operation (e.g., "success").
 * @property message A descriptive message about the outcome.
 * @property trackId A unique tracking ID for the operation.
 */
@Serializable
data class InsertResponse(
    val status: String,
    val message: String,
    @SerialName("track_id") val trackId: String,
)

/**
 * Represents the request body for deleting documents.
 * @property docIds A list of document IDs to be deleted.
 */
@Serializable
data class DeleteDocRequest(
    @SerialName("doc_ids") val docIds: List<String>,
)

/**
 * Configures the document-related routes for the Ktor application.
 *
 * This function sets up endpoints for uploading, inserting, and deleting text documents,
 * as well as checking the processing status.
 *
 * @param rag The LightRAG instance to be used for document operations.
 */
fun Application.configureDocumentRoutes(rag: LightRAG) {
    routing {
        route("/documents") {
            post("/upload") {
                // Simplified upload handling
                call.respond(InsertResponse("success", "File uploaded (mock)", "mock_track_id"))
            }

            post("/text") {
                val request = call.receive<InsertTextRequest>()
                if (request.text.isBlank()) {
                    call.respond(
                        HttpStatusCode.BadRequest,
                        mapOf("status" to "error", "message" to "text cannot be blank"),
                    )
                    return@post
                }
                runCatching { rag.insert(request.text, request.fileSource) }
                    .onSuccess { trackId ->
                        call.respond(InsertResponse("success", "Text received", trackId))
                    }.onFailure {
                        call.respond(
                            HttpStatusCode.InternalServerError,
                            mapOf("status" to "error", "message" to (it.message ?: "insert failed")),
                        )
                    }
            }

            post("/texts") {
                val request = call.receive<InsertTextsRequest>()
                if (request.texts.isEmpty() || request.texts.any { it.isBlank() }) {
                    call.respond(
                        HttpStatusCode.BadRequest,
                        mapOf("status" to "error", "message" to "texts cannot be empty or blank"),
                    )
                    return@post
                }
                runCatching { rag.insert(request.texts, request.fileSources) }
                    .onSuccess { trackId ->
                        call.respond(InsertResponse("success", "Texts received", trackId))
                    }.onFailure {
                        call.respond(
                            HttpStatusCode.InternalServerError,
                            mapOf("status" to "error", "message" to (it.message ?: "insert failed")),
                        )
                    }
            }

            get("/status_counts") {
                call.respond(rag.getProcessingStatus())
            }

            delete("/delete_document") {
                val request = call.receive<DeleteDocRequest>()
                // Simplified deletion
                val result = request.docIds.map { rag.deleteByDocId(it) }
                call.respond(result)
            }

            delete("/drop") {
                val result = rag.storageManager.drop()
                call.respond(result)
            }
        }
    }
}
