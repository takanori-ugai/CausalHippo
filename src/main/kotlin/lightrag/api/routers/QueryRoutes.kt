package lightrag.api.routers

import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.server.response.respondTextWriter
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import kotlinx.coroutines.flow.collect
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.core.LightRAG
import lightrag.core.QueryParam

/**
 * Represents the request body for a query.
 * @property query The query string.
 * @property mode The query mode (e.g., "mix").
 * @property onlyNeedContext Whether to return only the context.
 * @property responseType The desired response type.
 * @property topK The number of top results to return.
 * @property maxTokenForTextUnit The maximum number of tokens for a text unit.
 */
@Serializable
data class QueryRequest(
    val query: String,
    val mode: String = "mix",
    @SerialName("only_need_context")
    val onlyNeedContext: Boolean? = null,
    @SerialName("response_type")
    val responseType: String? = null,
    @SerialName("top_k")
    val topK: Int? = null,
    @SerialName("max_token_for_text_unit")
    val maxTokenForTextUnit: Int? = null,
    @SerialName("high_level_keywords")
    val highLevelKeywords: List<String>? = null,
    @SerialName("low_level_keywords")
    val lowLevelKeywords: List<String>? = null,
)

/**
 * Represents the response to a query.
 * @property response The response string.
 */
@Serializable
data class QueryResponse(
    val response: String,
)

/**
 * Configures the query-related routes for the Ktor application.
 *
 * This function sets up endpoints for handling standard and streaming queries.
 *
 * @param rag The LightRAG instance to be used for query operations.
 */
fun Application.configureQueryRoutes(rag: LightRAG) {
    routing {
        route("/query") {
            post {
                val request = call.receive<QueryRequest>()

                val param =
                    QueryParam(
                        mode = request.mode,
                        onlyNeedContext = request.onlyNeedContext ?: false,
                        responseType = request.responseType,
                        topK = request.topK ?: 10,
                        hlKeywords = request.highLevelKeywords ?: emptyList(),
                        llKeywords = request.lowLevelKeywords ?: emptyList(),
                    )

                val result = rag.query(request.query, param)
                call.respond(QueryResponse(result?.content ?: "No result generated."))
            }

            post("/stream") {
                val request = call.receive<QueryRequest>()
                val param =
                    QueryParam(
                        mode = request.mode,
                        onlyNeedContext = request.onlyNeedContext ?: false,
                        responseType = request.responseType,
                        topK = request.topK ?: 10,
                        hlKeywords = request.highLevelKeywords ?: emptyList(),
                        llKeywords = request.lowLevelKeywords ?: emptyList(),
                        stream = true,
                    )

                val result = rag.query(request.query, param)
                when {
                    result == null -> {
                        call.respondText("Query failed", status = HttpStatusCode.InternalServerError)
                    }

                    result.isStreaming && result.responseIterator != null -> {
                        call.respondTextWriter(ContentType.Text.Plain) {
                            result.responseIterator.collect { token ->
                                write(token)
                                flush()
                            }
                        }
                    }

                    else -> {
                        call.respond(QueryResponse(result.content ?: "No result generated."))
                    }
                }
            }
        }
    }
}
