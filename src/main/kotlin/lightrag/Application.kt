package lightrag

import dev.langchain4j.model.chat.ChatModel
import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationStopping
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.http.content.staticResources // Added missing import
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.plugins.swagger.swaggerUI
import io.ktor.server.response.respond
import io.ktor.server.response.respondRedirect
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.modules.SerializersModule
import kotlinx.serialization.modules.contextual
import lightrag.api.routers.configureDocumentRoutes
import lightrag.api.routers.configureGraphRoutes
import lightrag.api.routers.configureOllamaRoutes
import lightrag.api.routers.configureQueryRoutes
import lightrag.core.LightRAG
import lightrag.di.appModule
import lightrag.services.StorageManager
import lightrag.utils.AnyValueSerializer
import org.koin.ktor.ext.inject
import org.koin.ktor.plugin.Koin

private val logger = KotlinLogging.logger {}

/**
 * The main entry point of the application.
 */
fun main() {
    embeddedServer(Netty, port = 8080, host = "0.0.0.0", module = Application::module)
        .start(wait = true)
}

/**
 * The main module of the application.
 */
fun Application.module() {
    install(ContentNegotiation) {
        json(
            Json {
                ignoreUnknownKeys = true
                prettyPrint = true
                serializersModule =
                    SerializersModule {
                        contextual(Any::class, AnyValueSerializer)
                    }
            },
        )
    }
    install(CORS) {
        anyHost()
        allowHeader("Content-Type")
    }

    install(Koin) {
        modules(appModule)
    }

    val rag by inject<LightRAG>()
    val storageManager by inject<StorageManager>()
    val chatModel by inject<ChatModel>()

    val resetStorage = System.getenv("LIGHTRAG_RESET_STORAGE")?.equals("true", ignoreCase = true) == true
    if (resetStorage) {
        logger.warn { "LIGHTRAG_RESET_STORAGE=true detected. Dropping all persisted stores before initialization." }
        runBlocking { storageManager.drop() }
    }

    runBlocking { storageManager.initialize() }
    runBlocking { rag.rebuildDerivedStorageIfEmpty() }
    environment.monitor.subscribe(ApplicationStopping) {
        runBlocking { storageManager.persist() }
    }

    routing {
        get("/") { call.respondRedirect("/swagger") }
        swaggerUI(path = "swagger", swaggerFile = "openapi/documentation.yaml")
        staticResources("/ui", "static")
        get("/app") { call.respondRedirect("/ui/index.html") }
    }

    configureDocumentRoutes(rag)
    configureQueryRoutes(rag)
    configureGraphRoutes(rag)
    configureOllamaRoutes(rag, chatModel)
}
