package com.youtu.graphrag.server.api

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.server.websocket.DefaultWebSocketServerSession
import io.ktor.websocket.Frame
import io.ktor.websocket.send
import java.util.concurrent.ConcurrentHashMap

class WebSocketConnectionManager {
    private val logger = KotlinLogging.logger {}
    private val objectMapper = jacksonObjectMapper()
    private val activeConnections = ConcurrentHashMap<String, DefaultWebSocketServerSession>()

    fun connect(
        clientId: String,
        session: DefaultWebSocketServerSession,
    ) {
        activeConnections[clientId] = session
    }

    fun disconnect(clientId: String) {
        activeConnections.remove(clientId)
    }

    suspend fun sendMessage(
        clientId: String,
        message: Map<String, Any?>,
    ) {
        val session = activeConnections[clientId] ?: return

        runCatching {
            val payload = objectMapper.writeValueAsString(message)
            session.send(Frame.Text(payload))
        }.onFailure { error ->
            logger.warn(error) { "Error sending websocket message to client '$clientId'" }
            disconnect(clientId)
        }
    }
}
