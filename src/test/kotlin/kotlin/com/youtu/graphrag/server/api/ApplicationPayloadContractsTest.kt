package com.youtu.graphrag.server.api

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ApplicationPayloadContractsTest {
    @Test
    fun `progress payload keeps strict websocket contract fields`() {
        val payload =
            progressPayload(
                stage = "retrieval",
                progress = 65,
                message = "Initial retrieval...",
                timestamp = "2026-05-13T00:00:00",
            )

        assertEquals(setOf("type", "stage", "progress", "message", "timestamp"), payload.keys)
        assertEquals("progress", payload["type"])
        assertEquals("retrieval", payload["stage"])
        assertEquals(65, payload["progress"])
    }

    @Test
    fun `stage event payload keeps strict websocket contract fields`() {
        val payload =
            stageEventPayload(
                type = "complete",
                stage = "construction",
                message = "Graph construction completed!",
                timestamp = "2026-05-13T00:00:00",
            )

        assertEquals(setOf("type", "stage", "message", "timestamp"), payload.keys)
        assertEquals("complete", payload["type"])
        assertEquals("construction", payload["stage"])
    }

    @Test
    fun `qa update payload merges extra fields without dropping envelope`() {
        val payload =
            qaUpdatePayload(
                stage = "ircot",
                extra = mapOf("step" to 2, "max_steps" to 5),
                timestamp = "2026-05-13T00:00:00",
            )

        assertEquals("qa_update", payload["type"])
        assertEquals("ircot", payload["stage"])
        assertEquals(2, payload["step"])
        assertEquals(5, payload["max_steps"])
        assertTrue(payload.keys.containsAll(listOf("type", "stage", "timestamp", "step", "max_steps")))
    }
}
