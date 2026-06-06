package com.youtu.graphrag.parity

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

object JsonParityUtils {
    private val mapper = jacksonObjectMapper()

    fun canonicalizeJson(raw: String): String {
        val root = mapper.readTree(raw)
        return canonicalize(root).toString()
    }

    fun assertJsonEquivalent(
        expected: String,
        actual: String,
    ) {
        assertEquals(canonicalizeJson(expected), canonicalizeJson(actual), "JSON payloads differ")
    }

    fun assertApproximatelyEquals(
        expected: Double,
        actual: Double,
        tolerance: Double,
        label: String = "value",
    ) {
        require(tolerance >= 0.0) { "tolerance must be >= 0.0" }
        val delta = abs(expected - actual)
        assertTrue(
            delta <= tolerance,
            "$label expected=$expected actual=$actual tolerance=$tolerance delta=$delta",
        )
    }

    private fun canonicalize(node: JsonNode): JsonNode {
        if (node.isObject) {
            val canonical = mapper.nodeFactory.objectNode()
            node.fieldNames().asSequence().toList().sorted().forEach { fieldName ->
                canonical.set<JsonNode>(fieldName, canonicalize(node.get(fieldName)))
            }
            return canonical
        }

        if (node.isArray) {
            val canonical = mapper.nodeFactory.arrayNode()
            node.forEach { child -> canonical.add(canonicalize(child)) }
            return canonical
        }

        return node
    }
}
