package com.youtu.graphrag.shared.llm

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class LlmOutputParserTest {
    @Test
    fun `clean llm content strips code fence and json prefix`() {
        val raw = "```json\n{\"a\":1}\n```"

        val cleaned = LlmOutputParser.cleanLlmContent(raw)

        assertEquals("{\"a\":1}", cleaned)
    }

    @Test
    fun `parse json value extracts first json block from prefixed text`() {
        val raw = "Here is the output:\n{\"sub_questions\":[{\"sub-question\":\"Q1\"}],\"involved_types\":{\"nodes\":[]}}"

        val parsed = LlmOutputParser.parseJsonObject(raw)

        assertNotNull(parsed)
        assertTrue(parsed.containsKey("sub_questions"))
    }

    @Test
    fun `parse json value repairs trailing commas and python literals`() {
        val raw = "{'sub_questions':[{'sub-question':'Q1',},], 'involved_types': {'nodes': None, 'relations': [], 'attributes': []},}"

        val parsed = LlmOutputParser.parseJsonObject(raw)

        assertNotNull(parsed)
        assertTrue(parsed.containsKey("sub_questions"))
    }

    @Test
    fun `parse json value returns null on non json output`() {
        val parsed = LlmOutputParser.parseJsonObject("not valid json at all")
        assertNull(parsed)
    }
}
