package hipporag.prompts

import hipporag.utils.Message
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PromptTemplateManagerTest {
    @Test
    fun testRenderWithPromptUser() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val templates = manager.listTemplateNames()
        if ("rag_qa_musique" in templates) {
            val messages = manager.render("rag_qa_musique", promptUser = "Test passage")
            assertTrue(messages.isNotEmpty())
            assertTrue(messages.any { it.content.contains("Test passage") })
        }
    }

    @Test
    fun testRenderWithVariables() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val templates = manager.listTemplateNames()
        if ("ner" in templates) {
            val messages = manager.render("ner", variables = mapOf("passage" to "Test passage"))
            assertTrue(messages.isNotEmpty())
        }
    }

    @Test
    fun testListTemplateNames() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val templates = manager.listTemplateNames()
        assertTrue(templates.isNotEmpty())
    }

    @Test
    fun testIsTemplateNameValid() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val templates = manager.listTemplateNames()
        if (templates.isNotEmpty()) {
            assertTrue(manager.isTemplateNameValid(templates.first()))
        }
        assertFalse(manager.isTemplateNameValid("nonexistent_template"))
    }

    @Test
    fun testRenderInvalidTemplateThrows() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        assertFailsWith<IllegalArgumentException> {
            manager.render("nonexistent_template", promptUser = "test")
        }
    }

    @Test
    fun testRoleMapping() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "sys", "user" to "usr", "assistant" to "asst"),
            )

        val templates = manager.listTemplateNames()
        if ("ner" in templates) {
            val messages = manager.render("ner", promptUser = "Test")
            val roles = messages.map { it.role }.toSet()
            assertTrue(roles.any { it == "sys" || it == "usr" || it == "asst" })
        }
    }

    @Test
    fun testTemplateVariableSubstitution() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val templates = manager.listTemplateNames()
        if ("rag_qa_musique" in templates) {
            val messages =
                manager.render(
                    "rag_qa_musique",
                    variables = mapOf("prompt_user" to "custom value"),
                )
            assertTrue(messages.any { it.content.contains("custom value") })
        }
    }

    @Test
    fun testNerTemplateExists() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val templates = manager.listTemplateNames()
        assertTrue(templates.contains("ner"), "NER template should be loaded")
    }

    @Test
    fun testRagQaMusiqueTemplateExists() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val templates = manager.listTemplateNames()
        assertTrue(templates.contains("rag_qa_musique"), "RAG QA Musique template should be loaded")
    }

    @Test
    fun testTripleExtractionTemplateExists() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val templates = manager.listTemplateNames()
        assertTrue(templates.contains("triple_extraction"), "Triple extraction template should be loaded")
    }

    @Test
    fun testNerTemplateStructure() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val messages = manager.render("ner", promptUser = "Test passage")
        assertTrue(messages.size >= 2, "NER template should have at least system and user messages")
        assertTrue(messages.any { it.role == "system" })
        assertTrue(messages.any { it.role == "user" })
    }

    @Test
    fun testRagQaMusiqueTemplateStructure() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val messages = manager.render("rag_qa_musique", promptUser = "Question: Test?")
        assertTrue(messages.isNotEmpty())
        assertTrue(messages.any { it.role == "system" || it.role == "user" })
    }

    @Test
    fun testTripleExtractionTemplateStructure() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val messages =
            manager.render(
                "triple_extraction",
                variables = mapOf("passage" to "Test", "named_entity_json" to "{}"),
            )
        assertTrue(messages.isNotEmpty())
        assertTrue(messages.any { it.role == "system" })
    }

    @Test
    fun testTemplateWithMissingVariable() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val messages = manager.render("ner", variables = emptyMap())
        assertTrue(messages.isNotEmpty())
        assertTrue(messages.any { it.content.contains("\${passage}") })
    }

    @Test
    fun testMultipleVariableSubstitution() {
        val manager =
            PromptTemplateManager(
                roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
            )

        val messages =
            manager.render(
                "triple_extraction",
                variables =
                    mapOf(
                        "passage" to "Custom passage",
                        "named_entity_json" to """{"entities": ["A", "B"]}""",
                    ),
            )
        assertTrue(messages.any { it.content.contains("Custom passage") })
        assertTrue(messages.any { it.content.contains("entities") })
    }
}
