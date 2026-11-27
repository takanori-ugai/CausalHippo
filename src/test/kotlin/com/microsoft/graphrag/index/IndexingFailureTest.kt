package com.microsoft.graphrag.index

import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiChatModel
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.runBlocking
import org.junit.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class IndexingFailureTest {
    @Test
    fun `embedding stops at the first failed chunk or entity`() {
        runBlocking {
            val model = mockk<EmbeddingModel>()
            val failure = IllegalArgumentException("provider failed")
            every { model.embed("first") } throws failure
            val workflow = EmbedWorkflow(model)

            val chunkError =
                assertFailsWith<IllegalStateException> {
                    workflow.embedChunks(listOf(DocumentChunk("c1", "source", "first"), DocumentChunk("c2", "source", "second")))
                }
            val entityError =
                assertFailsWith<IllegalStateException> {
                    workflow.embedEntities(listOf(Entity("e1", "first", "type", "c1"), Entity("e2", "second", "type", "c2")))
                }

            assertEquals("GraphRAG embedding failed for chunk c1", chunkError.message)
            assertEquals("GraphRAG embedding failed for entity e1", entityError.message)
            assertTrue(generateSequence<Throwable>(chunkError) { it.cause }.any { it === failure })
            assertTrue(generateSequence<Throwable>(entityError) { it.cause }.any { it === failure })
            verify(exactly = 0) { model.embed("second") }
        }
    }

    @Test
    fun `graph extraction rejects malformed responses and accepts valid empty results`() {
        runBlocking {
            val model = mockk<OpenAiChatModel>()
            val workflow = ExtractGraphWorkflow(model)
            val chunks = listOf(DocumentChunk("c1", "source", "first"))

            for (response in listOf("", "no JSON", """{"entities":42}""")) {
                every { model.chat(any<List<ChatMessage>>()) } returns chatResponse(response)
                assertFailsWith<IllegalStateException> { workflow.extract(chunks) }
            }

            every { model.chat(any<List<ChatMessage>>()) } returns chatResponse("""{"entities":[],"relationships":[]}""")
            val result = workflow.extract(chunks)
            assertTrue(result.entities.isEmpty())
            assertTrue(result.relationships.isEmpty())
        }
    }

    private fun chatResponse(text: String): ChatResponse = ChatResponse.builder().aiMessage(AiMessage.from(text)).build()
}
