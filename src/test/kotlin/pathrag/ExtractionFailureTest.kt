package pathrag

import kotlinx.coroutines.runBlocking
import org.junit.Test
import pathrag.operate.extractEntities
import pathrag.storage.NanoVectorDBStorage
import pathrag.storage.NetworkXStorage
import pathrag.utils.EmbeddingFunc
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ExtractionFailureTest {
    @Test
    fun `retries malformed extraction and accepts valid empty results`() {
        runBlocking {
            val outputs = ArrayDeque(listOf("malformed", """{"entities":[],"relationships":[]}"""))
            extract(outputs)
            assertTrue(outputs.isEmpty())
        }
    }

    @Test
    fun `surfaces repeated malformed extraction with the source chunk ID`() {
        runBlocking {
            val outputs = ArrayDeque(listOf("malformed", "still malformed"))
            val failure = assertFailsWith<IllegalStateException> { extract(outputs) }
            assertTrue(failure.message.orEmpty().contains("chunk-1"))
            assertEquals(0, outputs.size)
        }
    }

    private suspend fun extract(outputs: ArrayDeque<String>) {
        val llm: suspend (String, String?, List<Map<String, String>>, Boolean, Boolean, Int?, Any?) -> String =
            { _, _, _, _, _, _, _ -> outputs.removeFirst() }
        val embed = EmbeddingFunc(1, 100, { inputs -> inputs.map { doubleArrayOf(1.0) } })
        extractEntities(
            chunks = mapOf("chunk-1" to mapOf("content" to "Some document text")),
            knowledgeGraphInst = NetworkXStorage("graph", emptyMap(), null),
            entityVdb = NanoVectorDBStorage("entities", emptyMap(), embed),
            relationshipsVdb = NanoVectorDBStorage("relations", emptyMap(), embed),
            globalConfig = mapOf("llm_model_func" to llm),
        )
    }
}
