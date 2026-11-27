package hipporag

import hipporag.config.BaseConfig
import hipporag.embeddingmodel.BaseEmbeddingModel
import hipporag.embeddingmodel.EmbeddingModelFactory
import hipporag.embeddingmodel.getEmbeddingModel
import hipporag.graph.SimpleGraph
import hipporag.llm.BaseLLM
import hipporag.llm.getLlm
import hipporag.utils.LlmResult
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkStatic
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class OpenieRetryTest {
    @Test
    fun `normal indexing retries a cached failure and adds recovered graph edges`() {
        val dir = createTempDirectory("hippo-openie-retry").toFile()
        mockkStatic(::getLlm, ::getEmbeddingModel)
        try {
            val llm = mockk<BaseLLM>()
            val embeddings = mockk<BaseEmbeddingModel>()
            val factory = mockk<EmbeddingModelFactory>()
            every { getLlm(any()) } returns llm
            every { getEmbeddingModel() } returns factory
            every { factory.create(any(), any()) } returns embeddings
            every { embeddings.batchEncode(any(), any(), any()) } answers {
                Array(firstArg<List<String>>().size) { doubleArrayOf(1.0, 0.0) }
            }
            every { llm.infer(any()) } throws IllegalStateException("provider temporarily unavailable")
            val config = BaseConfig(saveDir = dir.path, llmName = "retry-llm", embeddingModelName = "retry-embedding")
            val documents = listOf("Paris is the capital of France.")
            val graphFile = dir.resolve("retry-llm_retry-embedding/graph.json")

            val first = HippoRag(initialConfig = config)
            first.index(documents)
            assertEquals(1, first.lastIndexingReport.failedChunkCount)
            assertEquals(0, SimpleGraph.load(graphFile).ecount())

            every { llm.infer(any()) } returnsMany
                listOf(
                    LlmResult("""{"named_entities":["Paris","France"]}""", emptyMap()),
                    LlmResult("""{"triples":[["Paris","capital of","France"]]}""", emptyMap()),
                )
            val second = HippoRag(initialConfig = config)
            second.index(documents)
            assertEquals(0, second.lastIndexingReport.failedChunkCount)
            assertTrue(SimpleGraph.load(graphFile).ecount() > 0)
        } finally {
            unmockkStatic(::getLlm, ::getEmbeddingModel)
            dir.deleteRecursively()
        }
    }
}
