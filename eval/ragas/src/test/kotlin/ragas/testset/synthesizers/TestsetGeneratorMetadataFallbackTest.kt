package ragas.testset.synthesizers

import ragas.model.SingleTurnSample
import ragas.testset.graph.Node
import ragas.testset.graph.NodeType
import ragas.testset.graph.Relationship
import kotlin.test.Test
import kotlin.test.assertEquals

class TestsetGeneratorMetadataFallbackTest {
    @Test
    fun scoreSingleHopCandidatesUsesSourceDocumentMetadataFallback() {
        val generator = TestsetGenerator()
        val plainChunk =
            Node(
                id = "plain",
                type = NodeType.CHUNK,
                properties = mutableMapOf("page_content" to "same content baseline text for scoring"),
            )
        val enrichedChunk =
            Node(
                id = "enriched",
                type = NodeType.CHUNK,
                properties =
                    mutableMapOf(
                        "page_content" to "same content baseline text for scoring",
                        "source_document_summary" to "summary",
                        "source_document_entities" to "Kotlin, JVM",
                        "source_document_topic" to "language",
                    ),
            )

        val method =
            TestsetGenerator::class.java
                .getDeclaredMethod(
                    "scoreSingleHopCandidates",
                    List::class.java,
                    List::class.java,
                ).apply { isAccessible = true }

        @Suppress("UNCHECKED_CAST")
        val scored =
            method.invoke(generator, listOf(plainChunk, enrichedChunk), listOf<Relationship>()) as List<Pair<Node, Double>>

        assertEquals("enriched", scored.first().first.id)
    }

    @Test
    fun generateSingleHopSamplesUsesSourceDocumentMetadataFallback() {
        val generator = TestsetGenerator()
        val chunk =
            Node(
                id = "chunk",
                type = NodeType.CHUNK,
                properties =
                    mutableMapOf(
                        "page_content" to "Chunk content about Kotlin and coroutines.",
                        "source_document_summary" to "Kotlin summary",
                        "source_document_entities" to "Kotlin, coroutines",
                        "source_document_topic" to "programming",
                    ),
            )

        val method =
            TestsetGenerator::class.java
                .getDeclaredMethod(
                    "generateSingleHopSamples",
                    List::class.java,
                ).apply { isAccessible = true }

        @Suppress("UNCHECKED_CAST")
        val samples = method.invoke(generator, listOf(chunk)) as List<TestsetSample>
        val sample = samples.single()
        val eval = sample.evalSample as SingleTurnSample

        assertEquals("single_hop_entity", sample.synthesizerName)
        assertEquals("Kotlin summary", eval.response)
    }
}
