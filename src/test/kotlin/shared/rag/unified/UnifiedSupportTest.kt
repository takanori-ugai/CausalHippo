package shared.rag.unified

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UnifiedSupportTest {
    @Test
    fun `collectUnsupported returns all unsupported flags`() {
        val caps =
            RagCapabilities(
                supportedModes = setOf(UnifiedMode.HYBRID),
                supportsStreaming = false,
                supportsGraphPaths = false,
                supportsFollowUpQueries = false,
                supportsReferences = false,
            )

        val query =
            UnifiedQuery(
                text = "q",
                mode = UnifiedMode.NAIVE,
                streaming = true,
                includeGraphPaths = true,
                includeFollowUps = true,
                includeReferences = true,
            )

        val unsupported = collectUnsupported(query, caps)
        assertTrue("mode:NAIVE" in unsupported)
        assertTrue("streaming" in unsupported)
        assertTrue("graphPaths" in unsupported)
        assertTrue("followUps" in unsupported)
        assertTrue("references" in unsupported)
    }

    @Test
    fun `buildMetadata contains rag id and mode`() {
        val metadata =
            buildMetadata(
                ragId = RagId.CAUSAL_RAG,
                modeUsed = "causal",
                unsupported = listOf("streaming"),
                extra = mapOf("topK" to 5),
            )

        assertEquals("CAUSAL_RAG", metadata["ragId"])
        assertEquals("causal", metadata["modeUsed"])
        assertEquals(5, metadata["topK"])
    }
}
