package shared.rag.unified

/**
 * Minimal usage sample for the unified API.
 */
fun main() {
    val question = "How does Alpha relate to Gamma?"
    val openAiApiKey =
        requireNotNull(System.getenv("OPENAI_API_KEY")) {
            "OPENAI_API_KEY is required to run this sample with OpenAI embeddings."
        }
    val handle =
        UnifiedRagFactory.create(
            ragId = RagId.CAUSAL_RAG,
            overrides =
                mapOf(
                    "useUnifiedPersistence" to true,
                    "persistenceBackend" to "filesystem_snapshot",
                    "useUnifiedSpiForRetrievalAndIndex" to true,
                    "embeddingApiKey" to openAiApiKey,
                    "embeddingModel" to "text-embedding-3-small",
                ),
        )

    try {
        val rag = handle.rag
        val passages =
            listOf(
                "Alpha influences Beta.",
                "Beta affects Gamma.",
            )
        rag.upsert(passages)

        val response =
            rag.query(
                question,
                UnifiedQuery(
                    mode = UnifiedMode.HYBRID,
                    topK = 5,
                    includeAnswer = true,
                    includeContext = true,
                ),
            )

        println("RAG: ${handle.id}")
        println("Persistence backend: ${handle.persistence?.backendId ?: "none"}")
        println("Answer: ${response.answer ?: "(no answer)"}")
        println("Context rows: ${response.context.size}")
    } finally {
        handle.close()
    }
}
