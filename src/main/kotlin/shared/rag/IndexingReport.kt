package shared.rag

/** Recoverable extraction failures from the most recent indexing operation. */
data class IndexingReport(
    val failures: List<IndexingFailure> = emptyList(),
) {
    val failedChunkCount: Int
        get() = failures.distinctBy { it.stage to it.chunk }.size
}

data class IndexingFailure(
    val stage: String,
    val chunk: String,
    val reason: String,
)
