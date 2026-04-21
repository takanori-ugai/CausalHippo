package hipporag.utils

import kotlinx.serialization.Contextual
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Raw NER extraction output for a chunk.
 *
 * @property chunkId chunk identifier.
 * @property response raw model response.
 * @property uniqueEntities extracted entity strings.
 * @property metadata model metadata.
 */
@Serializable
data class NerRawOutput(
    val chunkId: String,
    val response: String?,
    val uniqueEntities: List<String>,
    val metadata: Map<String, @Contextual Any?>,
)

/**
 * Raw triple extraction output for a chunk.
 *
 * @property chunkId chunk identifier.
 * @property response raw model response.
 * @property triples extracted triples.
 * @property metadata model metadata.
 */
@Serializable
data class TripleRawOutput(
    val chunkId: String,
    val response: String?,
    val triples: List<List<String>>,
    val metadata: Map<String, @Contextual Any?>,
)

/**
 * Retrieval result for a single query.
 *
 * @property question query string.
 * @property docs retrieved documents/passages.
 * @property docScores optional scores aligned to [docs].
 * @property answer optional predicted answer.
 * @property goldAnswers optional gold answers.
 * @property goldDocs optional gold documents.
 */
@Serializable
data class QuerySolution(
    val question: String,
    val docs: List<String>,
    var docScores: DoubleArray? = null,
    var answer: String? = null,
    var goldAnswers: MutableList<String>? = null,
    var goldDocs: MutableList<String>? = null,
)

/**
 * Output for a linking step in retrieval.
 *
 * @property score node scores.
 * @property type linking strategy type.
 */
@Serializable
data class LinkingOutput(
    val score: DoubleArray,
    val type: LinkingOutputType,
)

/** Supported linking output types. */
@Serializable
enum class LinkingOutputType {
    NODE,
    DPR,
}

/**
 * Stored embedding row.
 *
 * @property hashId hashed identifier.
 * @property content original text content.
 * @property name optional name override.
 */
@Serializable
data class EmbeddingRow(
    val hashId: String,
    val content: String,
    val name: String? = null,
) {
    /** Converts this row into a graph-attribute map. */
    fun toAttributes(): Map<String, Any> =
        mapOf(
            "hash_id" to hashId,
            "content" to content,
            "name" to (name ?: hashId),
        )
}

/**
 * Chat message for LLM interactions.
 *
 * @property role message role.
 * @property content message content.
 */
@Serializable
data class Message(
    val role: String,
    val content: String,
)

/**
 * LLM inference result payload.
 *
 * @property response model response text.
 * @property metadata model metadata.
 * @property cacheHit whether the response came from cache.
 */
@Serializable
data class LlmResult(
    val response: String,
    val metadata: Map<String, @Contextual Any?>,
    val cacheHit: Boolean = false,
)

/**
 * Aggregated RAG QA result.
 *
 * @property solutions per-query solutions with answers.
 * @property responseMessages raw LLM responses.
 * @property metadata per-response metadata.
 * @property overallRetrievalResult optional retrieval metrics.
 * @property overallQaResults optional QA metrics.
 */
@Serializable
data class RagQaResult(
    val solutions: List<QuerySolution>,
    val responseMessages: List<String>,
    val metadata: List<Map<String, @Contextual Any?>>,
    val overallRetrievalResult: Map<String, Double>?,
    val overallQaResults: Map<String, Double>?,
)

/**
 * Serialized OpenIE result bundle.
 *
 * @property docs extracted documents.
 * @property avgEntChars average entity length (chars).
 * @property avgEntWords average entity length (words).
 */
@Serializable
data class OpenieResults(
    val docs: List<OpenieDoc>,
    @SerialName("avg_ent_chars") val avgEntChars: Double,
    @SerialName("avg_ent_words") val avgEntWords: Double,
)

/**
 * Serialized OpenIE document record.
 *
 * @property idx chunk identifier.
 * @property passage original passage text.
 * @property extractedEntities extracted entity strings.
 * @property extractedTriples extracted triples.
 */
@Serializable
data class OpenieDoc(
    val idx: String,
    val passage: String,
    @SerialName("extracted_entities") val extractedEntities: List<String>,
    @SerialName("extracted_triples") val extractedTriples: List<List<String>>,
)
