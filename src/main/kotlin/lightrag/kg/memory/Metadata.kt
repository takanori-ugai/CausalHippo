package lightrag.kg.memory

/**
 * Represents metadata for a vector.
 * @property content The content of the vector.
 * @property vector The vector itself.
 * @property entityName The name of the entity.
 * @property srcId The ID of the source node.
 * @property tgtId The ID of the target node.
 * @property raw The raw metadata.
 */
data class Metadata(
    val content: String? = null,
    val vector: List<Float>? = null,
    val entityName: String? = null,
    val srcId: String? = null,
    val tgtId: String? = null,
    val raw: Map<String, Any> = emptyMap(),
)
