package lightrag.core

/**
 * Configuration for LightRAG addons and overrides.
 *
 * @property neo4j Optional configuration for Neo4j.
 * @property overrides Optional overrides for LightRAG settings.
 * @property extras Optional extra configuration parameters.
 * @property cosineBetterThreshold Optional threshold for cosine similarity.
 */
data class AddonConfig(
    val neo4j: Neo4jConfig? = null,
    val overrides: LightRagOverrides = LightRagOverrides(),
    val extras: Map<String, Any> = emptyMap(),
    val cosineBetterThreshold: Double? = null,
) {
    /**
     * Converts the [AddonConfig] to a map.
     * @return A map representation of the [AddonConfig].
     */
    fun toMap(): Map<String, Any> {
        val base = mutableMapOf<String, Any>()
        neo4j?.let { base["neo4j"] = it.toMap().filterValues { value -> value != null } }
        base.putAll(overrides.toMap())
        base.putAll(extras)
        cosineBetterThreshold?.let { base["cosine_better_than_threshold"] = it }
        return base
    }
}

/**
 * Overrides for LightRAG settings.
 *
 * @property chunkTokenSize Optional override for the chunk token size.
 * @property chunkOverlapTokenSize Optional override for the chunk overlap token size.
 * @property entityTypes Optional override for the entity types.
 * @property language Optional override for the language.
 * @property cosineBetterThreshold Optional override for the cosine similarity threshold.
 */
data class LightRagOverrides(
    val chunkTokenSize: Int? = null,
    val chunkOverlapTokenSize: Int? = null,
    val entityTypes: List<String>? = null,
    val language: String? = null,
    val cosineBetterThreshold: Double? = null,
) {
    /**
     * Converts the [LightRagOverrides] to a map.
     * @return A map representation of the [LightRagOverrides].
     */
    fun toMap(): Map<String, Any> =
        buildMap {
            chunkTokenSize?.let { put("chunk_token_size", it) }
            chunkOverlapTokenSize?.let { put("chunk_overlap_token_size", it) }
            entityTypes?.let { put("entity_types", it) }
            language?.let { put("language", it) }
            cosineBetterThreshold?.let { put("cosine_better_than_threshold", it) }
        }
}
