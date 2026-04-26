package shared.rag

/**
 * Common vector storage contract shared across RAG implementations.
 *
 * @param ValueType value type stored in each metadata map.
 */
interface CommonVectorStorage<ValueType> {
    /**
     * Searches vectors by textual query and returns top-K matches.
     */
    suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, ValueType>>

    /**
     * Inserts or updates vector rows.
     */
    suspend fun upsert(data: Map<String, Map<String, ValueType>>)

    /**
     * Deletes vectors for an entity.
     */
    suspend fun deleteEntity(entityName: String)

    /**
     * Deletes relation vectors associated with an entity.
     */
    suspend fun deleteEntityRelation(entityName: String)
}
