package shared.rag

/**
 * Common vector storage contract shared across RAG implementations.
 *
 * @param ValueType value type stored in the returned and persisted metadata maps.
 */
interface CommonVectorStorage<ValueType> {
    /**
     * Searches vectors by textual [query] and returns up to [topK] nearest matches.
     *
     * @param query text query used to perform similarity search.
     * @param topK maximum number of result rows to return.
     * @return ordered vector rows represented as metadata maps.
     */
    suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, ValueType>>

    /**
     * Inserts or updates vector rows.
     *
     * @param data row ID to metadata map payload.
     */
    suspend fun upsert(data: Map<String, Map<String, ValueType>>)

    /**
     * Deletes vectors for an entity.
     *
     * @param entityName entity identifier used by the storage backend.
     */
    suspend fun deleteEntity(entityName: String)

    /**
     * Deletes relation vectors associated with an entity.
     *
     * @param entityName entity identifier whose relation vectors should be removed.
     */
    suspend fun deleteEntityRelation(entityName: String)
}
