package shared.rag

/**
 * Common synchronous/asynchronous contract for RAG implementations.
 *
 * @param QueryOptions options type used to control query behavior.
 * @param QueryResult response type returned by query operations.
 */
interface CommonRag<QueryOptions, QueryResult> {
    /**
     * Synchronously ingests or updates a single text payload.
     *
     * @param data text payload to index.
     */
    fun upsert(data: String)

    /**
     * Synchronously ingests or updates multiple text payloads.
     *
     * @param data text payloads to index.
     */
    fun upsert(data: Collection<String>)

    /**
     * Synchronously ingests or updates multiple text payloads from an array.
     *
     * @param data text payloads to index.
     */
    fun upsert(data: Array<String>) = upsert(data.asList())

    /**
     * Synchronously ingests or updates multiple text payloads from a sequence.
     *
     * @param data text payloads to index.
     */
    fun upsert(data: Sequence<String>) = upsert(data.toList())

    /**
     * Asynchronously ingests or updates a single text payload.
     *
     * @param data text payload to index.
     */
    suspend fun aupsert(data: String)

    /**
     * Asynchronously ingests or updates multiple text payloads.
     *
     * @param data text payloads to index.
     */
    suspend fun aupsert(data: Collection<String>)

    /**
     * Asynchronously ingests or updates multiple text payloads from an array.
     *
     * @param data text payloads to index.
     */
    suspend fun aupsert(data: Array<String>) = aupsert(data.asList())

    /**
     * Asynchronously ingests or updates multiple text payloads from a sequence.
     *
     * @param data text payloads to index.
     */
    suspend fun aupsert(data: Sequence<String>) = aupsert(data.toList())

    /**
     * Synchronously removes all indexed data for this RAG instance.
     */
    fun drop()

    /**
     * Asynchronously removes all indexed data for this RAG instance.
     */
    suspend fun adrop()

    /**
     * Synchronously saves graph state to persistent storage.
     *
     * @param path output location for serialized graph data.
     */
    fun saveGraph(path: String)

    /**
     * Asynchronously saves graph state to persistent storage.
     *
     * @param path output location for serialized graph data.
     */
    suspend fun asaveGraph(path: String)

    /**
     * Synchronously loads graph state from persistent storage.
     *
     * @param path input location for serialized graph data.
     */
    fun loadGraph(path: String)

    /**
     * Asynchronously loads graph state from persistent storage.
     *
     * @param path input location for serialized graph data.
     */
    suspend fun aloadGraph(path: String)

    /**
     * Synchronously inspects graph state for diagnostics.
     *
     * @return diagnostic graph snapshot.
     */
    fun inspectGraph(): Map<String, Any?>

    /**
     * Asynchronously inspects graph state for diagnostics.
     *
     * @return diagnostic graph snapshot.
     */
    suspend fun ainspectGraph(): Map<String, Any?>

    /**
     * Synchronously executes a retrieval query.
     *
     * @param query query text.
     * @param param query-time options.
     * @return query result payload.
     */
    fun query(
        query: String,
        param: QueryOptions,
    ): QueryResult

    /**
     * Asynchronously executes a retrieval query.
     *
     * @param query query text.
     * @param param query-time options.
     * @return query result payload.
     */
    suspend fun aquery(
        query: String,
        param: QueryOptions,
    ): QueryResult
}
