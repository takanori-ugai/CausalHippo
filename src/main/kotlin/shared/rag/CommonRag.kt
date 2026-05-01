package shared.rag

/**
 * Common synchronous/asynchronous contract for RAG implementations.
 *
 * @param QueryOptions options type used to control query behavior.
 * @param QueryResult response type returned by query operations.
 */
interface CommonRag<QueryOptions, QueryResult> {
    /**
     * Synchronously ingest or update data.
     */
    fun upsert(data: String)

    /**
     * Synchronously ingest or update data.
     */
    fun upsert(data: Collection<String>)

    /**
     * Synchronously ingest or update data.
     */
    fun upsert(data: Array<String>) = upsert(data.asList())

    /**
     * Synchronously ingest or update data.
     */
    fun upsert(data: Sequence<String>) = upsert(data.toList())

    /**
     * Asynchronously ingest or update data.
     */
    suspend fun aupsert(data: String)

    /**
     * Asynchronously ingest or update data.
     */
    suspend fun aupsert(data: Collection<String>)

    /**
     * Asynchronously ingest or update data.
     */
    suspend fun aupsert(data: Array<String>) = aupsert(data.asList())

    /**
     * Asynchronously ingest or update data.
     */
    suspend fun aupsert(data: Sequence<String>) = aupsert(data.toList())

    /**
     * Synchronously drop all data for this RAG instance.
     */
    fun drop()

    /**
     * Asynchronously drop all data for this RAG instance.
     */
    suspend fun adrop()

    /**
     * Synchronously save graph data to persistent storage.
     */
    fun saveGraph(path: String)

    /**
     * Asynchronously save graph data to persistent storage.
     */
    suspend fun asaveGraph(path: String)

    /**
     * Synchronously load graph data from persistent storage.
     */
    fun loadGraph(path: String)

    /**
     * Asynchronously load graph data from persistent storage.
     */
    suspend fun aloadGraph(path: String)

    /**
     * Synchronously inspect graph data.
     */
    fun inspectGraph(): Map<String, Any?>

    /**
     * Asynchronously inspect graph data.
     */
    suspend fun ainspectGraph(): Map<String, Any?>

    /**
     * Synchronously execute a query.
     */
    fun query(
        query: String,
        param: QueryOptions,
    ): QueryResult

    /**
     * Asynchronously execute a query.
     */
    suspend fun aquery(
        query: String,
        param: QueryOptions,
    ): QueryResult
}
