package shared.rag.spi.persistence

/**
 * Vector index persistence namespace abstraction.
 */
interface VectorIndexStore {
    fun upsert(record: VectorRecord)

    fun upsert(records: Collection<VectorRecord>) {
        records.forEach(::upsert)
    }

    fun delete(id: String)

    fun query(
        vector: List<Double>,
        topK: Int = 5,
    ): List<VectorRecord>

    fun snapshot(): VectorSnapshot

    fun restore(snapshot: VectorSnapshot)
}

data class VectorRecord(
    val id: String,
    val vector: List<Double>,
    val metadata: Map<String, Any?> = emptyMap(),
)

data class VectorSnapshot(
    val dimensions: Int? = null,
    val metric: String = "cosine",
    val records: List<VectorRecord> = emptyList(),
    val metadata: Map<String, Any?> = emptyMap(),
)
