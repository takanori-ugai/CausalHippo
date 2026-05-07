package shared.rag.spi.persistence

/**
 * Key-value persistence namespace abstraction.
 */
interface KvStore {
    fun put(
        key: String,
        value: Any?,
    )

    fun putAll(entries: Map<String, Any?>) {
        entries.forEach { (key, value) -> put(key, value) }
    }

    fun get(key: String): Any?

    fun delete(key: String)

    fun entries(): Map<String, Any?>

    fun snapshot(): KvSnapshot

    fun restore(snapshot: KvSnapshot)
}

data class KvSnapshot(
    val entries: Map<String, Any?> = emptyMap(),
    val metadata: Map<String, Any?> = emptyMap(),
)
