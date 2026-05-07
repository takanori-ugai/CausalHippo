package shared.rag.spi.persistence

/**
 * Artifact/file persistence namespace abstraction.
 */
interface ArtifactStore {
    fun writeBytes(
        path: String,
        bytes: ByteArray,
    )

    fun writeText(
        path: String,
        text: String,
    ) = writeBytes(path, text.toByteArray(Charsets.UTF_8))

    fun readBytes(path: String): ByteArray?

    fun readText(path: String): String? = readBytes(path)?.toString(Charsets.UTF_8)

    fun listPaths(): List<String>

    fun delete(path: String)

    fun snapshot(): ArtifactSnapshot

    fun restore(snapshot: ArtifactSnapshot)
}

data class ArtifactSnapshot(
    val filesBase64: Map<String, String> = emptyMap(),
    val metadata: Map<String, Any?> = emptyMap(),
)
