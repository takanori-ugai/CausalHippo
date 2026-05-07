package shared.rag.spi.persistence

import java.nio.file.Path

/**
 * Filesystem-backed reference backend for unified persistence checkpoints.
 */
class FilesystemSnapshotBackend : PersistenceBackendFactory {
    override fun open(config: Map<String, Any?>): PersistenceSession {
        val rootDir =
            config["rootDir"]
                ?.toString()
                ?.trim()
                ?.takeIf { it.isNotEmpty() }
                ?.let { Path.of(it).toAbsolutePath().normalize() }

        return FilesystemSnapshotSession(
            rootDir = rootDir,
            metadata = config.metadataMap(),
        )
    }
}

private class FilesystemSnapshotSession(
    private val rootDir: Path?,
    metadata: Map<String, String>,
) : SnapshotPersistenceSession(
        backendId = "filesystem_snapshot",
        capabilities =
            PersistenceCapabilities(
                supportsGraphPersistence = true,
                supportsVectorPersistence = true,
                supportsKvPersistence = true,
                supportsAtomicCheckpoint = true,
                supportsIncrementalCheckpoint = false,
                supportsCrossBackendImport = true,
            ),
        manifestMetadata = metadata,
    ) {
    override fun resolvePath(path: String): Path {
        val raw = Path.of(path)
        return when {
            raw.isAbsolute -> raw.normalize()
            rootDir != null -> rootDir.resolve(raw).toAbsolutePath().normalize()
            else -> raw.toAbsolutePath().normalize()
        }
    }
}

internal fun Map<String, Any?>.metadataMap(): Map<String, String> {
    val metadata = this["metadata"] as? Map<*, *> ?: return emptyMap()
    return metadata.entries.associate { entry ->
        entry.key.toString() to (entry.value?.toString() ?: "")
    }
}
