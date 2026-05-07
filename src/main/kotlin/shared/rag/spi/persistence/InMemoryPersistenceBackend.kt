package shared.rag.spi.persistence

/**
 * In-memory backend useful for tests and local prototyping.
 */
class InMemoryPersistenceBackend : PersistenceBackendFactory {
    override fun open(config: Map<String, Any?>): PersistenceSession =
        InMemoryPersistenceSession(
            metadata = config.metadataMap(),
        )
}

private class InMemoryPersistenceSession(
    metadata: Map<String, String>,
) : SnapshotPersistenceSession(
        backendId = "in_memory",
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
    )
