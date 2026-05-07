package shared.rag.spi.persistence

/**
 * Capability flags exposed by a persistence backend.
 */
data class PersistenceCapabilities(
    val supportsGraphPersistence: Boolean,
    val supportsVectorPersistence: Boolean,
    val supportsKvPersistence: Boolean,
    val supportsAtomicCheckpoint: Boolean,
    val supportsIncrementalCheckpoint: Boolean,
    val supportsCrossBackendImport: Boolean,
)
