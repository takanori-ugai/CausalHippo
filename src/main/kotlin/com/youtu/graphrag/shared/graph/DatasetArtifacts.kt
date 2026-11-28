package com.youtu.graphrag.shared.graph

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption.ATOMIC_MOVE
import java.nio.file.StandardCopyOption.REPLACE_EXISTING

/** Coordinates publication and reading of a dataset's two artifacts within the JVM. */
internal object DatasetArtifacts {
    private val locks = Array(64) { Any() }

    fun <T> read(
        graph: Path,
        action: () -> T,
    ): T = synchronized(locks[(graph.toAbsolutePath().normalize().hashCode() and Int.MAX_VALUE) % locks.size], action)

    fun publish(
        graph: Path,
        graphBytes: ByteArray,
        chunks: Path,
        chunkBytes: ByteArray,
    ) {
        val absoluteGraph = graph.toAbsolutePath()
        val absoluteChunks = chunks.toAbsolutePath()
        Files.createDirectories(absoluteGraph.parent)
        Files.createDirectories(absoluteChunks.parent)
        val stagedGraph = Files.createTempFile(absoluteGraph.parent, ".graph-", ".tmp")
        try {
            val stagedChunks = Files.createTempFile(absoluteChunks.parent, ".chunks-", ".tmp")
            try {
                Files.write(stagedGraph, graphBytes)
                Files.write(stagedChunks, chunkBytes)
                read(absoluteGraph) {
                    val backup =
                        if (Files.exists(absoluteChunks)) {
                            Files.createTempFile(absoluteChunks.parent, ".previous-chunks-", ".tmp")
                        } else {
                            null
                        }
                    var recovered = true
                    try {
                        if (backup != null) Files.copy(absoluteChunks, backup, REPLACE_EXISTING)
                        Files.move(stagedChunks, absoluteChunks, ATOMIC_MOVE, REPLACE_EXISTING)
                        try {
                            Files.move(stagedGraph, absoluteGraph, ATOMIC_MOVE, REPLACE_EXISTING)
                        } catch (error: Exception) {
                            runCatching {
                                if (backup == null) {
                                    Files.deleteIfExists(absoluteChunks)
                                } else {
                                    Files.move(backup, absoluteChunks, ATOMIC_MOVE, REPLACE_EXISTING)
                                }
                            }.exceptionOrNull()?.let {
                                recovered = false
                                error.addSuppressed(IllegalStateException("Previous chunks could not be restored from $backup", it))
                            }
                            throw error
                        }
                    } finally {
                        if (backup != null && recovered) Files.deleteIfExists(backup)
                    }
                }
            } finally {
                Files.deleteIfExists(stagedChunks)
            }
        } finally {
            Files.deleteIfExists(stagedGraph)
        }
    }
}
