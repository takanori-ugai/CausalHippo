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
    ): T =
        synchronized(lockFor(graph)) {
            check(!Files.exists(recoveryMarker(graph))) { "Dataset artifacts require a successful rebuild: $graph" }
            action()
        }

    fun readPair(
        graph: Path,
        chunks: Path,
    ): Pair<ByteArray, ByteArray>? =
        read(graph) {
            if (!Files.exists(graph)) return@read null
            Files.readAllBytes(graph) to if (Files.exists(chunks)) Files.readAllBytes(chunks) else byteArrayOf()
        }

    private fun lockFor(graph: Path): Any = locks[(graph.toAbsolutePath().normalize().hashCode() and Int.MAX_VALUE) % locks.size]

    private fun recoveryMarker(graph: Path): Path = graph.resolveSibling(".${graph.fileName}.recovery-required")

    private fun replace(
        source: Path,
        target: Path,
    ) {
        Files.move(source, target, ATOMIC_MOVE, REPLACE_EXISTING)
    }

    fun publish(
        graph: Path,
        graphBytes: ByteArray,
        chunks: Path,
        chunkBytes: ByteArray,
        move: (Path, Path) -> Unit = ::replace,
        shouldPublish: () -> Boolean = { true },
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
                synchronized(lockFor(absoluteGraph)) {
                    if (!shouldPublish()) return@synchronized
                    val marker = recoveryMarker(absoluteGraph)
                    val previouslyUnhealthy = Files.exists(marker)
                    val backup =
                        if (Files.exists(absoluteChunks)) {
                            Files.createTempFile(absoluteChunks.parent, ".previous-chunks-", ".tmp")
                        } else {
                            null
                        }
                    var recovered = true
                    var published = false
                    try {
                        if (backup != null) Files.copy(absoluteChunks, backup, REPLACE_EXISTING)
                        if (!previouslyUnhealthy) Files.createFile(marker)
                        move(stagedChunks, absoluteChunks)
                        try {
                            move(stagedGraph, absoluteGraph)
                            published = true
                        } catch (error: Exception) {
                            runCatching {
                                if (backup == null) {
                                    Files.deleteIfExists(absoluteChunks)
                                } else {
                                    move(backup, absoluteChunks)
                                }
                            }.exceptionOrNull()?.let {
                                recovered = false
                                error.addSuppressed(IllegalStateException("Previous chunks could not be restored from $backup", it))
                            }
                            throw error
                        }
                    } finally {
                        if (published || (recovered && !previouslyUnhealthy)) Files.deleteIfExists(marker)
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
