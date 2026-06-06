package com.youtu.graphrag.shared.retriever

import org.jetbrains.bio.npy.NpzFile
import java.nio.ByteOrder
import java.nio.file.Path
import java.util.Base64
import kotlin.io.path.createDirectories
import kotlin.io.path.exists

/**
 * NPZ codec for float32 1D embedding vectors, backed by Multik's JVM NPY/NPZ stack.
 */
internal object NpzEmbeddingCache {
    private const val ENCODED_KEY_PREFIX = "b64_"

    fun read(
        path: Path,
        expectedDimensions: Int,
    ): Map<String, FloatArray> {
        if (!path.exists()) {
            return emptyMap()
        }

        val vectors = linkedMapOf<String, FloatArray>()
        NpzFile.read(path).use { reader ->
            reader.introspect().forEach { entry ->
                val key = decodeKey(entry.name) ?: return@forEach
                if (key.isBlank()) {
                    return@forEach
                }
                if (entry.type != Float::class.java && entry.type != Float::class.javaObjectType) {
                    return@forEach
                }

                val npyArray = reader[entry.name]
                if (npyArray.shape.size != 1 || npyArray.shape[0] != expectedDimensions) {
                    return@forEach
                }
                vectors[key] = npyArray.asFloatArray()
            }
        }
        return vectors
    }

    fun write(
        path: Path,
        vectorsByKey: Map<String, FloatArray>,
        expectedDimensions: Int,
    ) {
        if (vectorsByKey.isEmpty()) {
            return
        }

        path.parent?.createDirectories()
        NpzFile.write(path, compressed = true).use { writer ->
            vectorsByKey.forEach { (key, vector) ->
                if (key.isBlank() || vector.size != expectedDimensions) {
                    return@forEach
                }
                val encodedName = encodeKey(key)
                writer.write(
                    name = encodedName,
                    data = vector,
                    shape = intArrayOf(vector.size),
                    order = ByteOrder.LITTLE_ENDIAN,
                )
            }
        }
    }

    private fun encodeKey(key: String): String {
        val encoded = Base64.getUrlEncoder().withoutPadding().encodeToString(key.toByteArray(Charsets.UTF_8))
        return ENCODED_KEY_PREFIX + encoded
    }

    private fun decodeKey(entryName: String): String? {
        if (!entryName.startsWith(ENCODED_KEY_PREFIX)) {
            // Backward compatibility with existing NPZ caches that store raw keys.
            return entryName
        }
        val encoded = entryName.removePrefix(ENCODED_KEY_PREFIX)
        return runCatching {
            val bytes = Base64.getUrlDecoder().decode(encoded)
            bytes.toString(Charsets.UTF_8)
        }.getOrNull()
    }
}
