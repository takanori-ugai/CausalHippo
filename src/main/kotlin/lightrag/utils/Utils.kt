package lightrag.utils

import java.security.MessageDigest
import java.util.UUID

/**
 * Computes the MD5 hash of a string.
 * @param input The string to hash.
 * @return The MD5 hash of the string.
 */
fun computeMd5(input: String): String {
    val md = MessageDigest.getInstance("MD5")
    val digest = md.digest(input.toByteArray())
    return digest.joinToString("") { "%02x".format(it) }
}

/**
 * Generates a track ID with a given prefix.
 * @param prefix The prefix for the track ID.
 * @return The generated track ID.
 */
fun generateTrackId(prefix: String): String = "$prefix-${UUID.randomUUID()}"

/**
 * Chunks a list into sublists of a given size with a given overlap.
 * @param size The size of the chunks.
 * @param overlap The overlap between chunks.
 * @return A list of sublists.
 */
fun <T> List<T>.chunked(
    size: Int,
    overlap: Int,
): List<List<T>> {
    require(size > 0) { "Size must be greater than 0" }
    require(overlap >= 0) { "Overlap must be non-negative" }
    require(overlap < size) { "Overlap must be less than size" }

    val result = mutableListOf<List<T>>()
    var index = 0
    while (index < this.size) {
        val end = minOf(index + size, this.size)
        result.add(this.subList(index, end))
        index += (size - overlap)
        if (end == this.size) break
    }
    return result
}
