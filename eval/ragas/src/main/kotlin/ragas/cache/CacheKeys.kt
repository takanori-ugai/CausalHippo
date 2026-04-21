package ragas.cache

import java.security.MessageDigest

/**
 * Returns a deterministic hash key suitable for cache indexing.
 *
 * @param raw Raw string input used to derive a stable key.
 */
fun stableCacheKey(raw: String): String {
    val digest = MessageDigest.getInstance("SHA-256")
    val bytes = digest.digest(raw.toByteArray(Charsets.UTF_8))
    return bytes.joinToString("") { b -> "%02x".format(b) }
}
