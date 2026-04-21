package lightrag.core

/**
 * Configuration for Neo4j.
 *
 * @property uri The URI of the Neo4j instance.
 * @property username The username for authentication.
 * @property password The password for authentication.
 * @property database The database to use.
 */
data class Neo4jConfig(
    val uri: String? = null,
    val username: String? = null,
    val password: String? = null,
    val database: String? = null,
) {
    /**
     * Converts the [Neo4jConfig] to a map.
     * @return A map representation of the [Neo4jConfig].
     */
    fun toMap(): Map<String, Any?> =
        mapOf(
            "uri" to uri,
            "username" to username,
            "password" to password,
            "database" to database,
        )
}
