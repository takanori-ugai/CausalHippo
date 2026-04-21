package lightrag.utils

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * A utility object for JSON operations.
 */
object JsonUtils {
    /**
     * Escapes a string for use in JSON.
     * @param input The string to escape.
     * @return The escaped string.
     */
    fun escape(input: String): String =
        input
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
            .replace("\b", "\\b")
            .replace("\u000C", "\\f")

    /**
     * Converts an object to a JSON string.
     * @param obj The object to convert.
     * @return The JSON string.
     */
    inline fun <reified T> convertObjectToJson(obj: T): String = Json.encodeToString(obj)

    /**
     * Converts a JSON string to an object.
     * @param json The JSON string to convert.
     * @return The object.
     */
    inline fun <reified T> convertJsonToObject(json: String): T = Json.decodeFromString<T>(json)
}
