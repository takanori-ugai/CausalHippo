package lightrag.kg.json

import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.encodeToJsonElement
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

/**
 * Represents an entry in the key-value store.
 * @property value The value of the entry.
 */
@Serializable
data class KVEntry(
    val value: KVValue,
)

/**
 * Represents the value of a key-value entry.
 * @property data The data of the value.
 */
@Serializable
data class KVValue(
    @Serializable(with = AnyMapSerializer::class)
    val data: Map<String, Any?>,
)

/**
 * Serializes a map of string keys to arbitrary values via JsonElement to keep persistence flexible.
 */
object AnyMapSerializer : kotlinx.serialization.KSerializer<Map<String, Any?>> {
    private val delegate = MapSerializer(String.serializer(), JsonElement.serializer())

    override val descriptor: SerialDescriptor
        get() = delegate.descriptor

    override fun serialize(
        encoder: Encoder,
        value: Map<String, Any?>,
    ) {
        val jsonMap = value.mapValues { (_, v) -> v.toJsonElement() }
        encoder.encodeSerializableValue(delegate, jsonMap)
    }

    override fun deserialize(decoder: Decoder): Map<String, Any?> {
        val jsonMap = decoder.decodeSerializableValue(delegate)
        return jsonMap.mapValues { (_, v) -> v.toAny() }
    }
}

private fun Any?.toJsonElement(): JsonElement =
    when (this) {
        null -> JsonNull
        is JsonElement -> this
        is Boolean -> JsonPrimitive(this)
        is Number -> JsonPrimitive(this)
        is String -> JsonPrimitive(this)
        is List<*> -> JsonArray(this.map { it.toJsonElement() })
        is Map<*, *> -> JsonObject(this.entries.associate { it.key.toString() to it.value.toJsonElement() })
        else -> JsonPrimitive(this.toString())
    }

private fun JsonElement.toAny(): Any? =
    when (this) {
        is JsonNull -> {
            null
        }

        is JsonPrimitive -> {
            if (isString) {
                content
            } else {
                booleanOrNull ?: longOrNull ?: doubleOrNull ?: content
            }
        }

        is JsonArray -> {
            this.map { it.toAny() }
        }

        is JsonObject -> {
            this.mapValues { it.value.toAny() }
        }
    }
