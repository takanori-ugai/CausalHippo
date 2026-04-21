package lightrag.utils

import kotlinx.serialization.KSerializer
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
import kotlinx.serialization.json.intOrNull

/**
 * A pragmatic serializer for values typed as `Any` in API payloads.
 *
 * Kotlinx serialization cannot infer a serializer for `Any`, so we map common
 * JSON-friendly shapes (primitives, lists, and maps) to JsonElement on encode
 * and back on decode. This keeps KnowledgeGraph responses serializable.
 */
object AnyValueSerializer : KSerializer<Any> {
    override val descriptor: SerialDescriptor = JsonElement.serializer().descriptor

    override fun serialize(
        encoder: Encoder,
        value: Any,
    ) {
        val element = value.toJsonElement()
        encoder.encodeSerializableValue(JsonElement.serializer(), element)
    }

    override fun deserialize(decoder: Decoder): Any {
        val element = decoder.decodeSerializableValue(JsonElement.serializer())
        return element.toAny()
    }
}

private fun Any?.toJsonElement(): JsonElement =
    when (this) {
        null -> {
            JsonNull
        }

        is JsonElement -> {
            this
        }

        is Boolean -> {
            JsonPrimitive(this)
        }

        is Number -> {
            JsonPrimitive(this)
        }

        is String -> {
            JsonPrimitive(this)
        }

        is Iterable<*> -> {
            JsonArray(this.map { it.toJsonElement() })
        }

        is Map<*, *> -> {
            JsonObject(
                this.entries.associate { (k, v) ->
                    k.toString() to v.toJsonElement()
                },
            )
        }

        else -> {
            JsonPrimitive(this.toString())
        }
    }

private fun JsonElement.toAny(): Any =
    when (this) {
        is JsonNull -> {
            "null"
        }

        is JsonPrimitive -> {
            booleanOrNull ?: intOrNull ?: doubleOrNull ?: content
        }

        is JsonArray -> {
            this.map { it.toAny() }
        }

        is JsonObject -> {
            this.mapValues { it.value.toAny() }
        }
    }
