package hipporag.utils

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.InternalSerializationApi
import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.SerialKind
import kotlinx.serialization.descriptors.buildSerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonEncoder
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

/**
 * JSON serializer for arbitrary Kotlin values.
 */
@OptIn(InternalSerializationApi::class, ExperimentalSerializationApi::class)
object AnyValueSerializer : KSerializer<Any> {
    /** Descriptor for arbitrary JSON values. */
    override val descriptor: SerialDescriptor =
        buildSerialDescriptor(
            "AnyValue",
            SerialKind.CONTEXTUAL,
        )

    override fun serialize(
        encoder: Encoder,
        value: Any,
    ) {
        val jsonEncoder = encoder as? JsonEncoder ?: error("AnyValueSerializer can be used only with Json")
        jsonEncoder.encodeJsonElement(toJsonElement(value))
    }

    override fun deserialize(decoder: Decoder): Any {
        val jsonDecoder = decoder as? JsonDecoder ?: error("AnyValueSerializer can be used only with Json")
        return fromJsonElement(jsonDecoder.decodeJsonElement()) ?: JsonNull
    }

    private fun toJsonElement(value: Any?): JsonElement =
        when (value) {
            null -> {
                JsonNull
            }

            is JsonElement -> {
                value
            }

            is String -> {
                JsonPrimitive(value)
            }

            is Boolean -> {
                JsonPrimitive(value)
            }

            is Int -> {
                JsonPrimitive(value)
            }

            is Long -> {
                JsonPrimitive(value)
            }

            is Float -> {
                JsonPrimitive(value)
            }

            is Double -> {
                JsonPrimitive(value)
            }

            is Number -> {
                JsonPrimitive(value.toString())
            }

            is Map<*, *> -> {
                JsonObject(
                    value.entries
                        .mapNotNull { (key, entryValue) -> key?.toString()?.let { it to toJsonElement(entryValue) } }
                        .toMap(),
                )
            }

            is Iterable<*> -> {
                JsonArray(value.map { toJsonElement(it) })
            }

            is Array<*> -> {
                JsonArray(value.map { toJsonElement(it) })
            }

            else -> {
                JsonPrimitive(value.toString())
            }
        }

    private fun fromJsonElement(element: JsonElement): Any? =
        when (element) {
            is JsonNull -> {
                null
            }

            is JsonPrimitive -> {
                if (element.isString) {
                    element.content
                } else {
                    element.booleanOrNull
                        ?: element.intOrNull
                        ?: element.longOrNull
                        ?: element.doubleOrNull
                        ?: element.content
                }
            }

            is JsonArray -> {
                element.map { fromJsonElement(it) }
            }

            is JsonObject -> {
                element.mapValues { (_, value) -> fromJsonElement(value) }
            }
        }
}
