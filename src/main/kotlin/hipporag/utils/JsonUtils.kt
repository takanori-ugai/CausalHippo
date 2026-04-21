package hipporag.utils

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonBuilder
import kotlinx.serialization.modules.SerializersModule
import kotlinx.serialization.modules.contextual

/**
 * Serializers module that registers [AnyValueSerializer] for polymorphic values.
 */
val hippoSerializersModule: SerializersModule =
    SerializersModule {
        contextual(Any::class, AnyValueSerializer)
    }

/**
 * Builds a [Json] instance with default HippoRAG serializers.
 *
 * Custom serializers provided in [configure] are merged with HippoRAG defaults.
 */
fun jsonWithDefaults(configure: JsonBuilder.() -> Unit = {}): Json =
    Json {
        configure()
        serializersModule =
            SerializersModule {
                include(hippoSerializersModule)
                include(serializersModule)
            }
    }
