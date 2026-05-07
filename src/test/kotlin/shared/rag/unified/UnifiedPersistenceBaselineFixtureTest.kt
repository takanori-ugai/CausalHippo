package shared.rag.unified

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

class UnifiedPersistenceBaselineFixtureTest {
    private val root: Path = Path.of("src/test/resources/unified_persistence_parity/p0")
    private val json = Json { ignoreUnknownKeys = true }

    @Test
    fun `inventory covers all unified rag backends`() {
        val inventory = readJsonObject(root.resolve("inventory.json"))
        val rows = inventory["rags"] as? JsonArray ?: fail("inventory.json must contain 'rags' array")
        val ragIds =
            rows.map { row ->
                val rowObj = row as? JsonObject ?: fail("inventory row must be a JSON object")
                val ragId = rowObj["ragId"] as? JsonPrimitive ?: fail("inventory row missing ragId")
                ragId.content
            }.toSet()

        assertEquals(
            setOf(
                "GRAPH_RAG",
                "LIGHT_RAG",
                "PATH_RAG",
                "HIPPO_RAG",
                "CAUSAL_RAG",
                "CAUSAL_HIPPO_RAG",
            ),
            ragIds,
        )
    }

    @Test
    fun `locked fixture hashes remain unchanged`() {
        val lock = readJsonObject(root.resolve("fixture_hashes.json"))
        val files = lock["files"] as? JsonObject ?: fail("fixture_hashes.json must contain 'files' object")
        assertTrue(files.isNotEmpty(), "fixture_hashes.json must not be empty")

        files.entries.forEach { entry ->
            val relativePath = entry.key
            val expectedHashElement = entry.value as? JsonPrimitive ?: fail("Hash entry must be a primitive: $relativePath")
            val expectedHash = expectedHashElement.content
            val filePath = root.resolve(relativePath)
            assertTrue(Files.exists(filePath), "Locked fixture file is missing: $relativePath")
            val actualHash = sha256Hex(filePath)
            assertEquals(expectedHash, actualHash, "Fixture hash mismatch for $relativePath")
        }
    }

    @Test
    fun `all inspect fixtures normalize to common graph shape`() {
        val inventory = readJsonObject(root.resolve("inventory.json"))
        val rows = inventory["rags"] as? JsonArray ?: fail("inventory.json must contain 'rags' array")

        rows.forEach { row ->
            val rowObj = row as? JsonObject ?: fail("inventory row must be a JSON object")
            val ragId = (rowObj["ragId"] as? JsonPrimitive)?.content ?: fail("inventory row missing ragId")
            val baseline = rowObj["baseline"] as? JsonObject ?: fail("$ragId missing baseline object")
            val inspectRel =
                (baseline["inspectFixture"] as? JsonPrimitive)?.content ?: fail("$ragId missing baseline.inspectFixture")

            val inspectPath = root.resolve(inspectRel)
            assertTrue(Files.exists(inspectPath), "Inspect fixture missing for $ragId: $inspectRel")

            val payload = readFixtureMap(inspectPath)
            val normalized = normalizeGraphInspection(payload)
            assertTrue(normalized["nodes"] is List<*>, "$ragId normalized nodes must be a list")
            assertTrue(normalized["edges"] is List<*>, "$ragId normalized edges must be a list")
            assertTrue(normalized["metadata"] is Map<*, *>, "$ragId normalized metadata must be a map")
            val metadata = normalized["metadata"] as Map<*, *>
            assertTrue(metadata.containsKey("nodeCount"), "$ragId metadata missing nodeCount")
            assertTrue(metadata.containsKey("edgeCount"), "$ragId metadata missing edgeCount")
        }
    }

    private fun readJsonObject(path: Path): JsonObject {
        val content = Files.readString(path)
        val parsed = json.parseToJsonElement(content)
        return parsed as? JsonObject ?: fail("Expected JSON object in: $path")
    }

    private fun readFixtureMap(path: Path): Map<String, Any?> {
        val content = Files.readString(path)
        val element = json.parseToJsonElement(content)
        val raw = jsonElementToAny(element)
        @Suppress("UNCHECKED_CAST")
        return raw as? Map<String, Any?> ?: fail("Fixture is not a JSON object: $path")
    }

    private fun jsonElementToAny(element: JsonElement): Any? =
        when (element) {
            is JsonObject -> element.entries.associate { (k, v) -> k to jsonElementToAny(v) }
            is JsonArray -> element.map { jsonElementToAny(it) }
            is JsonPrimitive -> jsonPrimitiveToAny(element)
        }

    private fun jsonPrimitiveToAny(primitive: JsonPrimitive): Any? {
        if (primitive.isString) return primitive.content
        val raw = primitive.content
        if (raw == "true") return true
        if (raw == "false") return false
        raw.toLongOrNull()?.let { return it }
        raw.toDoubleOrNull()?.let { return it }
        return raw
    }

    private fun sha256Hex(path: Path): String {
        val digest = MessageDigest.getInstance("SHA-256")
        Files.newInputStream(path).use { input ->
            val buffer = ByteArray(8192)
            while (true) {
                val read = input.read(buffer)
                if (read <= 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }
}
