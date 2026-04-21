package ragas

import ragas.backends.BackendInfo
import ragas.backends.BackendRegistry
import ragas.backends.InMemoryBackend
import ragas.backends.LocalCsvBackend
import ragas.backends.LocalJsonlBackend
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class BackendsTest {
    @Test
    fun inMemoryBackendStoresDatasetsAndExperiments() {
        val backend = InMemoryBackend()
        val datasetRows = listOf(mapOf("id" to 1, "text" to "hello"))
        val experimentRows = listOf(mapOf("metric" to "faithfulness", "score" to 0.8))

        backend.saveDataset("d1", datasetRows)
        backend.saveExperiment("e1", experimentRows)

        assertEquals(datasetRows, backend.loadDataset("d1"))
        assertEquals(experimentRows, backend.loadExperiment("e1"))
        assertEquals(listOf("d1"), backend.listDatasets())
        assertEquals(listOf("e1"), backend.listExperiments())
    }

    @Test
    fun localCsvBackendRoundTripsSimpleRows() {
        val root = createTempDirectory("ragas-csv-test").toFile()
        val backend = LocalCsvBackend(root.absolutePath)

        val rows =
            listOf(
                mapOf("id" to 1, "text" to "hello,world", "flag" to true),
                mapOf("id" to 2, "text" to "plain", "flag" to false),
            )
        backend.saveDataset("dataset1", rows)

        val loaded = backend.loadDataset("dataset1")
        assertEquals(rows.size, loaded.size)
        assertEquals("1", loaded[0]["id"])
        assertEquals("hello,world", loaded[0]["text"])
        assertEquals("true", loaded[0]["flag"])
        assertEquals("2", loaded[1]["id"])
        assertEquals("plain", loaded[1]["text"])
        assertEquals("false", loaded[1]["flag"])
        assertEquals(listOf("dataset1"), backend.listDatasets())
    }

    @Test
    fun localCsvBackendRoundTripsMultilineQuotedFields() {
        val root = createTempDirectory("ragas-csv-test").toFile()
        val backend = LocalCsvBackend(root.absolutePath)

        val rows =
            listOf(
                mapOf("id" to 1, "text" to "first line\nsecond line"),
                mapOf("id" to 2, "text" to "plain"),
            )
        backend.saveDataset("dataset_multiline", rows)

        val loaded = backend.loadDataset("dataset_multiline")
        assertEquals("first line\nsecond line", loaded[0]["text"])
        assertEquals("plain", loaded[1]["text"])
    }

    @Test
    fun localCsvBackendRejectsTraversalStyleNames() {
        val root = createTempDirectory("ragas-csv-test").toFile()
        val backend = LocalCsvBackend(root.absolutePath)

        val error =
            assertFailsWith<IllegalArgumentException> {
                backend.saveDataset("../evil", listOf(mapOf("id" to 1)))
            }
        assertTrue(error.message.orEmpty().contains("Invalid name '../evil'"))
    }

    @Test
    fun localJsonlBackendRoundTripsNestedRows() {
        val root = createTempDirectory("ragas-jsonl-test").toFile()
        val backend = LocalJsonlBackend(root.absolutePath)

        val rows =
            listOf(
                mapOf(
                    "id" to 1,
                    "name" to "alice",
                    "tags" to listOf("a", "b"),
                    "meta" to mapOf("x" to 1, "ok" to true),
                ),
            )
        backend.saveExperiment("exp1", rows)

        val loaded = backend.loadExperiment("exp1")
        assertEquals(1L, loaded[0]["id"])
        assertEquals("alice", loaded[0]["name"])
        assertEquals(listOf("a", "b"), loaded[0]["tags"])
        @Suppress("UNCHECKED_CAST")
        val meta = loaded[0]["meta"] as Map<String, Any?>
        assertEquals(1L, meta["x"])
        assertEquals(true, meta["ok"])
        assertEquals(listOf("exp1"), backend.listExperiments())
    }

    @Test
    fun localJsonlBackendRejectsUnsupportedValueTypes() {
        val root = createTempDirectory("ragas-jsonl-test").toFile()
        val backend = LocalJsonlBackend(root.absolutePath)

        val rows = listOf(mapOf("id" to 1, "unsupported" to Any()))
        assertFailsWith<IllegalArgumentException> {
            backend.saveExperiment("exp-bad", rows)
        }
    }

    @Test
    fun localJsonlBackendRejectsTraversalStyleNames() {
        val root = createTempDirectory("ragas-jsonl-test").toFile()
        val backend = LocalJsonlBackend(root.absolutePath)

        val error =
            assertFailsWith<IllegalArgumentException> {
                backend.saveExperiment("../evil", listOf(mapOf("id" to 1)))
            }
        assertTrue(error.message.orEmpty().contains("Invalid name '../evil'"))
    }

    @Test
    fun backendRegistrySupportsAliases() {
        val registry = BackendRegistry()
        registry.register("custom", ::InMemoryBackend, aliasList = listOf("c"))

        assertTrue(registry.contains("custom"))
        assertTrue(registry.contains("c"))
        val backend = registry.create("c")
        assertTrue(backend is InMemoryBackend)
    }

    @Test
    fun backendRegistryRejectsAliasCollisionsAcrossDifferentBackends() {
        val registry = BackendRegistry()
        registry.register("first", ::InMemoryBackend, aliasList = listOf("shared"))

        val error =
            assertFailsWith<IllegalArgumentException> {
                registry.register("second", ::InMemoryBackend, aliasList = listOf("shared"))
            }
        assertTrue(error.message.orEmpty().contains("Alias 'shared' is already registered"))
    }

    @Test
    fun backendRegistryReRegisterReplacesAliasSetForSameBackend() {
        val registry = BackendRegistry()
        registry.register("custom", ::InMemoryBackend, aliasList = listOf("old"))
        registry.register("custom", ::InMemoryBackend, aliasList = listOf("new"))

        assertFalse(registry.contains("old"))
        assertTrue(registry.contains("new"))
        assertTrue(registry.contains("custom"))
    }

    @Test
    fun backendRegistryRejectsAliasThatShadowsCanonicalBackendName() {
        val registry = BackendRegistry()
        registry.register("csv", ::InMemoryBackend)

        val error =
            assertFailsWith<IllegalArgumentException> {
                registry.register("custom", ::InMemoryBackend, aliasList = listOf("csv"))
            }
        assertTrue(error.message.orEmpty().contains("conflicts with canonical backend name"))
    }

    @Test
    fun backendRegistryDiscoversServiceLoaderProviders() {
        val registry = BackendRegistry()
        val loadedProviders = registry.discoverBackends()

        assertEquals(1, loadedProviders)
        assertTrue(registry.contains("test/discovered"))
        assertTrue(registry.contains("td"))
        assertFalse(registry.contains("test/partial"))
        assertFalse(registry.contains("tp"))
        val backend = registry.create("td")
        assertTrue(backend is InMemoryBackend)
    }

    @Test
    fun backendRegistryRollsBackFailedProviderRegistrationAtomically() {
        val registry = BackendRegistry()

        assertEquals(1, registry.discoverBackends())
        assertFalse(registry.contains("test/partial"))
        assertFalse(registry.contains("tp"))

        assertEquals(1, registry.discoverBackends(force = true))
        assertFalse(registry.contains("test/partial"))
        assertFalse(registry.contains("tp"))
    }

    @Test
    fun backendRegistryProvidesInspectionMetadata() {
        val registry = BackendRegistry()
        registry.register(
            name = "custom/info",
            factory = ::InMemoryBackend,
            aliasList = listOf("ci"),
            backendClass = InMemoryBackend::class,
            description = "custom backend for info test",
            source = "test",
        )

        val info: BackendInfo = registry.getBackendInfo("ci")

        assertEquals("custom/info", info.name)
        assertEquals(listOf("ci"), info.aliases)
        assertTrue(info.implementationClass.contains("InMemoryBackend"))
        assertEquals("test", info.source)
        assertEquals("custom backend for info test", info.description)
    }

    @Test
    fun backendRegistryInspectionDoesNotInstantiateFactoryWithoutBackendClass() {
        val registry = BackendRegistry()
        var createCalls = 0
        registry.register(
            name = "custom/lazy-info",
            factory = {
                createCalls += 1
                InMemoryBackend()
            },
            aliasList = listOf("cli"),
            description = "custom backend for lazy info test",
            source = "test",
        )

        val info: BackendInfo = registry.getBackendInfo("cli")

        assertEquals("unknown", info.implementationClass)
        assertEquals(0, createCalls)
    }

    @Test
    fun backendRegistryListAllNamesIncludesAliases() {
        val registry = BackendRegistry()
        registry.register(
            name = "custom/list",
            factory = ::InMemoryBackend,
            aliasList = listOf("cl1", "cl2"),
        )

        val listing = registry.listAllNames()

        assertEquals(listOf("custom/list", "cl1", "cl2"), listing["custom/list"])
    }

    @Test
    fun backendRegistryClearWithResetReappliesBootstrapRegistrations() {
        val registry =
            BackendRegistry {
                register("builtin/test", ::InMemoryBackend, aliasList = listOf("bt"), source = "builtin")
            }

        assertTrue(registry.contains("builtin/test"))
        assertTrue(registry.contains("bt"))

        registry.clear(resetDiscovery = true)

        assertTrue(registry.contains("builtin/test"))
        assertTrue(registry.contains("bt"))
    }

    @Test
    fun missingDatasetRaisesFileNotFound() {
        val backend = InMemoryBackend()
        assertFailsWith<java.io.FileNotFoundException> {
            backend.loadDataset("missing")
        }
    }
}
