package ragas.backends

import java.util.ServiceConfigurationError
import java.util.ServiceLoader
import kotlin.reflect.KClass

/**
 * Metadata describing a registered backend implementation.
 *
 * @property name Canonical backend name.
 * @property aliases Registered aliases that resolve to [name].
 * @property implementationClass Fully qualified implementation class name.
 * @property module Module/package name derived from [implementationClass].
 * @property description Human-readable backend description.
 * @property source Registration source (for example `builtin` or `runtime`).
 */
data class BackendInfo(
    val name: String,
    val aliases: List<String>,
    val implementationClass: String,
    val module: String,
    val description: String,
    val source: String,
)

/**
 * ServiceLoader extension point used to register additional backends at runtime.
 */
interface BackendDiscoveryProvider {
    /**
     * Registers one or more backends into [registry].
     *
     * @param registry Backend registry to update.
     */
    fun registerBackends(registry: BackendRegistry)
}

/** Registry for backend factories, aliases, and backend metadata. */
class BackendRegistry(
    private val bootstrap: (BackendRegistry.() -> Unit)? = null,
) {
    private data class BackendRegistration(
        val factory: () -> BaseBackend,
        val backendClass: KClass<out BaseBackend>?,
        val description: String?,
        val source: String,
    )

    private val backends = mutableMapOf<String, BackendRegistration>()
    private val aliases = mutableMapOf<String, String>()
    private var discovered = false

    init {
        bootstrap?.invoke(this)
    }

    /**
     * Adds or replaces a backend registration.
     *
     * @param name Name or identifier.
     * @param factory Factory function that creates backend instances.
     * @param aliasList Aliases to register for [name].
     * @param backendClass Optional backend implementation class metadata.
     * @param description Optional human-readable backend description.
     * @param source Registration source label.
     */
    @Synchronized
    fun register(
        name: String,
        factory: () -> BaseBackend,
        aliasList: List<String> = emptyList(),
        backendClass: KClass<out BaseBackend>? = null,
        description: String? = null,
        source: String = "runtime",
    ) {
        require(name.isNotBlank()) { "Backend name must be a non-empty string" }
        backends[name] =
            BackendRegistration(
                factory = factory,
                backendClass = backendClass,
                description = description,
                source = source,
            )
        aliases
            .filterValues { registeredName -> registeredName == name }
            .keys
            .toList()
            .forEach(aliases::remove)
        registerAliases(name, aliasList)
    }

    /**
     * Associates aliases with a registered backend name.
     *
     * @param name Name or identifier.
     * @param aliasList Aliases to map to the backend name.
     * @param overwrite Whether existing aliases can be overwritten.
     */
    @Synchronized
    fun registerAliases(
        name: String,
        aliasList: List<String>,
        overwrite: Boolean = false,
    ) {
        require(name in backends) { "Backend '$name' not found." }
        aliasList.forEach { alias ->
            if (alias.isNotBlank()) {
                require(alias !in backends || alias == name) {
                    "Alias '$alias' conflicts with canonical backend name '$alias'."
                }
                val existing = aliases[alias]
                require(overwrite || existing == null || existing == name) {
                    "Alias '$alias' is already registered for backend '$existing'."
                }
                aliases[alias] = name
            }
        }
    }

    /**
     * Discovers backends via Java ServiceLoader and applies provider registrations.
     *
     * When [force] is true, previously discovered providers are reloaded.
     *
     * @param force Whether discovery should be forced even if already done.
     */
    @Synchronized
    fun discoverBackends(force: Boolean = false): Int {
        if (discovered) {
            if (!force) {
                return 0
            }
            discovered = false
        }
        var loadedProviders = 0
        val loader = ServiceLoader.load(BackendDiscoveryProvider::class.java)
        val iterator = loader.iterator()
        while (true) {
            val provider =
                try {
                    if (!iterator.hasNext()) {
                        break
                    }
                    iterator.next()
                } catch (e: ServiceConfigurationError) {
                    System.err.println(
                        "Backend discovery provider loading failed: ${e.message ?: e::class.simpleName}",
                    )
                    continue
                }
            val backendsSnapshot = backends.toMap()
            val aliasesSnapshot = aliases.toMap()
            val registrationFailure =
                runCatching {
                    provider.registerBackends(this)
                }.exceptionOrNull()
            if (registrationFailure == null) {
                loadedProviders += 1
            } else {
                if (registrationFailure is Error) {
                    throw registrationFailure
                }
                backends.clear()
                backends.putAll(backendsSnapshot)
                aliases.clear()
                aliases.putAll(aliasesSnapshot)
                val providerName = provider::class.qualifiedName ?: provider.javaClass.name
                System.err.println(
                    "Backend discovery provider failed: $providerName: " +
                        "${registrationFailure.message ?: registrationFailure::class.simpleName}",
                )
            }
        }
        discovered = true
        return loadedProviders
    }

    @Synchronized
    private fun ensureDiscovered() {
        if (!discovered) {
            discoverBackends()
        }
    }

    /**
     * Creates a backend instance by canonical name or alias.
     *
     * @param name Backend canonical name or alias.
     */
    fun create(name: String): BaseBackend {
        ensureDiscovered()
        val resolved = aliases[name] ?: name
        val registration =
            backends[resolved]
                ?: throw NoSuchElementException("Backend '$name' not found. Available backends: ${availableNames()}")
        return registration.factory()
    }

    /**
     * Returns whether a backend name or alias exists.
     *
     * @param name Backend canonical name or alias.
     */
    fun contains(name: String): Boolean {
        ensureDiscovered()
        return name in backends || name in aliases
    }

    /** Returns canonical backend names. */
    fun keys(): Set<String> {
        ensureDiscovered()
        return backends.keys
    }

    /** Returns sorted canonical names and aliases. */
    fun availableNames(): List<String> {
        ensureDiscovered()
        return (backends.keys + aliases.keys).sorted()
    }

    /** Returns canonical name to `[canonical + aliases]` mapping. */
    fun listAllNames(): Map<String, List<String>> {
        ensureDiscovered()
        return backends.keys.associateWith { name ->
            listOf(name) + aliasesFor(name)
        }
    }

    /**
     * Returns metadata for a backend resolved from name or alias.
     *
     * @param name Backend canonical name or alias.
     */
    fun getBackendInfo(name: String): BackendInfo {
        ensureDiscovered()
        val resolved = aliases[name] ?: name
        val registration =
            backends[resolved]
                ?: throw NoSuchElementException("Backend '$name' not found. Available backends: ${availableNames()}")
        val backendClass = registration.backendClass
        val qualifiedName = backendClass?.qualifiedName ?: backendClass?.java?.name ?: "unknown"
        val module = qualifiedName.substringBeforeLast('.', "")
        return BackendInfo(
            name = resolved,
            aliases = aliasesFor(resolved),
            implementationClass = qualifiedName,
            module = module,
            description = registration.description ?: "No documentation available",
            source = registration.source,
        )
    }

    /** Returns metadata for all registered canonical backends. */
    fun listBackendInfo(): List<BackendInfo> {
        ensureDiscovered()
        return backends.keys.sorted().map { name ->
            getBackendInfo(name)
        }
    }

    private fun aliasesFor(name: String): List<String> = aliases.filterValues { it == name }.keys.sorted()

    /**
     * Clears all registrations and aliases.
     *
     * When [resetDiscovery] is true, discovery state resets and any bootstrap registrations are re-applied.
     *
     * @param resetDiscovery Whether to reset discovery state.
     */
    @JvmOverloads
    fun clear(resetDiscovery: Boolean = true) {
        backends.clear()
        aliases.clear()
        if (resetDiscovery) {
            bootstrap?.invoke(this)
            discovered = false
        }
    }
}

/** Process-wide default backend registry preloaded with built-in backends. */
val BACKEND_REGISTRY =
    BackendRegistry {
        register(
            name = "inmemory",
            factory = ::InMemoryBackend,
            backendClass = InMemoryBackend::class,
            description = "In-memory backend for temporary dataset and experiment storage.",
            source = "builtin",
        )
        register(
            name = "local/csv",
            factory = { LocalCsvBackend(".") },
            aliasList = listOf("csv"),
            backendClass = LocalCsvBackend::class,
            description = "Local filesystem backend using CSV storage.",
            source = "builtin",
        )
        register(
            name = "local/jsonl",
            factory = { LocalJsonlBackend(".") },
            aliasList = listOf("jsonl"),
            backendClass = LocalJsonlBackend::class,
            description = "Local filesystem backend using JSONL storage.",
            source = "builtin",
        )
    }

/**
 * Convenience helper to register a backend into [BACKEND_REGISTRY].
 *
 * @param name Canonical backend name.
 * @param factory Factory that creates backend instances.
 * @param aliases Aliases to register for [name].
 * @param backendClass Optional backend implementation class metadata.
 * @param description Optional human-readable backend description.
 * @param source Registration source label.
 */
fun registerBackend(
    name: String,
    factory: () -> BaseBackend,
    aliases: List<String> = emptyList(),
    backendClass: KClass<out BaseBackend>? = null,
    description: String? = null,
    source: String = "runtime",
) {
    BACKEND_REGISTRY.register(name, factory, aliases, backendClass, description, source)
}
