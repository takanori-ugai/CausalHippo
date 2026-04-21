package ragas.backends

class TestBackendDiscoveryProvider : BackendDiscoveryProvider {
    override fun registerBackends(registry: BackendRegistry) {
        registry.register(
            name = "test/discovered",
            factory = ::InMemoryBackend,
            aliasList = listOf("td"),
            backendClass = InMemoryBackend::class,
            description = "Test-only discovered backend",
            source = "service-loader-test",
        )
    }
}
