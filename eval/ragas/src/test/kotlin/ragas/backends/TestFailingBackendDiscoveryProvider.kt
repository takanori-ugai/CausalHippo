package ragas.backends

class TestFailingBackendDiscoveryProvider : BackendDiscoveryProvider {
    override fun registerBackends(registry: BackendRegistry) {
        registry.register(
            name = "test/partial",
            factory = ::InMemoryBackend,
            aliasList = listOf("tp"),
            backendClass = InMemoryBackend::class,
            description = "Should be rolled back when provider fails",
            source = "service-loader-test",
        )
        error("Intentional provider failure after partial registration")
    }
}
