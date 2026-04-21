package hipporag.demo

/**
 * Demo entry point configured for local Ollama endpoints.
 */
fun main(args: Array<String>) {
    runDemo(args) { config, parsed ->
        config.llmProvider = "ollama"
        config.ollamaBaseUrl = parsed.llmBaseUrl ?: "http://localhost:11434"
        config.embeddingProvider = "ollama"
        config.embeddingBaseUrl = parsed.embeddingBaseUrl ?: "http://localhost:11434"
    }
}
