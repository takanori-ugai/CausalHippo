package hipporag.demo

/**
 * Demo entry point configured for OpenAI-compatible endpoints.
 */
fun main(args: Array<String>) {
    runDemo(args) { config, parsed ->
        config.llmBaseUrl = parsed.llmBaseUrl
        config.embeddingBaseUrl = parsed.embeddingBaseUrl
        config.forceIndexFromScratch = false
        config.forceOpenieFromScratch = false
    }
}
