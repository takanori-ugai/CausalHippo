package shared.config

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class CommonRagConfigLoaderTest {
    @Test
    fun `parses common config and resolves causalrag config`() {
        val raw =
            """
            {
              "shared": {
                "llmProvider": "openai",
                "modelName": "gpt-shared",
                "embeddingModel": "text-embedding-shared",
                "llmApiKey": "shared-key",
                "chunkTokenSize": 900,
                "chunkOverlapTokenSize": 90,
                "contextTokenBudget": 3500
              },
              "causalrag": {
                "modelName": "gpt-causal",
                "templateStyle": "structured",
                "minCausalMatches": 2,
                "chunk_token_size": 700,
                "ingestChunkOverlapTokenSize": 70,
                "prompt_context_token_budget": 2800
              }
            }
            """.trimIndent()

        val common = CommonRagConfigLoader.parseOrNull(raw)
        assertNotNull(common)
        val pipeline = common.toPipelineConfig()
        assertEquals("gpt-causal", pipeline.modelName)
        assertEquals("text-embedding-shared", pipeline.embeddingModel)
        assertEquals("openai", pipeline.llmProvider)
        assertEquals("shared-key", pipeline.llmApiKey)
        assertEquals("structured", pipeline.templateStyle)
        assertEquals(2, pipeline.minCausalMatches)
        assertEquals(700, pipeline.ingestChunkTokenSize)
        assertEquals(70, pipeline.ingestChunkOverlapTokenSize)
        assertEquals(2800, pipeline.promptContextTokenBudget)
    }

    @Test
    fun `resolves hipporag settings with shared fallback`() {
        val raw =
            """
            {
              "shared": {
                "modelName": "gpt-shared",
                "embeddingModel": "text-embedding-shared",
                "llmProvider": "openai",
                "llmApiKey": "hippo-key"
              },
              "hipporag": {
                "llmName": "gpt-hippo",
                "retrievalTopK": 123,
                "openieMode": "offline"
              }
            }
            """.trimIndent()

        val common = CommonRagConfigLoader.parseOrNull(raw)
        assertNotNull(common)
        val hippo = common.toHippoBaseConfig()
        assertEquals("gpt-hippo", hippo.llmName)
        assertEquals("text-embedding-shared", hippo.embeddingModelName)
        assertEquals("openai", hippo.llmProvider)
        assertEquals("hippo-key", hippo.openAiApiKey)
        assertEquals(123, hippo.retrievalTopK)
        assertEquals("offline", hippo.openieMode)
    }

    @Test
    fun `resolves pathrag and lightrag module settings`() {
        val raw =
            """
            {
              "shared": {
                "llmProvider": "openai",
                "modelName": "gpt-shared",
                "embeddingModel": "text-embedding-shared",
                "embeddingModelDimensions": 1024,
                "llmApiKey": "shared-key",
                "llmBaseUrl": "http://localhost:1234/v1"
              },
              "pathrag": {
                "workingDir": "./path-work",
                "kvStorage": "JsonKVStorage",
                "chunkTokenSize": 512
              },
              "lightrag": {
                "workingDir": "./light-work",
                "graphStorageName": "InMemoryGraphStorage",
                "chunkTokenSize": 600,
                "entityTypes": ["Person", "Location"]
              }
            }
            """.trimIndent()

        val common = CommonRagConfigLoader.parseOrNull(raw)
        assertNotNull(common)

        val path = common.toPathRagSettings()
        assertEquals("./path-work", path.workingDir)
        assertEquals("openai", path.llmProvider)
        assertEquals("gpt-shared", path.llmModelName)
        assertEquals("text-embedding-shared", path.embeddingModelName)
        assertEquals("http://localhost:1234/v1", path.baseUrl)
        assertEquals(512, path.chunkTokenSize)
        val runtimeSettings = path.toRuntimeSettingsMap()
        assertEquals("openai", runtimeSettings["LLM_PROVIDER"])
        assertEquals("gpt-shared", runtimeSettings["OPENAI_MODEL"])
        assertEquals("text-embedding-shared", runtimeSettings["OPENAI_EMBEDDING_MODEL"])
        assertEquals("http://localhost:1234/v1", runtimeSettings["OPENAI_API_BASE"])
        assertEquals("gpt-shared", runtimeSettings["OLLAMA_MODEL"])
        assertEquals("text-embedding-shared", runtimeSettings["OLLAMA_EMBED_MODEL"])
        assertEquals("http://localhost:1234/v1", runtimeSettings["OLLAMA_BASE_URL"])

        val light = common.toLightRagSettings()
        assertEquals("./light-work", light.workingDir)
        assertEquals("openai", light.provider)
        assertEquals("gpt-shared", light.llmModelName)
        assertEquals("text-embedding-shared", light.embeddingModelName)
        assertEquals(1024, light.embeddingModelDimensions)
        assertEquals(600, light.chunkTokenSize)
        assertEquals(listOf("Person", "Location"), light.entityTypes)
    }

    @Test
    fun `returns null for non-common config`() {
        val raw = """{"modelName":"plain-causal"}"""
        assertNull(CommonRagConfigLoader.parseOrNull(raw))
    }

    @Test
    fun `module alias keys override shared canonical keys for causalrag`() {
        val raw =
            """
            {
              "shared": {
                "modelName": "shared-model",
                "embeddingModel": "shared-embed",
                "llmProvider": "openai",
                "llmApiKey": "shared-key",
                "llmBaseUrl": "http://shared/v1",
                "embeddingApiKey": "shared-embed-key"
              },
              "causalrag": {
                "llmModel": "causal-alias-model",
                "embeddingModelName": "causal-alias-embed",
                "provider": "ollama",
                "apiKey": "causal-api-key",
                "baseUrl": "http://causal/v1",
                "embeddingApiKey": "causal-embed-key"
              }
            }
            """.trimIndent()

        val common = CommonRagConfigLoader.parseOrNull(raw)
        assertNotNull(common)
        val pipeline = common.toPipelineConfig()
        assertEquals("causal-alias-model", pipeline.modelName)
        assertEquals("causal-alias-embed", pipeline.embeddingModel)
        assertEquals("ollama", pipeline.llmProvider)
        assertEquals("causal-api-key", pipeline.llmApiKey)
        assertEquals("http://causal/v1", pipeline.llmBaseUrl)
        assertEquals("causal-embed-key", pipeline.embeddingApiKey)
    }

    @Test
    fun `module alias keys override shared canonical keys for pathrag`() {
        val raw =
            """
            {
              "shared": {
                "llmProvider": "openai",
                "modelName": "shared-model",
                "embeddingModel": "shared-embed",
                "llmApiKey": "shared-key",
                "llmBaseUrl": "http://shared/v1"
              },
              "pathrag": {
                "provider": "ollama",
                "modelName": "path-alias-model",
                "openaiEmbeddingModel": "path-alias-embed",
                "openAiApiKey": "path-api-key",
                "openAiApiBase": "http://path/v1"
              }
            }
            """.trimIndent()

        val common = CommonRagConfigLoader.parseOrNull(raw)
        assertNotNull(common)
        val path = common.toPathRagSettings()
        assertEquals("ollama", path.llmProvider)
        assertEquals("path-alias-model", path.llmModelName)
        assertEquals("path-alias-embed", path.embeddingModelName)
        assertEquals("path-api-key", path.apiKey)
        assertEquals("http://path/v1", path.baseUrl)
    }

    @Test
    fun `module alias keys override shared canonical keys for lightrag`() {
        val raw =
            """
            {
              "shared": {
                "llmProvider": "openai",
                "modelName": "shared-model",
                "embeddingModel": "shared-embed",
                "embeddingModelDimensions": 1536,
                "llmApiKey": "shared-key",
                "llmBaseUrl": "http://shared/v1"
              },
              "lightrag": {
                "provider": "ollama",
                "modelName": "light-alias-model",
                "embeddingModel": "light-alias-embed",
                "embedding_dimension": 768,
                "openAiApiKey": "light-api-key",
                "llmBaseUrl": "http://light/v1"
              }
            }
            """.trimIndent()

        val common = CommonRagConfigLoader.parseOrNull(raw)
        assertNotNull(common)
        val light = common.toLightRagSettings()
        assertEquals("ollama", light.provider)
        assertEquals("light-alias-model", light.llmModelName)
        assertEquals("light-alias-embed", light.embeddingModelName)
        assertEquals(768, light.embeddingModelDimensions)
        assertEquals("light-api-key", light.apiKey)
        assertEquals("http://light/v1", light.baseUrl)
    }
}
