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
                "llmApiKey": "shared-key"
              },
              "causalrag": {
                "modelName": "gpt-causal",
                "templateStyle": "structured",
                "minCausalMatches": 2
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

        val light = common.toLightRagSettings()
        assertEquals("./light-work", light.workingDir)
        assertEquals("openai", light.provider)
        assertEquals("gpt-shared", light.llmModelName)
        assertEquals("text-embedding-shared", light.embeddingModelName)
        assertEquals(600, light.chunkTokenSize)
        assertEquals(listOf("Person", "Location"), light.entityTypes)
    }

    @Test
    fun `returns null for non-common config`() {
        val raw = """{"modelName":"plain-causal"}"""
        assertNull(CommonRagConfigLoader.parseOrNull(raw))
    }
}
