package hipporag.config

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class BaseConfigTest {
    @Test
    fun testDefaultValues() {
        val config = BaseConfig()
        assertEquals("outputs", config.saveDir)
        assertEquals("gpt-5.4-mini", config.llmName)
        assertEquals("text-embedding-3-small", config.embeddingModelName)
        assertNull(config.llmBaseUrl)
        assertNull(config.embeddingBaseUrl)
        assertEquals("online", config.openieMode)
        assertEquals(false, config.forceIndexFromScratch)
        assertEquals(false, config.forceOpenieFromScratch)
        assertEquals(0.0, config.temperature)
        assertEquals(5, config.maxRetryAttempts)
    }

    @Test
    fun testCopyWithModifications() {
        val original = BaseConfig()
        val modified = original.copy(saveDir = "custom_outputs", llmName = "gpt-4")

        assertEquals("custom_outputs", modified.saveDir)
        assertEquals("gpt-4", modified.llmName)
        assertEquals("outputs", original.saveDir)
        assertEquals("gpt-5.4-mini", original.llmName)
    }

    @Test
    fun testToMap() {
        val config =
            BaseConfig(
                saveDir = "test_dir",
                llmName = "test-model",
                openAiApiKey = "secret-key",
            )

        val map = config.toMap()

        assertEquals("test_dir", map["saveDir"])
        assertEquals("test-model", map["llmName"])
        assertEquals("***", map["openAiApiKey"])
        assertTrue(map.containsKey("embeddingModelName"))
        assertTrue(map.containsKey("retrievalTopK"))
        assertTrue(map.containsKey("linkingTopK"))
    }

    @Test
    fun testToMapMasksSecrets() {
        val config =
            BaseConfig(
                openAiApiKey = "my-secret-key",
            )

        val map = config.toMap()

        assertEquals("***", map["openAiApiKey"])
    }

    @Test
    fun testToMapWithNullSecrets() {
        val config = BaseConfig()
        val map = config.toMap()

        assertNull(map["openAiApiKey"])
    }

    @Test
    fun testNumericConfigValues() {
        val config = BaseConfig()
        assertEquals(2048, config.maxNewTokens)
        assertEquals(1, config.numGenChoices)
        assertEquals(200, config.retrievalTopK)
        assertEquals(5, config.linkingTopK)
        assertEquals(0.05, config.passageNodeWeight)
        assertEquals(5, config.qaTopK)
        assertEquals(2047, config.synonymyEdgeTopK)
        assertEquals(0.8, config.synonymyEdgeSimThreshold)
        assertEquals(0.5, config.damping)
    }

    @Test
    fun testBooleanConfigValues() {
        val config = BaseConfig()
        assertEquals(false, config.skipGraph)
        assertEquals(false, config.forceIndexFromScratch)
        assertEquals(false, config.forceOpenieFromScratch)
        assertEquals(false, config.isDirectedGraph)
        assertEquals(true, config.embeddingReturnAsNormalized)
        assertEquals(true, config.saveOpenie)
    }

    @Test
    fun testEmbeddingConfig() {
        val config = BaseConfig()
        assertEquals(16, config.embeddingBatchSize)
        assertEquals(2048, config.embeddingMaxSeqLen)
        assertEquals("auto", config.embeddingModelDtype)
    }

    @Test
    fun testModifyConfigFields() {
        val config = BaseConfig()
        config.saveDir = "modified_dir"
        config.llmName = "modified-model"
        config.temperature = 0.7
        config.retrievalTopK = 100

        assertEquals("modified_dir", config.saveDir)
        assertEquals("modified-model", config.llmName)
        assertEquals(0.7, config.temperature)
        assertEquals(100, config.retrievalTopK)
    }

    @Test
    fun testResponseFormatDefault() {
        val config = BaseConfig()
        val responseFormat = config.responseFormat
        assertNotNull(responseFormat)
        assertEquals("json_object", responseFormat["type"])
    }

    @Test
    fun testPreprocessingConfig() {
        val config = BaseConfig()
        assertEquals("TextPreprocessor", config.textPreprocessorClassName)
        assertEquals("gpt-4o", config.preprocessEncoderName)
        assertEquals(128, config.preprocessChunkOverlapTokenSize)
        assertNull(config.preprocessChunkMaxTokenSize)
        assertEquals("by_token", config.preprocessChunkFunc)
    }
}
