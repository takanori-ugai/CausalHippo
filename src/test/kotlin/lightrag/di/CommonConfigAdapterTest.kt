package lightrag.di

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals

class CommonConfigAdapterTest {
    @Test
    fun `loads lightrag config from common json`() {
        val root = Files.createTempDirectory("lightrag-common-config-test")
        try {
            val configPath = root.resolve("common_rag.json")
            Files.writeString(
                configPath,
                """
                {
                  "shared": {
                    "llmProvider": "ollama",
                    "modelName": "qwen2.5:7b",
                    "embeddingModel": "nomic-embed-text",
                    "llmBaseUrl": "http://localhost:11434"
                  },
                  "lightrag": {
                    "workingDir": "./lightrag_workspace",
                    "graphStorageName": "Neo4jGraphStorage",
                    "vectorStorageName": "Neo4jVectorStorage",
                    "chunkTokenSize": 700,
                    "chunkOverlapTokenSize": 70,
                    "entityTypes": ["EntityA", "EntityB"],
                    "language": "Japanese",
                    "cosineBetterThreshold": 0.42
                  }
                }
                """.trimIndent(),
            )

            val config = loadLightRagConfigFromCommonJson(configPath.toString())

            assertEquals("ollama", config.provider)
            assertEquals("qwen2.5:7b", config.ollama.chatModelName)
            assertEquals("nomic-embed-text", config.ollama.embeddingModelName)
            assertEquals("http://localhost:11434", config.ollama.baseUrl)
            assertEquals("./lightrag_workspace", config.storage.workingDir)
            assertEquals("Neo4jGraphStorage", config.storage.graphStorageName)
            assertEquals("Neo4jVectorStorage", config.storage.vectorStorageName)
            assertEquals(700, config.addonConfig.chunkTokenSize)
            assertEquals(70, config.addonConfig.chunkOverlapTokenSize)
            assertEquals(listOf("EntityA", "EntityB"), config.addonConfig.entityTypes)
            assertEquals("Japanese", config.addonConfig.language)
            assertEquals(0.42, config.addonConfig.cosineBetterThreshold)
        } finally {
            root.toFile().deleteRecursively()
        }
    }
}
