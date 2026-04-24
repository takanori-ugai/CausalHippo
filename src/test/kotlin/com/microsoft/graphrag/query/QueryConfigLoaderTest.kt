package com.microsoft.graphrag.query

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals

class QueryConfigLoaderTest {
    @Test
    fun `loads shared model defaults from common config`() {
        val root = Files.createTempDirectory("query-config-loader-test")
        try {
            val configDir = root.resolve("config")
            Files.createDirectories(configDir)
            Files.writeString(
                configDir.resolve("common_rag.json"),
                """
                {
                  "shared": {
                    "modelName": "gpt-5.4-mini",
                    "embeddingModel": "text-embedding-3-large"
                  }
                }
                """.trimIndent(),
            )

            val config = QueryConfigLoader.load(root, configPath = null)

            assertEquals("gpt-5.4-mini", config.defaultChatModel?.model)
            assertEquals("text-embedding-3-large", config.defaultEmbeddingModel)
            assertEquals("gpt-5.4-mini", config.basic.chat?.model)
            assertEquals("text-embedding-3-large", config.basic.embeddingModel)
        } finally {
            root.toFile().deleteRecursively()
        }
    }
}
