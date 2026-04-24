package com.microsoft.graphrag.index

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals

class IndexConfigLoaderTest {
    @Test
    fun `loads shared directories from common config`() {
        val root = Files.createTempDirectory("index-config-loader-test")
        try {
            val configDir = root.resolve("config")
            Files.createDirectories(configDir)
            Files.writeString(
                configDir.resolve("common_rag.json"),
                """
                {
                  "shared": {
                    "rootDir": "graph_project",
                    "inputDir": "docs",
                    "outputDir": "index_output"
                  }
                }
                """.trimIndent(),
            )

            val config = IndexConfigLoader.load(root = root, configPath = null).graphConfig

            assertEquals(root.resolve("graph_project").normalize(), config.rootDir)
            assertEquals(root.resolve("graph_project/docs").normalize(), config.inputDir)
            assertEquals(root.resolve("graph_project/index_output").normalize(), config.outputDir)
            assertEquals(root.resolve("graph_project/index_output/update_output").normalize(), config.updateOutputDir)
        } finally {
            root.toFile().deleteRecursively()
        }
    }
}
