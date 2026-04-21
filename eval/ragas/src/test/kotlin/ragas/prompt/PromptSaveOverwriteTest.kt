package ragas.prompt

import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class PromptSaveOverwriteTest {
    @Test
    fun testSimplePromptSaveOverwrite() {
        val dir = createTempDirectory("ragas-simple-prompt-overwrite").toFile()
        try {
            val path = dir.resolve("prompt.json").absolutePath
            val prompt1 = SimplePrompt("Instruction 1")
            prompt1.save(path)

            val prompt2 = SimplePrompt("Instruction 2")
            // Should fail without overwrite=true
            assertFailsWith<IllegalArgumentException> {
                prompt2.save(path, overwrite = false)
            }

            // Should succeed with overwrite=true
            prompt2.save(path, overwrite = true)
            val loaded = SimplePrompt.load(path)
            assertEquals("Instruction 2", loaded.instruction)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun testDynamicFewShotPromptSaveOverwrite() {
        val dir = createTempDirectory("ragas-dynamic-prompt-overwrite").toFile()
        try {
            val path = dir.resolve("prompt.json").absolutePath
            val prompt1 = DynamicFewShotPrompt("Instruction 1")
            prompt1.save(path)

            val prompt2 = DynamicFewShotPrompt("Instruction 2")
            // Should fail without overwrite=true
            assertFailsWith<IllegalArgumentException> {
                prompt2.save(path, overwrite = false)
            }

            // Should succeed with overwrite=true
            prompt2.save(path, overwrite = true)
            val loaded = DynamicFewShotPrompt.load(path)
            assertEquals("Instruction 2", loaded.instruction)
        } finally {
            dir.deleteRecursively()
        }
    }
}
