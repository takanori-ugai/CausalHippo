package ragas

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.embeddings.BaseRagasEmbedding
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.prompt.DynamicFewShotPrompt
import ragas.prompt.PromptCollection
import ragas.prompt.SimplePrompt
import ragas.runtime.RunConfig
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PromptTest {
    @Test
    fun simplePromptFormatsAndPersists() {
        val prompt =
            SimplePrompt("Answer: {question}")
                .addExample(
                    input = mapOf("question" to "1+1"),
                    output = mapOf("answer" to "2"),
                )

        val formatted = prompt.format(mapOf("question" to "2+2"))
        assertTrue(formatted.contains("Answer: 2+2"))
        assertTrue(formatted.contains("--------EXAMPLES-----------"))
        assertTrue(formatted.contains("Now perform the same with the following input"))
        assertTrue(formatted.contains("Output:"))

        val dir = createTempDirectory("ragas-prompt-test").toFile()
        try {
            val path = dir.resolve("prompt.json").absolutePath
            prompt.save(path)

            val loaded = SimplePrompt.load(path)
            assertEquals(prompt.instruction, loaded.instruction)
            assertEquals(prompt.examples.size, loaded.examples.size)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun simplePromptMatchesPythonPydanticPromptLayout() {
        val prompt =
            SimplePrompt(
                instruction = "Answer: {question}",
                examples =
                    listOf(
                        ragas.prompt.PromptExample(
                            input = mapOf("question" to "1+1"),
                            output = mapOf("answer" to "2"),
                        ),
                    ),
                outputJsonSchema = "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}}}",
            )

        val formatted = prompt.format(mapOf("question" to "2+2"))
        assertTrue(
            formatted.contains(
                "Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:",
            ),
        )
        assertTrue(formatted.contains("Use double quotes, properly escaped with a backslash where necessary."))
        assertTrue(formatted.contains("--------EXAMPLES-----------"))
        assertTrue(formatted.contains("Now perform the same with the following input"))
        assertTrue(formatted.contains("Input: {\n    \"question\": \"2+2\"\n}"))
        assertTrue(formatted.endsWith("Output: "))
    }

    @Test
    fun simplePromptSupportsNoneInputFrame() {
        val prompt = SimplePrompt(instruction = "Do task.")
        val formatted = prompt.format(null)
        assertTrue(formatted.contains("Input: (None)"))
        assertTrue(formatted.endsWith("Output: "))
    }

    @Test
    fun simplePromptSaveFailsIfFileAlreadyExists() {
        val prompt = SimplePrompt("Hello")
        val dir = createTempDirectory("ragas-prompt-overwrite").toFile()
        try {
            val path = dir.resolve("prompt.json").absolutePath
            prompt.save(path)
            assertFailsWith<IllegalArgumentException> {
                prompt.save(path)
            }
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun promptCollectionSaveAndLoad() {
        val holder =
            DummyPromptHolder(
                mutableMapOf(
                    "p1" to SimplePrompt("Hello {name}"),
                    "p2" to SimplePrompt("Bye {name}"),
                ),
            )
        val dir = createTempDirectory("ragas-prompts").toFile()
        try {
            holder.savePrompts(dir.absolutePath)

            val loaded = holder.loadPrompts(dir.absolutePath)
            assertEquals(2, loaded.size)
            assertEquals("Hello {name}", loaded.getValue("p1").instruction)
            assertTrue(dir.resolve("p1_english.json").exists())
            assertTrue(dir.resolve("p2_english.json").exists())
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun simplePromptAdaptTranslatesExamplesAndInstruction() =
        runBlocking {
            val llm = FakeTranslationLlm()
            val prompt =
                SimplePrompt(
                    instruction = "Answer the question.",
                    examples =
                        listOf(
                            ragas.prompt.PromptExample(
                                input = mapOf("question" to "hello"),
                                output = mapOf("answer" to "world"),
                            ),
                        ),
                )

            val adapted = prompt.adapt(targetLanguage = "spanish", llm = llm, adaptInstruction = true)
            assertEquals("spanish", adapted.language)
            assertEquals("hello [spanish]", adapted.examples.first().input["question"])
            assertEquals("world [spanish]", adapted.examples.first().output["answer"])
            assertEquals("Answer the question. [spanish]", adapted.instruction)
            assertTrue(adapted.originalHash != null)
        }

    @Test
    fun dynamicFewShotPromptSelectsMostSimilarExamples() =
        runBlocking {
            val prompt =
                DynamicFewShotPrompt(
                    instruction = "Answer.",
                    maxSimilarExamples = 1,
                    embeddings = FakeEmbedding(),
                ).addExample(
                    input = mapOf("query" to "cat"),
                    output = mapOf("answer" to "animal"),
                ).addExample(
                    input = mapOf("query" to "car"),
                    output = mapOf("answer" to "vehicle"),
                )

            val formatted = prompt.format(mapOf("query" to "cat"))
            assertTrue(formatted.contains("\"answer\": \"animal\""))
            assertTrue(!formatted.contains("\"answer\": \"vehicle\""))
        }
}

private class DummyPromptHolder(
    private var prompts: MutableMap<String, SimplePrompt>,
) : PromptCollection {
    override fun getPrompts(): Map<String, SimplePrompt> = prompts

    override fun setPrompts(prompts: Map<String, SimplePrompt>) {
        this.prompts = prompts.toMutableMap()
    }
}

private class FakeTranslationLlm : BaseRagasLlm {
    override var runConfig: RunConfig = RunConfig()

    override suspend fun generateText(
        prompt: String,
        n: Int,
        temperature: Double?,
        stop: List<String>?,
    ): LlmResult {
        val inputJson = prompt.substringAfter("Input:").substringBeforeLast("Output:").trim()
        val data = Json.parseToJsonElement(inputJson).jsonObject
        val target = data["target_language"]?.jsonPrimitive?.content.orEmpty()
        val statements =
            data["statements"]
                ?.jsonArray
                ?.map { item -> item.jsonPrimitive.content }
                .orEmpty()
        val translated = statements.map { statement -> "$statement [$target]" }
        val response = """{"statements": ${Json.encodeToString(translated)}}"""
        return LlmResult(generations = listOf(LlmGeneration(text = response)))
    }
}

private class FakeEmbedding : BaseRagasEmbedding {
    override suspend fun embedText(text: String): List<Float> =
        when {
            "cat" in text -> listOf(1f, 0f)
            "car" in text -> listOf(0f, 1f)
            else -> listOf(0f, 0f)
        }
}
