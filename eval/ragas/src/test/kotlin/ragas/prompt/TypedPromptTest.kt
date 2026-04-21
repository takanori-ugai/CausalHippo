package ragas.prompt

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.serializer
import ragas.embeddings.BaseRagasEmbedding
import ragas.llms.BaseRagasLlm
import ragas.llms.LlmGeneration
import ragas.llms.LlmResult
import ragas.runtime.RunConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TypedPromptTest {
    @Test
    fun typedPromptFormatsWithExamplesAndInputFrame() =
        runBlocking {
            val prompt =
                TypedPrompt(
                    inputSerializer = MathInput.serializer(),
                    outputSerializer = MathOutput.serializer(),
                    model =
                        TypedPromptModel(
                            instruction = "Solve the math question.",
                            examples =
                                listOf(
                                    TypedPromptExample(
                                        input = MathInput(question = "1+1"),
                                        output = MathOutput(answer = "2"),
                                    ),
                                ),
                        ),
                )

            val formatted = prompt.format(MathInput(question = "2+2"))

            assertTrue(formatted.contains("Solve the math question."))
            assertTrue(formatted.contains("--------EXAMPLES-----------"))
            assertTrue(formatted.contains("Now perform the same with the following input"))
            assertTrue(formatted.contains("\"question\": \"2+2\""))
            assertTrue(formatted.endsWith("Output: "))
        }

    @Test
    fun structuredParserExtractsJsonFromWrappedText() {
        val parser = StructuredOutputParser(MathOutput.serializer())

        val parsed =
            parser.parse(
                """
                Here is your answer:
                {"answer":"4"}
                Thanks.
                """.trimIndent(),
            )

        assertEquals(MathOutput(answer = "4"), parsed)
    }

    @Test
    fun structuredParserExtractsNumericPrimitiveFromWrappedText() {
        val parser = StructuredOutputParser(Int.serializer())

        val parsed = parser.parse("Result: 42.")

        assertEquals(42, parsed)
    }

    @Test
    fun structuredParserExtractsBooleanPrimitiveFromWrappedText() {
        val parser = StructuredOutputParser(Boolean.serializer())

        val parsed = parser.parse("Final verdict is true, proceed.")

        assertEquals(true, parsed)
    }

    @Test
    fun structuredParserExtractsStringPrimitiveFromWrappedText() {
        val parser = StructuredOutputParser(String.serializer())

        val parsed = parser.parse("""The answer is "ok".""")

        assertEquals("ok", parsed)
    }

    @Test
    fun typedPromptGenerateRetriesOnParseFailure() =
        runBlocking {
            val llm = FakeLlm(outputs = listOf("not-json", "{\"answer\":\"4\"}"))
            val prompt =
                TypedPrompt(
                    inputSerializer = MathInput.serializer(),
                    outputSerializer = MathOutput.serializer(),
                    model = TypedPromptModel(instruction = "Solve the math question."),
                )

            val result =
                prompt.generate(
                    llm = llm,
                    input = MathInput(question = "2+2"),
                    config = StructuredOutputParserConfig(maxParseRetries = 1),
                )

            assertEquals(MathOutput(answer = "4"), result)
            assertEquals(2, llm.calls)
        }

    @Test
    fun dynamicFewShotTypedPromptSelectsMostSimilarExample() =
        runBlocking {
            val prompt =
                DynamicFewShotTypedPrompt(
                    inputSerializer = QueryInput.serializer(),
                    outputSerializer = LabelOutput.serializer(),
                    model =
                        TypedPromptModel(
                            instruction = "Classify the query.",
                            examples =
                                listOf(
                                    TypedPromptExample(
                                        input = QueryInput(query = "cat"),
                                        output = LabelOutput(label = "animal"),
                                    ),
                                    TypedPromptExample(
                                        input = QueryInput(query = "car"),
                                        output = LabelOutput(label = "vehicle"),
                                    ),
                                ),
                        ),
                    maxSimilarExamples = 1,
                    embeddings = TypedFakeEmbedding(),
                )

            val formatted = prompt.format(QueryInput(query = "cat"))
            assertTrue(formatted.contains("\"label\": \"animal\""))
            assertTrue(!formatted.contains("\"label\": \"vehicle\""))
        }

    @Test
    fun fewShotPydanticPromptUsesSchemaAndParsesOutput() =
        runBlocking {
            val prompt =
                FewShotPydanticPrompt(
                    inputSerializer = MathInput.serializer(),
                    outputSerializer = MathOutput.serializer(),
                    instruction = "Solve the math question.",
                    outputJsonSchema = """{"type":"object","properties":{"answer":{"type":"string"}}}""",
                    examples =
                        listOf(
                            TypedPromptExample(
                                input = MathInput(question = "1+1"),
                                output = MathOutput(answer = "2"),
                            ),
                        ),
                )

            val formatted = prompt.format(MathInput(question = "2+2"))
            assertTrue(formatted.contains("\"type\":\"object\""))
            assertTrue(formatted.contains("\"answer\": \"2\""))

            val llm = FakeLlm(outputs = listOf("""{"answer":"4"}"""))
            val result = prompt.generate(llm = llm, input = MathInput(question = "2+2"))
            assertEquals(MathOutput(answer = "4"), result)
        }

    @Test
    fun typedPromptFormatHandlesRecursiveOutputDescriptor() =
        runBlocking {
            val prompt =
                TypedPrompt(
                    inputSerializer = MathInput.serializer(),
                    outputSerializer = RecursiveOutput.serializer(),
                    model = TypedPromptModel(instruction = "Extract recursive JSON."),
                )

            val formatted = prompt.format(MathInput(question = "2+2"))
            assertTrue(formatted.contains("Expected JSON shape:"))
            assertTrue(formatted.contains("\"child\": recursive"))
        }

    @Test
    fun typedPromptFormatIncludesLanguageInstructionWhenNonDefault() =
        runBlocking {
            val prompt =
                TypedPrompt(
                    inputSerializer = MathInput.serializer(),
                    outputSerializer = MathOutput.serializer(),
                    model =
                        TypedPromptModel(
                            instruction = "Resuelve la pregunta.",
                            language = "spanish",
                        ),
                )

            val formatted = prompt.format(MathInput(question = "2+2"))
            assertTrue(formatted.contains("Write the response in spanish."))
        }

    @Serializable
    private data class MathInput(
        val question: String,
    )

    @Serializable
    private data class MathOutput(
        val answer: String,
    )

    @Serializable
    private data class QueryInput(
        val query: String,
    )

    @Serializable
    private data class LabelOutput(
        val label: String,
    )

    @Serializable
    private data class RecursiveOutput(
        val value: String,
        val child: RecursiveOutput? = null,
    )

    private class FakeLlm(
        private val outputs: List<String>,
    ) : BaseRagasLlm {
        override var runConfig: RunConfig = RunConfig()
        var calls: Int = 0

        override suspend fun generateText(
            prompt: String,
            n: Int,
            temperature: Double?,
            stop: List<String>?,
        ): LlmResult {
            val response = outputs[calls]
            calls += 1
            return LlmResult(generations = listOf(LlmGeneration(text = response)))
        }
    }

    private class TypedFakeEmbedding : BaseRagasEmbedding {
        override suspend fun embedText(text: String): List<Float> =
            when {
                "\"cat\"" in text -> listOf(1f, 0f)
                "\"car\"" in text -> listOf(0f, 1f)
                else -> listOf(0f, 0f)
            }

        override suspend fun embedTexts(texts: List<String>): List<List<Float>> = texts.map { text -> embedText(text) }
    }
}
