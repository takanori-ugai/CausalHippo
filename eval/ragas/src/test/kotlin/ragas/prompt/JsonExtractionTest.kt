package ragas.prompt

import java.lang.reflect.Method
import kotlin.test.Test
import kotlin.test.assertEquals

class JsonExtractionTest {
    @Test
    fun extractFirstJsonObjectIgnoresBracesInStrings() {
        val method: Method = SimplePrompt.Companion::class.java.getDeclaredMethod("extractFirstJsonObject", String::class.java)
        method.isAccessible = true

        val input =
            """
            Some preamble
            {
                "data": "A string with { braces } and \" escaped quotes",
                "nested": { "key": "value" }
            }
            Postamble
            """.trimIndent()

        val expected =
            """
            {
                "data": "A string with { braces } and \" escaped quotes",
                "nested": { "key": "value" }
            }
            """.trimIndent()

        val result = method.invoke(SimplePrompt.Companion, input) as String
        assertEquals(expected, result)
    }
}
