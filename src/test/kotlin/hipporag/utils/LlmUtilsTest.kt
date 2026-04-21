package hipporag.utils

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class LlmUtilsTest {
    @Test
    fun testFilterInvalidTriplesRemovesDuplicates() {
        val triples =
            listOf(
                listOf("A", "relates", "B"),
                listOf("A", "relates", "B"),
                listOf("C", "knows", "D"),
            )
        val filtered = filterInvalidTriples(triples)

        assertEquals(2, filtered.size)
        assertTrue(filtered.contains(listOf("A", "relates", "B")))
        assertTrue(filtered.contains(listOf("C", "knows", "D")))
    }

    @Test
    fun testFilterInvalidTriplesRemovesIncomplete() {
        val triples =
            listOf(
                listOf("A", "relates", "B"),
                listOf("A", "relates"),
                listOf("A"),
                listOf(),
                listOf("C", "knows", "D"),
            )
        val filtered = filterInvalidTriples(triples)

        assertEquals(2, filtered.size)
        assertTrue(filtered.contains(listOf("A", "relates", "B")))
        assertTrue(filtered.contains(listOf("C", "knows", "D")))
    }

    @Test
    fun testFilterInvalidTriplesEmptyInput() {
        val triples = emptyList<List<String>>()
        val filtered = filterInvalidTriples(triples)

        assertTrue(filtered.isEmpty())
    }

    @Test
    fun testSafeUnicodeDecodeString() {
        val input = "Hello World"
        val result = safeUnicodeDecode(input)
        assertEquals("Hello World", result)
    }

    @Test
    fun testSafeUnicodeDecodeByteArray() {
        val input = "Hello World".toByteArray(Charsets.UTF_8)
        val result = safeUnicodeDecode(input)
        assertEquals("Hello World", result)
    }

    @Test
    fun testSafeUnicodeDecodeWithUnicodeEscapes() {
        val input = "Hello \\u0041 World"
        val result = safeUnicodeDecode(input)
        assertEquals("Hello A World", result)
    }

    @Test
    fun testSafeUnicodeDecodeWithMultipleEscapes() {
        val input = "\\u0048\\u0065\\u006c\\u006c\\u006f"
        val result = safeUnicodeDecode(input)
        assertEquals("Hello", result)
    }

    @Test
    fun testSafeUnicodeDecodeNoEscapes() {
        val input = "No escapes here"
        val result = safeUnicodeDecode(input)
        assertEquals("No escapes here", result)
    }

    @Test
    fun testSafeUnicodeDecodeByteArrayWithUnicodeEscapes() {
        val input = "Hello \\u0041 World".toByteArray(Charsets.UTF_8)
        val result = safeUnicodeDecode(input)
        assertEquals("Hello A World", result)
    }

    @Test
    fun testSafeUnicodeDecodeSurrogatePair() {
        val input = "\\uD83D\\uDE00"
        val result = safeUnicodeDecode(input)
        assertEquals("\uD83D\uDE00", result)
    }

    @Test
    fun testFixBrokenGeneratedJsonValidJson() {
        val validJson = """{"key": "value", "array": [1, 2, 3]}"""
        val result = fixBrokenGeneratedJson(validJson)
        assertEquals(validJson, result)
    }

    @Test
    fun testFixBrokenGeneratedJsonUnclosedBrace() {
        val brokenJson = """{"key": "value", "array": [1, 2, 3]"""
        val result = fixBrokenGeneratedJson(brokenJson)
        assertTrue(result.endsWith("]}"))
        assertTrue(result.contains("\"key\": \"value\""))
    }

    @Test
    fun testFixBrokenGeneratedJsonUnclosedArray() {
        val brokenJson = """{"key": "value", "array": [1, 2, 3"""
        val result = fixBrokenGeneratedJson(brokenJson)
        assertTrue(result.endsWith("]}"))
    }

    @Test
    fun testFixBrokenGeneratedJsonWithTrailingComma() {
        val brokenJson = """{"key": "value", "array": [1, 2, 3],"""
        val result = fixBrokenGeneratedJson(brokenJson)
        assertFalse(result.endsWith(","))
        assertTrue(result.contains("\"array\""))
    }

    @Test
    fun testFixBrokenGeneratedJsonMultipleUnclosed() {
        val brokenJson = """{"key": {"nested": {"deep": "value"""
        val result = fixBrokenGeneratedJson(brokenJson)
        val openBraces = result.count { it == '{' }
        val closeBraces = result.count { it == '}' }
        assertEquals(openBraces, closeBraces)
    }

    @Test
    fun testRetryWithBackoffSuccess() {
        var attempts = 0
        val result =
            retryWithBackoff(maxAttempts = 3) {
                attempts++
                "success"
            }
        assertEquals("success", result)
        assertEquals(1, attempts)
    }

    @Test
    fun testRetryWithBackoffRetriesOnException() {
        var attempts = 0
        val result =
            retryWithBackoff(maxAttempts = 3, baseDelayMillis = 1) {
                attempts++
                if (attempts < 3) error("Temporary failure")
                "success"
            }
        assertEquals("success", result)
        assertEquals(3, attempts)
    }

    @Test
    fun testRetryWithBackoffThrowsAfterMaxAttempts() {
        var attempts = 0
        assertFailsWith<IllegalStateException> {
            retryWithBackoff(maxAttempts = 3, baseDelayMillis = 1) {
                attempts++
                error("Persistent failure")
            }
        }
        assertEquals(3, attempts)
    }

    @Test
    fun testRetryWithBackoffCustomRetryCondition() {
        var attempts = 0
        val result =
            retryWithBackoff(
                maxAttempts = 3,
                baseDelayMillis = 1,
                retryOn = { it is IllegalStateException },
            ) {
                attempts++
                if (attempts < 2) error("Retry")
                "success"
            }
        assertEquals("success", result)
        assertEquals(2, attempts)
    }

    @Test
    fun testRetryWithBackoffDoesNotRetryWrongException() {
        var attempts = 0
        assertFailsWith<IllegalArgumentException> {
            retryWithBackoff(
                maxAttempts = 3,
                retryOn = { it is IllegalStateException },
            ) {
                attempts++
                throw IllegalArgumentException("Wrong exception")
            }
        }
        assertEquals(1, attempts)
    }

    @Test
    fun testConvertFormatToTemplateSimple() {
        val original = "Hello {name}!"
        val result = convertFormatToTemplate(original)
        assertEquals("Hello \${name}!", result)
    }

    @Test
    fun testConvertFormatToTemplateWithMapping() {
        val original = "Hello {name}!"
        val mapping = mapOf("name" to "username")
        val result = convertFormatToTemplate(original, placeholderMapping = mapping)
        assertEquals("Hello \${username}!", result)
    }

    @Test
    fun testConvertFormatToTemplateWithStaticValues() {
        val original = "Hello {name}, you are {age} years old"
        val statics = mapOf("age" to 25)
        val result = convertFormatToTemplate(original, staticValues = statics)
        assertEquals("Hello \${name}, you are 25 years old", result)
    }

    @Test
    fun testConvertFormatToTemplateMultiplePlaceholders() {
        val original = "{greeting} {name}, welcome to {place}!"
        val result = convertFormatToTemplate(original)
        assertEquals("\${greeting} \${name}, welcome to \${place}!", result)
    }

    @Test
    fun testConvertFormatToTemplateNoPlaceholders() {
        val original = "Hello World!"
        val result = convertFormatToTemplate(original)
        assertEquals("Hello World!", result)
    }

    @Test
    fun testConvertFormatToTemplateMixedMappingAndStatic() {
        val original = "Hello {name}, you are {age} years old in {city}"
        val mapping = mapOf("name" to "username")
        val statics = mapOf("age" to 30)
        val result = convertFormatToTemplate(original, placeholderMapping = mapping, staticValues = statics)
        assertEquals("Hello \${username}, you are 30 years old in \${city}", result)
    }
}
