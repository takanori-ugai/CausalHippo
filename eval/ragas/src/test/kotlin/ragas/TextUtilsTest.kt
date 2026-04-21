package ragas

import ragas.metrics.tokenize
import kotlin.test.Test
import kotlin.test.assertEquals

class TextUtilsTest {
    @Test
    fun tokenizeSupportsUnicodeLettersAndNumbers() {
        assertEquals(listOf("café", "123"), tokenize("Café 123"))
        assertEquals(listOf("北京123"), tokenize("北京123"))
        assertEquals(listOf("नमस्ते"), tokenize("नमस्ते"))
    }
}
