package ragas.metrics.defaults

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class LlmJsonSupportTest {
    @Test
    fun parseFirstJsonObjectSkipsMalformedBalancedObject() {
        val raw = """example: {not-valid-json} final: {"rating":2}"""

        val parsed = LlmJsonSupport.parseFirstJsonObject(raw)

        assertEquals(2, parsed?.let { LlmJsonSupport.readIntLike(it, "rating") })
    }

    @Test
    fun parseFirstJsonObjectSkipsLeadingUnclosedFragment() {
        val raw = """prefix {"oops": final {"rating":4}"""

        val parsed = LlmJsonSupport.parseFirstJsonObject(raw)

        assertEquals(4, parsed?.let { LlmJsonSupport.readIntLike(it, "rating") })
    }

    @Test
    fun parseFirstJsonObjectReturnsNullWhenNoValidObjectExists() {
        val raw = """no object here {still-not-json"""

        val parsed = LlmJsonSupport.parseFirstJsonObject(raw)

        assertNull(parsed)
    }
}
