package ragas.integrations

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class UnsupportedFallbacksTest {
    @Test
    fun unsupportedIntegrationUsesStableMessageContract() {
        val thrown =
            assertFailsWith<UnsupportedOperationException> {
                unsupportedIntegration("langsmith")
            }

        assertEquals(
            "Integration 'langsmith' is not yet implemented in ragas-kotlin. Track parity in Plan.md.",
            thrown.message,
        )
    }
}
