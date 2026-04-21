package ragas.testset.synthesizers

import kotlin.test.Test
import kotlin.test.assertTrue

class TestsetGeneratorTokenizationTest {
    @Test
    fun tokenStatsCountsUnicodeLetterTokens() {
        val generator = TestsetGenerator()
        val method =
            TestsetGenerator::class.java
                .getDeclaredMethod(
                    "tokenStats",
                    String::class.java,
                ).apply { isAccessible = true }

        val stats = method.invoke(generator, "こんにちは世界 café résumé 12345")
        val totalField = stats.javaClass.getDeclaredField("total").apply { isAccessible = true }
        val total = totalField.getInt(stats)

        assertTrue(total > 0, "Unicode-aware tokenization should keep non-ASCII word tokens.")
    }
}
