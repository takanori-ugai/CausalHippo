package com.youtu.graphrag.parity

import kotlin.test.Test

class JsonParityUtilsTest {
    @Test
    fun `canonicalization ignores object key ordering`() {
        val left = """{"b":2,"a":{"z":1,"k":2}}"""
        val right = """{"a":{"k":2,"z":1},"b":2}"""

        JsonParityUtils.assertJsonEquivalent(left, right)
    }

    @Test
    fun `approximate matcher enforces tolerance`() {
        JsonParityUtils.assertApproximatelyEquals(
            expected = 1.0,
            actual = 1.0002,
            tolerance = 0.001,
            label = "score",
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun `throws on negative tolerance`() {
        JsonParityUtils.assertApproximatelyEquals(
            expected = 1.0,
            actual = 1.0,
            tolerance = -0.1,
        )
    }
}
