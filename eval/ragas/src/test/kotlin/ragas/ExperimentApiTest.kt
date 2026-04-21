package ragas

import kotlinx.coroutines.runBlocking
import ragas.backends.InMemoryBackend
import ragas.model.SingleTurnSample
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ExperimentApiTest {
    @Test
    fun experimentWrapperRunsAndSavesResults() =
        runBlocking {
            val backend = InMemoryBackend()
            val dataset =
                listOf(
                    SingleTurnSample(userInput = "q1", response = "a1"),
                    SingleTurnSample(userInput = "q2", response = "a2"),
                )

            val wrapper =
                experiment<SingleTurnSample>(backend = backend, namePrefix = "demo") { row ->
                    mapOf(
                        "user_input" to row.userInput,
                        "response_length" to row.response.orEmpty().length,
                    )
                }

            val result = wrapper.arun(dataset = dataset, name = "baseline")
            assertEquals("demo-baseline", result.name)

            val stored = backend.loadExperiment("demo-baseline")
            assertEquals(2, stored.size)
            assertEquals("q1", stored[0]["user_input"])
            assertEquals(2, (stored[1]["response_length"] as Number).toInt())
        }

    @Test
    fun experimentWrapperContinuesWhenTaskFails() =
        runBlocking {
            val backend = InMemoryBackend()
            val dataset = listOf(1, 2, 3)

            val wrapper =
                experiment<Int>(backend = backend) { value ->
                    if (value == 2) {
                        error("boom")
                    }
                    mapOf("value" to value)
                }

            val result = wrapper.arun(dataset = dataset, name = "partial")
            val stored = backend.loadExperiment(result.name)
            assertEquals(2, stored.size)
            val values = stored.mapNotNull { row -> (row["value"] as Number?)?.toInt() }.sorted()
            assertEquals(listOf(1, 3), values)
        }

    @Test
    fun resolveBackendRejectsUnsupportedTypes() {
        val error =
            assertFailsWith<IllegalArgumentException> {
                Experiment.resolveBackend(123)
            }
        assertTrue(error.message.orEmpty().contains("Unsupported backend type"))
    }
}
