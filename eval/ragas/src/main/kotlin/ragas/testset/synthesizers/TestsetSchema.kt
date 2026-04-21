package ragas.testset.synthesizers

import ragas.model.EvaluationDataset
import ragas.model.MultiTurnSample
import ragas.model.Sample
import ragas.model.SingleTurnSample

/**
 * One generated testset row with its originating synthesizer name.
 *
 * @property evalSample Evaluation sample payload.
 * @property synthesizerName Name of the synthesizer variant that produced the sample.
 */
data class TestsetSample(
    val evalSample: Sample,
    val synthesizerName: String,
)

/**
 * Container for synthesized samples and metadata about the generation run.
 *
 * @property samples Synthesized testset rows.
 * @property runId Stable identifier for this generation run.
 */
data class Testset(
    val samples: List<TestsetSample>,
    val runId: String =
        java.util.UUID
            .randomUUID()
            .toString(),
) {
    /**
     * Converts this testset to a strongly-typed evaluation dataset.
     *
     * @return [EvaluationDataset] containing only one concrete sample type.
     */
    fun toEvaluationDataset(): EvaluationDataset<out Sample> {
        if (samples.isEmpty()) {
            return EvaluationDataset(emptyList<SingleTurnSample>())
        }
        val evalSamples = samples.map { sample -> sample.evalSample }
        val firstType = evalSamples.first()::class
        require(evalSamples.all { it::class == firstType }) {
            "Mixed sample types are not supported in a single Testset."
        }
        return when (firstType) {
            SingleTurnSample::class -> EvaluationDataset(evalSamples.map { it as SingleTurnSample })
            MultiTurnSample::class -> EvaluationDataset(evalSamples.map { it as MultiTurnSample })
            else -> error("Unsupported sample type: $firstType")
        }
    }

    /**
     * Serializes testset samples to a list-of-maps representation.
     *
     * @return Serialized rows that include `synthesizer_name` metadata.
     */
    fun toList(): List<Map<String, Any?>> =
        samples.map { sample ->
            sample.evalSample.toMap() + mapOf("synthesizer_name" to sample.synthesizerName)
        }

    companion object {
        /**
         * Reconstructs a [Testset] from list-of-maps data.
         *
         * @param data Serialized rows, typically produced by [toList].
         * @return Reconstructed [Testset].
         */
        fun fromList(data: List<Map<String, Any?>>): Testset {
            val samples =
                data.map { row ->
                    val synthesizer = row["synthesizer_name"]?.toString() ?: "unknown"
                    val sample =
                        run {
                            require(row["user_input"] !is List<*>) {
                                "Multi-turn deserialization is not implemented for fromList() yet."
                            }
                            SingleTurnSample(
                                userInput = row["user_input"]?.toString(),
                                response = row["response"]?.toString(),
                                reference = row["reference"]?.toString(),
                            )
                        }
                    TestsetSample(sample, synthesizer)
                }
            return Testset(samples)
        }
    }
}
