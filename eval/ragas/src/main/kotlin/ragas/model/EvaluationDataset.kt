package ragas.model

import kotlin.reflect.KClass

/**
 * Homogeneous collection of evaluation samples.
 *
 * All samples must be the same runtime type (single-turn or multi-turn).
 */
data class EvaluationDataset<T : Sample>(
    val samples: List<T>,
) : Iterable<T> {
    init {
        validateSamples(samples)
    }

    /** Converts all samples to map form for serialization/reporting. */
    fun toList(): List<Map<String, Any?>> = samples.map { it.toMap() }

    /** Returns available feature/column names inferred from the first sample. */
    fun features(): Set<String> = samples.firstOrNull()?.toMap()?.keys ?: emptySet()

    /** Returns the runtime sample class, or `null` when the dataset is empty. */
    fun getSampleType(): KClass<out Sample>? = samples.firstOrNull()?.let { it::class }

    /** Iterates samples in insertion order. */
    override fun iterator(): Iterator<T> = samples.iterator()

    private fun validateSamples(items: List<T>) {
        if (items.isEmpty()) {
            return
        }
        val firstType = items.first()::class
        items.drop(1).forEachIndexed { index, sample ->
            require(sample::class == firstType) {
                "Sample at index ${index + 1} is ${sample::class}, expected $firstType."
            }
        }
    }
}
