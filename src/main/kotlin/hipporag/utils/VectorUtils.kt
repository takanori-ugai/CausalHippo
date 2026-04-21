package hipporag.utils

import kotlin.math.sqrt

/**
 * Returns a unit-length copy of [vector], or the original if it has zero norm.
 */
fun normalizeVector(vector: DoubleArray): DoubleArray {
    var sumSquares = 0.0
    for (v in vector) {
        sumSquares += v * v
    }
    val norm = sqrt(sumSquares)
    if (norm == 0.0) return vector
    return DoubleArray(vector.size) { idx -> vector[idx] / norm }
}
