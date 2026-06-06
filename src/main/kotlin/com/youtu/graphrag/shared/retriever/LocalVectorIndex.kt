package com.youtu.graphrag.shared.retriever

import kotlin.math.sqrt

data class VectorSearchHit<T>(
    val item: T,
    val score: Double,
)

interface TextEmbedder {
    val dimensions: Int
    val modelTag: String

    fun embed(text: String): FloatArray
}

interface SemanticVectorIndex<T> {
    val strategyTag: String

    fun search(
        query: String,
        limit: Int,
    ): List<VectorSearchHit<T>>
}

/**
 * Deterministic offline text embedder used as a JVM-local baseline until full FAISS/HNSW parity is implemented.
 */
class HashTextEmbedder(
    override val dimensions: Int = 384,
    private val modelName: String = "all-MiniLM-L6-v2",
) : TextEmbedder {
    override val modelTag: String = "hash:$modelName"
    private val tokenRegex = Regex("[\\p{L}\\p{N}_]{2,}")

    override fun embed(text: String): FloatArray {
        val vector = FloatArray(dimensions)
        val tokens = tokenRegex.findAll(text.lowercase()).map { match -> match.value }.toList()
        if (tokens.isEmpty()) {
            return vector
        }

        tokens.forEach { token ->
            val hash = token.hashCode()
            val bucket = positiveMod(hash, dimensions)
            val sign = if ((hash and 1) == 0) 1f else -1f
            vector[bucket] += sign
        }

        normalizeInPlace(vector)
        return vector
    }

    private fun positiveMod(
        value: Int,
        divisor: Int,
    ): Int {
        val mod = value % divisor
        return if (mod < 0) mod + divisor else mod
    }

    private fun normalizeInPlace(vector: FloatArray) {
        val norm = sqrt(vector.fold(0.0) { acc, value -> acc + value * value })
        if (norm <= 1e-12) {
            return
        }
        for (index in vector.indices) {
            vector[index] = (vector[index] / norm).toFloat()
        }
    }
}

class LocalVectorIndex<T> private constructor(
    private val entries: List<IndexedVector<T>>,
    private val embedder: TextEmbedder,
) : SemanticVectorIndex<T> {
    override val strategyTag: String = "hybrid_lexical_vector"

    override fun search(
        query: String,
        limit: Int,
    ): List<VectorSearchHit<T>> {
        if (entries.isEmpty() || limit <= 0) {
            return emptyList()
        }

        val queryVector = embedder.embed(query)
        val scored =
            entries.map { entry ->
                VectorSearchHit(
                    item = entry.item,
                    score = vectorCosineSimilarity(queryVector, entry.vector),
                ) to entry.ordinal
            }

        return scored
            .sortedWith(
                compareByDescending<Pair<VectorSearchHit<T>, Int>> { pair -> pair.first.score }
                    .thenBy { pair -> pair.second },
            ).take(limit)
            .map { pair -> pair.first }
    }

    companion object {
        fun <T> build(
            items: List<T>,
            embedder: TextEmbedder = HashTextEmbedder(),
            textSelector: (T) -> String,
            vectorSelector: ((T) -> FloatArray)? = null,
        ): LocalVectorIndex<T> {
            val entries =
                items.mapIndexed { index, item ->
                    IndexedVector(
                        item = item,
                        vector = vectorSelector?.invoke(item) ?: embedder.embed(textSelector(item)),
                        ordinal = index,
                    )
                }
            return LocalVectorIndex(entries, embedder)
        }
    }
}

internal fun vectorCosineSimilarity(
    left: FloatArray,
    right: FloatArray,
): Double {
    if (left.size != right.size) {
        return 0.0
    }
    var dot = 0.0
    for (index in left.indices) {
        dot += left[index] * right[index]
    }
    return dot.coerceIn(-1.0, 1.0)
}

private data class IndexedVector<T>(
    val item: T,
    val vector: FloatArray,
    val ordinal: Int,
)
