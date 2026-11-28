package hipporag.utils

import java.util.PriorityQueue

/** Returns the highest cosine similarities first, using key order to break ties. */
fun retrieveKnn(
    queryIds: List<String>,
    keyIds: List<String>,
    queryVecs: Array<DoubleArray>,
    keyVecs: Array<DoubleArray>,
    k: Int,
): Map<String, Pair<List<String>, List<Double>>> {
    require(queryIds.size == queryVecs.size) { "queryIds and queryVecs must have the same size" }
    require(keyIds.size == keyVecs.size) { "keyIds and keyVecs must have the same size" }
    require(k >= 0) { "k must be non-negative" }
    if (keyVecs.isEmpty() || queryVecs.isEmpty()) return emptyMap()
    val dimensions = keyVecs.first().size
    require(keyVecs.all { it.size == dimensions } && queryVecs.all { it.size == dimensions }) { "Embedding dimensions must match" }
    val keys = keyVecs.map(::normalizeVector)
    val limit = minOf(k, keys.size)
    val bestFirst = compareByDescending<Neighbor> { it.score }.thenBy { it.index }
    return queryIds.indices.associate { i ->
        val query = normalizeVector(queryVecs[i])
        val best = PriorityQueue(bestFirst.reversed())
        if (limit > 0) {
            for ((j, key) in keys.withIndex()) {
                val score = query.indices.sumOf { dim -> query[dim] * key[dim] }
                if (best.size < limit || score > best.peek().score) {
                    if (best.size == limit) best.remove()
                    best.add(Neighbor(j, score))
                }
            }
        }
        val hits = best.sortedWith(bestFirst)
        queryIds[i] to (hits.map { keyIds[it.index] } to hits.map { it.score })
    }
}

private data class Neighbor(
    val index: Int,
    val score: Double,
)
