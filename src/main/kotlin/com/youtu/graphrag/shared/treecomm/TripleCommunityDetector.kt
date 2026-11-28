package com.youtu.graphrag.shared.treecomm

import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.embedding.onnx.OnnxEmbeddingModel
import dev.langchain4j.model.embedding.onnx.PoolingMode
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel
import org.apache.commons.math3.ml.clustering.Clusterable
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer
import org.apache.commons.math3.ml.distance.EuclideanDistance
import org.apache.commons.math3.random.MersenneTwister
import java.nio.file.Path
import kotlin.math.sqrt

/** Tree-Comm on level-two entities, using outgoing triples as their semantic representation. */
internal class TripleCommunityDetector(
    private val members: List<String>,
    private val outgoing: Map<String, Map<String, String>>,
    private val degrees: Map<String, Int>,
    private val options: TreeCommOptions,
) {
    private data class Point(
        val name: String,
        val coordinates: DoubleArray,
    ) : Clusterable {
        override fun getPoint(): DoubleArray = coordinates
    }

    private val vectors by lazy {
        val model =
            when (options.embeddingModel) {
                "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2" -> {
                    miniLm
                }

                else -> {
                    val dir = Path.of(options.embeddingModel)
                    OnnxEmbeddingModel(dir.resolve("model.onnx"), dir.resolve("tokenizer.json"), PoolingMode.MEAN)
                }
            }
        members
            .chunked(128)
            .flatMap { batch ->
                val segments =
                    batch.map { name ->
                        val triples = outgoing[name].orEmpty().values.joinToString(" ")
                        TextSegment.from(triples.ifEmpty { name })
                    }
                batch.zip(model.embedAll(segments).content()).map { (name, embedding) ->
                    name to Point(name, embedding.vector().map(Float::toDouble).toDoubleArray())
                }
            }.toMap()
    }

    fun detect(): List<List<String>> {
        val initial = cluster(members).sortedByDescending { it.size }
        var remainingSplits = (options.maxTotalCommunities - initial.size).coerceAtLeast(0)
        return initial.flatMap { group ->
            if (group.size <= 3 || remainingSplits == 0) return@flatMap listOf(group)
            val refined = refine(group)
            if (refined.size - 1 > remainingSplits) return@flatMap listOf(group)
            remainingSplits -= refined.size - 1
            refined
        }
    }

    private fun cluster(names: List<String>): List<List<String>> {
        val k = (names.size / 10).coerceAtLeast(2).coerceAtMost(minOf(names.size / 2, 200))
        if (k <= 1) return listOf(names)
        val points = names.map { vectors.getValue(it) }
        val uniqueCount = points.distinctBy { it.coordinates.asList() }.size
        if (uniqueCount == 1) return listOf(names)
        val clusterer = KMeansPlusPlusClusterer<Point>(minOf(k, uniqueCount), 300, EuclideanDistance(), MersenneTwister(42))
        var best = emptyList<List<String>>()
        var minInertia = Double.POSITIVE_INFINITY
        repeat(5) {
            val clusters = clusterer.cluster(points)
            val inertia =
                clusters.sumOf { group ->
                    group.points.sumOf { point ->
                        point.coordinates.indices.sumOf { i ->
                            val delta = point.coordinates[i] - group.center.point[i]
                            delta * delta
                        }
                    }
                }
            if (inertia < minInertia) {
                minInertia = inertia
                best = clusters.map { it.points.map(Point::name) }
            }
        }
        return best
    }

    private fun refine(group: List<String>): List<List<String>> {
        var clusters = cluster(group)
        repeat(options.maxIterations) {
            val centers = clusters.map { keywords(it).first() }
            val centerSet = centers.toSet()
            val neighbors = centers.map { outgoing[it].orEmpty().keys.intersect(centerSet) }
            val pairs =
                buildList {
                    for (i in clusters.indices) {
                        for (j in i + 1 until clusters.size) {
                            if (clusters[i].size + clusters[j].size > 100) continue
                            val shared = neighbors[i].intersect(neighbors[j]).size
                            val union = neighbors[i].size + neighbors[j].size - shared
                            val structural = shared.toDouble() / union.coerceAtLeast(1)
                            val semantic = cosine(vectors.getValue(centers[i]).point, vectors.getValue(centers[j]).point)
                            val similarity = options.structWeight * structural + (1 - options.structWeight) * semantic
                            if (similarity >= maxOf(0.5, options.mergeThreshold)) add(Triple(i, j, similarity))
                        }
                    }
                }.sortedByDescending { it.third }
            if (pairs.isEmpty()) return clusters
            val used = mutableSetOf<Int>()
            clusters =
                buildList {
                    for ((i, j) in pairs) {
                        if (i in used || j in used) continue
                        add(clusters[i] + clusters[j])
                        used.add(i)
                        used.add(j)
                    }
                    for (i in clusters.indices) if (i !in used) add(clusters[i])
                }
        }
        return clusters
    }

    fun keywords(group: List<String>): List<String> {
        if (group.size <= 5) return group
        val mean = DoubleArray(vectors.getValue(group.first()).point.size)
        for (name in group) {
            val vector = vectors.getValue(name).point
            for (i in mean.indices) mean[i] += vector[i] / group.size
        }
        val maxDegree = group.maxOf { degrees[it] ?: 0 }.coerceAtLeast(1)
        return group
            .sortedByDescending { name ->
                options.structWeight * (degrees[name] ?: 0) / maxDegree +
                    (1 - options.structWeight) * cosine(vectors.getValue(name).point, mean)
            }.take(5)
    }

    private fun cosine(
        a: DoubleArray,
        b: DoubleArray,
    ): Double {
        val norm = sqrt(a.sumOf { it * it } * b.sumOf { it * it })
        return if (norm == 0.0) 0.0 else a.indices.sumOf { a[it] * b[it] } / norm
    }

    companion object {
        private val miniLm by lazy { AllMiniLmL6V2EmbeddingModel(Runnable::run) }
    }
}
