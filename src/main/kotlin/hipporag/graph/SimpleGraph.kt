package hipporag.graph

import hipporag.utils.jsonWithDefaults
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.Contextual
import kotlinx.serialization.Serializable
import java.io.File

/**
 * Lightweight in-memory graph with serialization support.
 *
 * @param directed whether edges are directed.
 */
class SimpleGraph(
    private val directed: Boolean,
) {
    private val logger = KotlinLogging.logger {}

    private val vertices = mutableListOf<MutableMap<String, Any>>()
    private val edges = mutableListOf<Edge>()
    private val nameToIndex = mutableMapOf<String, Int>()

    /** Returns the number of vertices. */
    fun vcount(): Int = vertices.size

    /** Returns the number of edges. */
    fun ecount(): Int = edges.size

    /** Returns the list of vertex names (if present). */
    fun vertexNames(): List<String> = vertices.mapNotNull { it["name"]?.toString() }

    /**
     * Adds vertices using columnar [attributes], where each list element is a vertex attribute value.
     */
    fun addVertices(attributes: Map<String, List<Any>>) {
        if (attributes.isEmpty()) return
        val count = attributes.values.first().size
        require(attributes.values.all { it.size == count }) {
            "All attribute lists must have the same length ($count)"
        }
        for (i in 0 until count) {
            val attr = mutableMapOf<String, Any>()
            for ((key, values) in attributes) {
                attr[key] = values[i]
            }
            val idx = vertices.size
            vertices.add(attr)
            val name = attr["name"]?.toString()
            if (name != null) {
                require(nameToIndex[name] == null) {
                    "Duplicate vertex name '$name' at index $idx"
                }
                nameToIndex[name] = idx
            }
        }
    }

    /**
     * Adds edges between named vertices with the supplied [weights].
     */
    fun addEdges(
        edgePairs: List<Pair<String, String>>,
        weights: List<Double>,
    ) {
        require(edgePairs.size == weights.size) {
            "edgePairs size (${edgePairs.size}) must match weights size (${weights.size})"
        }
        edgePairs.zip(weights).forEach { (pair, weight) ->
            val sourceIdx = nameToIndex[pair.first]
            val targetIdx = nameToIndex[pair.second]
            if (sourceIdx != null && targetIdx != null) {
                edges.add(Edge(sourceIdx, targetIdx, weight))
            } else {
                logger.warn {
                    "Skipping edge due to unknown vertex name(s): " +
                        "source=${pair.first}, target=${pair.second}"
                }
            }
        }
    }

    /**
     * Deletes vertices by name, removing associated edges.
     */
    fun deleteVertices(names: List<String>) {
        if (names.isEmpty()) return
        val removeSet = names.toSet()
        val survivingEdges =
            edges.mapNotNull { edge ->
                val sourceName = vertices.getOrNull(edge.source)?.get("name")?.toString()
                val targetName = vertices.getOrNull(edge.target)?.get("name")?.toString()
                if (sourceName == null || targetName == null) {
                    logger.warn {
                        "Dropping edge with unnamed vertex during deleteVertices: " +
                            "sourceIndex=${edge.source}, targetIndex=${edge.target}"
                    }
                    null
                } else if (sourceName in removeSet || targetName in removeSet) {
                    null
                } else {
                    Triple(sourceName, targetName, edge.weight)
                }
            }

        vertices.removeAll { it["name"]?.toString() in removeSet }
        edges.clear()
        nameToIndex.clear()
        vertices.forEachIndexed { idx, attrs ->
            val name = attrs["name"]?.toString()
            if (name != null) {
                nameToIndex[name] = idx
            }
        }

        for ((sourceName, targetName, weight) in survivingEdges) {
            val sourceIdx = nameToIndex[sourceName]
            val targetIdx = nameToIndex[targetName]
            if (sourceIdx != null && targetIdx != null) {
                edges.add(Edge(sourceIdx, targetIdx, weight))
            }
        }
    }

    /**
     * Computes personalized PageRank scores for all vertices.
     *
     * @param reset per-vertex reset probabilities (unnormalized).
     * @param damping damping factor.
     */
    fun personalizedPageRank(
        reset: DoubleArray,
        damping: Double,
    ): DoubleArray {
        val n = vertices.size
        if (n == 0) return DoubleArray(0)
        require(reset.size == n) { "reset size (${reset.size}) must match vertex count ($n)" }

        val resetSum = reset.sum()
        val resetProb = if (resetSum > 0) reset.map { it / resetSum }.toDoubleArray() else DoubleArray(n) { 1.0 / n }

        val adjacency = Array(n) { mutableListOf<Pair<Int, Double>>() }
        for (edge in edges) {
            adjacency[edge.source].add(edge.target to edge.weight)
            if (!directed) {
                adjacency[edge.target].add(edge.source to edge.weight)
            }
        }

        val outWeight = DoubleArray(n)
        for (i in 0 until n) {
            outWeight[i] = adjacency[i].sumOf { it.second }
        }

        var scores = DoubleArray(n) { 1.0 / n }
        val maxIter = 100
        val tol = 1e-6

        repeat(maxIter) {
            var danglingMass = 0.0
            for (i in 0 until n) {
                if (outWeight[i] == 0.0) danglingMass += scores[i]
            }
            val next =
                DoubleArray(n) {
                    (1.0 - damping) * resetProb[it] + damping * danglingMass * resetProb[it]
                }
            for (i in 0 until n) {
                val weightSum = outWeight[i]
                if (weightSum == 0.0) continue
                val contribution = damping * scores[i] / weightSum
                for ((j, w) in adjacency[i]) {
                    next[j] += contribution * w
                }
            }

            val delta = next.zip(scores).sumOf { (a, b) -> kotlin.math.abs(a - b) }
            scores = next
            if (delta < tol) return scores
        }

        return scores
    }

    /**
     * Serializes the graph into [file].
     */
    fun save(file: File) {
        val data =
            GraphData(
                directed = directed,
                vertices = vertices.map { it.toMap() },
                edges = edges.map { EdgeData(it.source, it.target, it.weight) },
            )
        val json = jsonWithDefaults { prettyPrint = false }
        file.writeText(json.encodeToString(GraphData.serializer(), data))
    }

    companion object {
        /**
         * Loads a [SimpleGraph] from [file].
         */
        fun load(file: File): SimpleGraph {
            val json = jsonWithDefaults { ignoreUnknownKeys = true }
            val data = json.decodeFromString(GraphData.serializer(), file.readText())
            val graph = SimpleGraph(data.directed)
            val attributes = mutableMapOf<String, MutableList<Any>>()
            if (data.vertices.isNotEmpty()) {
                for (vertex in data.vertices) {
                    val attr = vertex.toMutableMap()
                    val idx = graph.vertices.size
                    graph.vertices.add(attr)
                    val name = attr["name"]?.toString()
                    if (name != null) {
                        require(graph.nameToIndex[name] == null) {
                            "Duplicate vertex name '$name' at index $idx"
                        }
                        graph.nameToIndex[name] = idx
                    }
                }
            }
            if (data.edges.isNotEmpty()) {
                for (edge in data.edges) {
                    if (edge.source in 0 until graph.vertices.size &&
                        edge.target in 0 until graph.vertices.size
                    ) {
                        graph.edges.add(Edge(edge.source, edge.target, edge.weight))
                    }
                }
            }
            return graph
        }
    }

    private data class Edge(
        val source: Int,
        val target: Int,
        val weight: Double,
    )
}

/**
 * Serialized graph container.
 *
 * @property directed whether the graph is directed.
 * @property vertices vertex attribute maps.
 * @property edges edge list.
 */
@Serializable
data class GraphData(
    val directed: Boolean,
    val vertices: List<Map<String, @Contextual Any>>,
    val edges: List<EdgeData>,
)

/**
 * Serialized edge record.
 *
 * @property source source vertex index.
 * @property target target vertex index.
 * @property weight edge weight.
 */
@Serializable
data class EdgeData(
    val source: Int,
    val target: Int,
    val weight: Double,
)
