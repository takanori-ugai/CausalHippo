package causalrag.causalgraph.retriever

import causalrag.causalgraph.builder.CausalGraphBuilder
import causalrag.causalgraph.graph.DirectedGraph
import causalrag.utils.EmbeddingModel
import causalrag.utils.cosineSimilarity
import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

/**
 * Retrieves graph nodes and paths relevant to a natural-language query.
 *
 * @param builder Causal graph builder that provides graph structure, node text, and embeddings.
 * @param similarNodeThreshold Similarity threshold for adding semantically related nodes during path expansion.
 */
@Suppress("TooGenericExceptionCaught")
class CausalPathRetriever(
    private val builder: CausalGraphBuilder,
    private val similarNodeThreshold: Double = 0.8,
) {
    private val graph: DirectedGraph
        get() = builder.getGraph()
    private val nodeText = builder.nodeText
    private val encoder: EmbeddingModel? = builder.getEncoder()

    /**
     * Returns graph nodes that are semantically similar to the query.
     *
     * @param query User query.
     * @param topK Maximum number of nodes to return.
     * @param threshold Minimum similarity threshold.
     * @return Node identifiers paired with similarity scores.
     */
    fun retrieveNodes(
        query: String,
        topK: Int = 5,
        threshold: Double = 0.5,
    ): List<Pair<String, Double>> = retrieveNodes(query, topK, threshold, builder.nodeEmbeddings)

    private fun retrieveNodes(
        query: String,
        topK: Int,
        threshold: Double,
        nodeEmbeddings: Map<String, DoubleArray>,
    ): List<Pair<String, Double>> {
        if (encoder == null || nodeEmbeddings.isEmpty()) {
            logger.warn { "Encoder or node embeddings not available" }
            return emptyList()
        }
        return try {
            val qEmb = encoder.encode(query)
            val scores = mutableMapOf<String, Double>()
            for ((nodeId, emb) in nodeEmbeddings) {
                val sim = cosineSimilarity(qEmb, emb)
                if (sim >= threshold) {
                    scores[nodeId] = sim
                }
            }
            scores.entries
                .sortedByDescending { it.value }
                .take(topK)
                .map { it.key to it.value }
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error retrieving nodes" }
            emptyList()
        }
    }

    /**
     * Expands the query into a set of relevant causal nodes.
     *
     * @param query User query.
     * @param topK Maximum number of seed nodes to consider.
     * @param maxHops Number of ancestor or descendant hops to include.
     * @param includeSimilar Whether to include semantically similar nodes.
     * @return Relevant node identifiers.
     */
    fun retrievePathNodes(
        query: String,
        topK: Int = 5,
        maxHops: Int = 2,
        includeSimilar: Boolean = true,
    ): List<String> {
        val embeddings = builder.nodeEmbeddings
        val seeds = retrieveNodes(query, topK, 0.5, embeddings).map { it.first }
        return expandPathNodes(graph, seeds, maxHops, if (includeSimilar) embeddings else emptyMap(), similarNodeThreshold)
    }

    /**
     * Finds causal paths between query-relevant nodes.
     *
     * @param query User query.
     * @param maxPaths Maximum number of paths to return.
     * @param minPathLength Minimum number of nodes in a returned path.
     * @param maxPathLength Maximum search depth.
     * @return Ordered node labels for each path.
     */
    fun retrievePaths(
        query: String,
        maxPaths: Int = 5,
        minPathLength: Int = 2,
        maxPathLength: Int = 4,
    ): List<List<String>> {
        val relevantNodes = retrievePathNodes(query, topK = 5, maxHops = 1)
        return graph
            .shortestRelevantPaths(relevantNodes, maxPaths, minPathLength, maxPathLength)
            .map { path -> path.map { nodeText[it] ?: it } }
    }

    /**
     * Builds a short textual explanation of the causal paths relevant to a query.
     *
     * @param query User query.
     * @return Human-readable causal explanation.
     */
    fun getCausalExplanation(query: String): String {
        val paths = retrievePaths(query, maxPaths = 3)
        if (paths.isEmpty()) return "No relevant causal relationships found."
        val builder = StringBuilder()
        builder.append("Causal relationships relevant to '").append(query).append("':\n\n")
        for ((idx, path) in paths.withIndex()) {
            builder
                .append(idx + 1)
                .append(". ")
                .append(path.joinToString(" -> "))
                .append("\n")
        }
        return builder.toString()
    }

    /**
     * Extracts a subgraph containing query-relevant nodes.
     *
     * @param query User query.
     * @return Subgraph induced by the relevant nodes.
     */
    fun highlightSubgraph(query: String): DirectedGraph {
        val nodes = retrievePathNodes(query)
        return graph.subgraph(nodes.toSet())
    }
}

/** Expands causes and effects separately, then admits nodes similar to the mean seed embedding. */
internal fun expandPathNodes(
    graph: DirectedGraph,
    seeds: List<String>,
    maxHops: Int,
    embeddings: Map<String, DoubleArray>,
    similarThreshold: Double = 0.8,
): List<String> {
    val nodes = linkedSetOf<String>()
    for (seed in seeds) {
        nodes.add(seed)
        for (neighbors in listOf(graph::successors, graph::predecessors)) {
            var frontier = setOf(seed)
            val seen = mutableSetOf(seed)
            for (hop in 0 until maxHops) {
                frontier = frontier.flatMap(neighbors).filterTo(mutableSetOf()) { seen.add(it) }
                if (frontier.isEmpty()) break
                nodes.addAll(frontier)
            }
        }
    }
    val vectors = seeds.mapNotNull(embeddings::get)
    if (vectors.isNotEmpty()) {
        val mean = DoubleArray(vectors.first().size) { i -> vectors.sumOf { it[i] } / vectors.size }
        for ((node, vector) in embeddings) {
            if (node !in nodes && cosineSimilarity(mean, vector) > similarThreshold) nodes.add(node)
        }
    }
    return nodes.toList()
}
