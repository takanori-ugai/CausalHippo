package ragas.testset.transforms

import ragas.testset.graph.KnowledgeGraph
import ragas.testset.graph.Node
import ragas.testset.graph.NodeType
import ragas.testset.graph.Relationship

/**
 * Relationship builder that links consecutive chunks from the same parent document.
 *
 * @property name Transformation name.
 * @property filterNodes Node selection predicate for document nodes.
 */
class AdjacentChunkRelationshipBuilder(
    override val name: String = "adjacent_chunk_relationship_builder",
    override val filterNodes: (Node) -> Boolean = { node -> node.type == NodeType.DOCUMENT },
) : RelationshipBuilder(name = name, filterNodes = filterNodes) {
    /**
     * Builds relationships from the current graph based on this builder's strategy.
     *
     * @param kg Full graph to inspect.
     * @param filtered Graph filtered by [filterNodes] used for document candidate selection.
     * @return Generated `next` relationships between adjacent chunks.
     */
    override suspend fun build(
        kg: KnowledgeGraph,
        filtered: KnowledgeGraph,
    ): List<Relationship> {
        val documentIds =
            filtered.nodes
                .map { node -> node.id }
                .toSet()
        if (documentIds.isEmpty()) {
            return emptyList()
        }

        val chunkNodesById =
            kg.nodes
                .filter { node -> node.type == NodeType.CHUNK }
                .associateBy { node -> node.id }

        val byDocument =
            kg.relationships
                .asSequence()
                .filter { rel -> rel.type == "child" && rel.sourceId in documentIds }
                .mapNotNull { rel ->
                    val chunk = chunkNodesById[rel.targetId] ?: return@mapNotNull null
                    val index = chunk.getProperty("chunk_index")?.toIntOrNull() ?: Int.MAX_VALUE
                    Triple(rel.sourceId, index, chunk)
                }.groupBy { it.first }

        return byDocument.values.flatMap { triples ->
            val sorted = triples.sortedBy { it.second }.map { it.third }
            sorted
                .zipWithNext()
                .map { (left, right) ->
                    Relationship(
                        type = "next",
                        sourceId = left.id,
                        targetId = right.id,
                    )
                }
        }
    }
}

/**
 * Relationship builder that connects nodes sharing enough normalized keywords.
 *
 * @property name Transformation name.
 * @property filterNodes Node selection predicate for candidate nodes.
 * @property sourceProperty Source property key.
 * @property minSharedKeywords Minimum shared keywords threshold.
 */
class SharedKeywordRelationshipBuilder(
    override val name: String = "shared_keyword_relationship_builder",
    override val filterNodes: (Node) -> Boolean = { node -> node.type == NodeType.CHUNK },
    private val sourceProperty: String = "page_content",
    private val minSharedKeywords: Int = 2,
) : RelationshipBuilder(name = name, filterNodes = filterNodes) {
    init {
        require(minSharedKeywords > 0) { "minSharedKeywords must be > 0" }
    }

    /**
     * Builds relationships from the current graph based on this builder's strategy.
     *
     * @param kg Full graph to inspect.
     * @param filtered Graph filtered by [filterNodes] used for candidate node selection.
     * @return Generated `semantic_overlap` relationships.
     */
    override suspend fun build(
        kg: KnowledgeGraph,
        filtered: KnowledgeGraph,
    ): List<Relationship> {
        val nodes = filtered.nodes
        if (nodes.size < 2) {
            return emptyList()
        }

        val keywordsByNodeId =
            nodes.associate { node ->
                val keywords =
                    node
                        .getProperty(sourceProperty)
                        .orEmpty()
                        .lowercase()
                        .split(Regex("[^a-z0-9]+"))
                        .filter { token -> token.length >= 4 }
                        .toSet()
                node.id to keywords
            }

        val relationships = mutableListOf<Relationship>()
        for (leftIndex in 0 until nodes.size - 1) {
            val left = nodes[leftIndex]
            val leftKeywords = keywordsByNodeId[left.id].orEmpty()
            if (leftKeywords.isEmpty()) {
                continue
            }
            for (rightIndex in leftIndex + 1 until nodes.size) {
                val right = nodes[rightIndex]
                val shared = leftKeywords.intersect(keywordsByNodeId[right.id].orEmpty())
                if (shared.size >= minSharedKeywords) {
                    relationships +=
                        Relationship(
                            type = "semantic_overlap",
                            sourceId = left.id,
                            targetId = right.id,
                            properties =
                                mapOf(
                                    "shared_keyword_count" to shared.size.toString(),
                                ),
                        )
                }
            }
        }

        return relationships
    }
}
