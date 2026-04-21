package ragas.testset.transforms

import ragas.testset.graph.KnowledgeGraph
import ragas.testset.graph.Node
import ragas.testset.graph.Relationship

/**
 * Default node filter that includes every node.
 */
val defaultFilter: (Node) -> Boolean = { true }

/**
 * Base contract for graph transformations and execution-plan generation.
 */
interface BaseGraphTransformation {
    /** Stable transformation name used for logs and debugging. */
    val name: String

    /** Predicate used to select the nodes this transformation should consider. */
    val filterNodes: (Node) -> Boolean

    /**
     * Computes this transformation result from the provided graph.
     *
     * This method is side-effect free; graph mutation is applied by [generateExecutionPlan].
     *
     * @param kg Graph to inspect.
     * @return Transformation-specific result payload.
     */
    suspend fun transform(kg: KnowledgeGraph): Any?

    /**
     * Returns a shallow graph view containing only nodes accepted by the node filter and
     * relationships whose source and target are both in that node set.
     *
     * The returned graph copies node/relationship lists, but reuses original node and
     * relationship instances.
     *
     * @param kg Input graph.
     * @return Filtered shallow graph view.
     */
    fun filter(kg: KnowledgeGraph): KnowledgeGraph {
        val filteredNodes = kg.nodes.filter(filterNodes)
        val nodeIds = filteredNodes.map { node -> node.id }.toSet()
        val filteredRelationships =
            kg.relationships.filter { rel -> rel.sourceId in nodeIds && rel.targetId in nodeIds }
        return KnowledgeGraph(
            nodes = filteredNodes.toMutableList(),
            relationships = filteredRelationships.toMutableList(),
        )
    }

    /**
     * Produces deferred operations that execute this transformation against a graph.
     *
     * Returned tasks may be executed concurrently by the engine. Implementations should
     * snapshot or synchronize any shared mutable state they touch.
     *
     * @param kg Graph used to prepare execution tasks.
     * @return Deferred tasks that perform the transformation.
     */
    fun generateExecutionPlan(kg: KnowledgeGraph): List<suspend () -> Unit>
}

/**
 * Base transformation that extracts one property value from each selected node.
 *
 * @property name Transformation name.
 * @property filterNodes Node selection predicate.
 */
abstract class Extractor(
    override val name: String,
    override val filterNodes: (Node) -> Boolean = defaultFilter,
) : BaseGraphTransformation {
    /**
     * Returns the property key written by this extractor for a node.
     *
     * @param node Node to extract from.
     * @return Property key to write.
     */
    abstract fun propertyName(node: Node): String

    /**
     * Extracts the property value for [propertyName].
     *
     * @param node Node to extract from.
     * @return Property value to write.
     */
    abstract suspend fun extract(node: Node): String

    /**
     * Computes extraction results for filtered nodes.
     * @param kg Knowledge graph to transform or inspect.
     */
    override suspend fun transform(kg: KnowledgeGraph): List<Pair<Node, Pair<String, String>>> {
        val filtered = filter(kg)
        return filtered.nodes.map { node ->
            val key = propertyName(node)
            node to (key to extract(node))
        }
    }

    /**
     * Produces deferred operations that execute this transformation against a graph.
     * @param kg Knowledge graph to transform or inspect.
     */
    override fun generateExecutionPlan(kg: KnowledgeGraph): List<suspend () -> Unit> {
        val filtered = filter(kg)
        return filtered.nodes.map { node ->
            suspend {
                val key = propertyName(node)
                val shouldExtract =
                    synchronized(node) {
                        node.getProperty(key) == null
                    }
                if (shouldExtract) {
                    val value = extract(node)
                    synchronized(node) {
                        if (node.getProperty(key) == null) {
                            node.addProperty(key, value)
                        }
                    }
                }
            }
        }
    }
}

/**
 * Base transformation that splits selected nodes into derived nodes and relationships.
 *
 * @property name Transformation name.
 * @property filterNodes Node selection predicate.
 */
abstract class Splitter(
    override val name: String,
    override val filterNodes: (Node) -> Boolean = defaultFilter,
) : BaseGraphTransformation {
    /**
     * Splits one node into derived nodes and relationships.
     *
     * @param node Node to split.
     * @return Pair of created nodes and relationships.
     */
    abstract suspend fun split(node: Node): Pair<List<Node>, List<ragas.testset.graph.Relationship>>

    /**
     * Computes split outputs for filtered nodes.
     * @param kg Knowledge graph to transform or inspect.
     */
    override suspend fun transform(kg: KnowledgeGraph): Pair<List<Node>, List<ragas.testset.graph.Relationship>> {
        val filtered = filter(kg)
        val allNodes = mutableListOf<Node>()
        val allRelationships = mutableListOf<ragas.testset.graph.Relationship>()
        filtered.nodes.forEach { node ->
            val (nodes, relationships) = split(node)
            allNodes += nodes
            allRelationships += relationships
        }
        return allNodes to allRelationships
    }

    /**
     * Produces deferred operations that execute this transformation against a graph.
     * @param kg Knowledge graph to transform or inspect.
     */
    override fun generateExecutionPlan(kg: KnowledgeGraph): List<suspend () -> Unit> {
        val filtered = filter(kg)
        return filtered.nodes.map { node ->
            suspend {
                val (nodes, relationships) = split(node)
                synchronized(kg) {
                    kg.nodes += nodes
                    kg.relationships += relationships
                }
            }
        }
    }
}

/**
 * Base transformation that builds new relationships from the current graph state.
 *
 * @property name Transformation name.
 * @property filterNodes Node selection predicate.
 */
abstract class RelationshipBuilder(
    override val name: String,
    override val filterNodes: (Node) -> Boolean = defaultFilter,
) : BaseGraphTransformation {
    /**
     * Builds relationships to be added to the graph.
     *
     * Implementations must treat both graphs as read-only. When called from
     * [generateExecutionPlan], [kg] and [filtered] may be detached snapshot copies rather
     * than the live graph, so builders must not rely on object identity or mutate them.
     *
     * @param kg Full graph snapshot to inspect for context.
     * @param filtered Filtered snapshot used for candidate selection.
     * @return Relationships produced by this builder.
     */
    abstract suspend fun build(
        kg: KnowledgeGraph,
        filtered: KnowledgeGraph,
    ): List<Relationship>

    /**
     * Computes relationships to add based on the current graph state.
     * @param kg Knowledge graph to transform or inspect.
     */
    override suspend fun transform(kg: KnowledgeGraph): List<Relationship> {
        val filtered = filter(kg)
        return build(kg, filtered)
    }

    /**
     * Produces deferred operations that execute this transformation against a graph.
     *
     * The task calls [build] with snapshot copies to avoid observing concurrent graph
     * mutations while planning new relationships.
     *
     * @param kg Knowledge graph to transform or inspect.
     */
    override fun generateExecutionPlan(kg: KnowledgeGraph): List<suspend () -> Unit> {
        val snapshot =
            synchronized(kg) {
                KnowledgeGraph(
                    nodes =
                        kg.nodes
                            .map { node ->
                                Node(
                                    id = node.id,
                                    properties = node.properties.toMutableMap(),
                                    type = node.type,
                                )
                            }.toMutableList(),
                    relationships = kg.relationships.toMutableList(),
                )
            }
        val filtered = filter(snapshot)
        return listOf(
            suspend {
                val built = build(snapshot, filtered)
                synchronized(kg) {
                    built.forEach { relationship ->
                        if (!kg.hasRelationship(relationship)) {
                            kg.relationships += relationship
                        }
                    }
                }
            },
        )
    }
}

private fun KnowledgeGraph.hasRelationship(candidate: Relationship): Boolean =
    relationships.any { existing ->
        existing.type == candidate.type &&
            existing.sourceId == candidate.sourceId &&
            existing.targetId == candidate.targetId &&
            existing.bidirectional == candidate.bidirectional &&
            existing.properties == candidate.properties
    }
