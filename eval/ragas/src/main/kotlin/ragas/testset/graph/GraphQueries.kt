package ragas.testset.graph

/**
 * Returns descendant nodes reachable through child relationships from a parent node.
 *
 * @param node Root parent node.
 * @param graph Graph to traverse.
 * @param level Maximum traversal depth in `child` edges (1 means direct children only).
 * @return Child nodes found within the requested depth.
 */
fun getChildNodes(
    node: Node,
    graph: KnowledgeGraph,
    level: Int = 1,
): List<Node> {
    val children = mutableListOf<Node>()
    val visited = mutableSetOf<String>()

    /**
     * Depth-first traversal helper for relationship graph walk.
     *
     * @param currentNode Current node in traversal.
     * @param currentLevel Current traversal depth.
     */
    fun dfs(
        currentNode: Node,
        currentLevel: Int,
    ) {
        if (currentLevel > level) {
            return
        }

        graph.relationships
            .filter { rel -> rel.type == "child" && rel.sourceId == currentNode.id }
            .forEach { rel ->
                val child = graph.getNodeById(rel.targetId) ?: return@forEach
                if (!visited.add(child.id)) {
                    return@forEach
                }
                children += child
                dfs(child, currentLevel + 1)
            }
    }

    dfs(node, 1)
    return children
}

/**
 * Returns ancestor nodes reachable through child relationships from a child node.
 *
 * @param node Root child node.
 * @param graph Graph to traverse.
 * @param level Maximum traversal depth in reverse `child` edges (1 means direct parents only).
 * @return Parent nodes found within the requested depth.
 */
fun getParentNodes(
    node: Node,
    graph: KnowledgeGraph,
    level: Int = 1,
): List<Node> {
    val parents = mutableListOf<Node>()
    val visited = mutableSetOf<String>()

    /**
     * Depth-first traversal helper for relationship graph walk.
     *
     * @param currentNode Current node in traversal.
     * @param currentLevel Current traversal depth.
     */
    fun dfs(
        currentNode: Node,
        currentLevel: Int,
    ) {
        if (currentLevel > level) {
            return
        }

        graph.relationships
            .filter { rel -> rel.type == "child" && rel.targetId == currentNode.id }
            .forEach { rel ->
                val parent = graph.getNodeById(rel.sourceId) ?: return@forEach
                if (!visited.add(parent.id)) {
                    return@forEach
                }
                parents += parent
                dfs(parent, currentLevel + 1)
            }
    }

    dfs(node, 1)
    return parents
}
