package shared.rag.unified

internal fun UnifiedQuery.coercedTopK(): Int = topK.coerceAtLeast(1)

internal fun UnifiedQuery.intExtra(
    key: String,
    default: Int,
): Int =
    when (val value = extras[key]) {
        is Int -> value
        is Number -> value.toInt()
        is String -> value.toIntOrNull() ?: default
        else -> default
    }

internal fun UnifiedQuery.stringExtra(
    key: String,
    default: String? = null,
): String? =
    when (val value = extras[key]) {
        is String -> value
        else -> default
    }

internal fun UnifiedQuery.listStringExtra(key: String): List<String> {
    val value = extras[key] ?: return emptyList()
    return when (value) {
        is List<*> -> value.mapNotNull { it?.toString() }
        else -> emptyList()
    }
}

internal fun collectUnsupported(
    query: UnifiedQuery,
    capabilities: RagCapabilities,
): List<String> {
    val unsupported = mutableListOf<String>()
    if (query.mode !in capabilities.supportedModes) {
        unsupported += "mode:${query.mode}"
    }
    if (query.streaming && !capabilities.supportsStreaming) {
        unsupported += "streaming"
    }
    if (query.includeGraphPaths && !capabilities.supportsGraphPaths) {
        unsupported += "graphPaths"
    }
    if (query.includeFollowUps && !capabilities.supportsFollowUpQueries) {
        unsupported += "followUps"
    }
    if (query.includeReferences && !capabilities.supportsReferences) {
        unsupported += "references"
    }
    return unsupported
}

internal fun buildMetadata(
    ragId: RagId,
    modeUsed: String,
    unsupported: List<String> = emptyList(),
    extra: Map<String, Any?> = emptyMap(),
): Map<String, Any?> {
    val metadata =
        mutableMapOf<String, Any?>(
            "ragId" to ragId.name,
            "modeUsed" to modeUsed,
        )
    if (unsupported.isNotEmpty()) {
        metadata["unsupported"] = unsupported
    }
    metadata.putAll(extra)
    return metadata
}

internal fun textContextItems(items: List<String>): List<ContextItem> =
    items
        .map { it.trim() }
        .filter { it.isNotBlank() }
        .mapIndexed { index, text ->
            ContextItem(
                id = index.toString(),
                text = text,
            )
        }

internal fun textReferences(items: List<String>): List<ReferenceItem> =
    items
        .map { it.trim() }
        .filter { it.isNotBlank() }
        .mapIndexed { index, text ->
            ReferenceItem(
                id = index.toString(),
                snippet = text,
            )
        }

internal fun asStringMap(input: Any?): Map<String, Any?> {
    val map = input as? Map<*, *> ?: return emptyMap()
    return map.entries.associate { (k, v) -> k.toString() to v }
}

internal fun asStringList(input: Any?): List<String> =
    when (input) {
        is List<*> -> input.mapNotNull { it?.toString() }
        else -> emptyList()
    }

internal fun normalizeGraphInspection(payload: Map<String, Any?>): Map<String, Any?> {
    val directNodes = normalizeNodeRows(payload["nodes"])
    val directEdges = normalizeEdgeRows(payload["edges"])

    if (directNodes.isNotEmpty() || directEdges.isNotEmpty()) {
        val metadata = mergeGraphMetadata(payload["metadata"], directNodes.size, directEdges.size)
        return mapOf(
            "nodes" to directNodes,
            "edges" to directEdges,
            "metadata" to metadata,
        )
    }

    val causalGraph = asStringMap(payload["causal_graph"])
    val hippoGraph = asStringMap(payload["hippo_graph"])

    if (causalGraph.isNotEmpty() || hippoGraph.isNotEmpty()) {
        val causalNodes = normalizeNodeRows(causalGraph["nodes"]).map { row -> row + mapOf("graph" to "causal_graph") }
        val causalEdges = normalizeEdgeRows(causalGraph["edges"]).map { row -> row + mapOf("graph" to "causal_graph") }
        val hippoNodes = normalizeNodeRows(hippoGraph["nodes"]).map { row -> row + mapOf("graph" to "hippo_graph") }
        val hippoEdges = normalizeEdgeRows(hippoGraph["edges"]).map { row -> row + mapOf("graph" to "hippo_graph") }
        val nodes = causalNodes + hippoNodes
        val edges = causalEdges + hippoEdges
        val metadata = mergeGraphMetadata(payload["metadata"], nodes.size, edges.size)
        return mapOf(
            "nodes" to nodes,
            "edges" to edges,
            "metadata" to metadata,
        )
    }

    return mapOf(
        "nodes" to emptyList<Map<String, Any?>>(),
        "edges" to emptyList<Map<String, Any?>>(),
        "metadata" to mergeGraphMetadata(payload["metadata"], 0, 0),
    )
}

private fun normalizeNodeRows(rawNodes: Any?): List<Map<String, Any?>> {
    val asMap = rawNodes as? Map<*, *>
    if (asMap != null) {
        return asMap.entries.map { (id, value) ->
            val row = asStringMap(value)
            if (row.containsKey("id")) row else row + mapOf("id" to id.toString())
        }
    }

    val asList = rawNodes as? List<*> ?: return emptyList()
    return asList.mapIndexedNotNull { index, value ->
        val row = asStringMap(value)
        when {
            row.isNotEmpty() -> if (row.containsKey("id")) row else row + mapOf("id" to index.toString())
            value != null -> mapOf("id" to index.toString(), "value" to value.toString())
            else -> null
        }
    }
}

private fun normalizeEdgeRows(rawEdges: Any?): List<Map<String, Any?>> {
    val asList = rawEdges as? List<*> ?: return emptyList()
    return asList.mapIndexedNotNull { index, value ->
        val row = asStringMap(value)
        when {
            row.isNotEmpty() -> row
            value != null -> mapOf("id" to index.toString(), "value" to value.toString())
            else -> null
        }
    }
}

private fun mergeGraphMetadata(
    rawMetadata: Any?,
    nodeCount: Int,
    edgeCount: Int,
): Map<String, Any?> {
    val metadata = asStringMap(rawMetadata).toMutableMap()
    metadata.putIfAbsent("nodeCount", nodeCount)
    metadata.putIfAbsent("edgeCount", edgeCount)
    return metadata
}
