package com.youtu.graphrag.shared.graph

import com.fasterxml.jackson.annotation.JsonProperty

data class GraphNode(
    val label: String,
    val properties: Map<String, Any?>,
)

data class GraphRelationship(
    @field:JsonProperty("start_node")
    val startNode: GraphNode,
    val relation: String,
    @field:JsonProperty("end_node")
    val endNode: GraphNode,
)
