package pathrag.storage

import pathrag.utils.relationshipVectorId

internal fun orderedRelationshipEndpoints(entry: Map.Entry<String, Map<String, Any?>>): Pair<String, String>? {
    val source = entry.value["src_id"] as? String ?: return null
    val target = entry.value["tgt_id"] as? String ?: return null
    return (source to target).takeIf { entry.key == relationshipVectorId(source, target) }
}

internal fun vectorMetadata(
    entry: Map.Entry<String, Map<String, Any?>>,
    fields: Set<String>,
): Map<String, Any?> {
    val metadata = entry.value.filterKeys { it in fields }
    val (source, target) = orderedRelationshipEndpoints(entry) ?: return metadata
    return metadata + mapOf("src_id" to source, "tgt_id" to target)
}
