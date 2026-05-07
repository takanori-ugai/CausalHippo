package shared.rag.unified

/**
 * Backend-agnostic query mode names.
 */
enum class UnifiedMode {
    BASIC,
    LOCAL,
    GLOBAL,
    HYBRID,
    DRIFT,
    NAIVE,
    BYPASS,
    GRAPH,
    DPR,
    CAUSAL,
}
