package ragas.testset.transforms

/**
 * Canonical property-name constants used by testset graph transforms.
 */
object PropertyNames {
    /** Property key for summary text extracted from node content. */
    const val SUMMARY_LLM_BASED = "summary_llm_based"

    /** Property key for regex-derived entity list. */
    const val ENTITIES_REGEX = "entities_regex"

    /** Property key for topic tag derived from lexical/embedding heuristics. */
    const val EMBEDDING_TOPIC_TAG = "embedding_topic_tag"
}
