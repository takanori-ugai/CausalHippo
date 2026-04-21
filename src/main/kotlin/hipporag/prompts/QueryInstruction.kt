package hipporag.prompts

private const val DEFAULT_INSTRUCTION = "Given a question, retrieve relevant documents that best answer the question."

private val instructions =
    mapOf(
        "ner_to_node" to "Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.",
        "query_to_node" to "Given a question, retrieve relevant phrases that are mentioned in this question.",
        "query_to_fact" to "Given a question, retrieve relevant triplet facts that match this question.",
        "query_to_sentence" to "Given a question, retrieve relevant sentences that best answer the question.",
        "query_to_passage" to DEFAULT_INSTRUCTION,
    )

/**
 * Returns the instruction string for a query embedding variant.
 */
fun getQueryInstruction(name: String): String = instructions[name] ?: DEFAULT_INSTRUCTION
